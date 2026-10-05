"""Explanation agent: LLM (tool calling) explains existing holdings; verifier keeps only quotes found in filings.
Explanations only - never feeds the signal or portfolio weights."""
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.text import _txt_anonymise

_AGENT_NOTE = ("explanations only; not used in portfolio construction; LLM pretraining may postdate the "
               "formation date, so explanations are post-hoc")
_AGENT_SYSTEM = (
    "You are a buy-side analyst. Explain why the systematic model is long/short this stock at this date using "
    "ONLY tool outputs (get_filings, get_profile, get_score). The company is anonymised; do not guess its identity. "
    " Every fact must carry a verbatim quote from a filing "
    "plus its document_id. Use no knowledge after the formation date. Reply with strict JSON: "
    '{"thesis": str, "facts": [{"fact": str, "quote": str, "document_id": str}], "risks": [str], "confidence": 0-1}.')
_AGENT_TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {"permno": {"type": "integer"}, p: {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["permno", p]}}}
    for n, d, p in [("get_filings", "Recent 8-K filings filed on/before before_eom", "before_eom"),
                    ("get_profile", "Known characteristics at month-end eom", "eom"),
                    ("get_score", "Model score and within-month percentile at eom", "eom")]]


def _agent_date(x):
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _agent_cfg(cfg):
    return {"url": cfg.get("agent_url", "http://localhost:11434/v1/chat/completions"),
            "model": cfg.get("agent_model", "gpt-oss:20b"), "steps": int(cfg.get("agent_max_steps", 4)),
            "top_n": int(cfg.get("agent_top_n", 10)), "max_filings": int(cfg.get("agent_max_filings", 5)),
            "max_chars": int(cfg.get("agent_max_chars", 6000))}


def get_filings(cfg, permno, before_eom):
    """8-K filings for permno with filing_date <= before_eom (look-ahead control), latest first."""
    a = _agent_cfg(cfg)
    d = _agent_date(before_eom)
    df = (pl.scan_parquet(cfg["data_files"]["8k"]).filter(pl.col("permno") == int(permno))
          .select(["document_id", "permno", "filing_date", "items", "text"]
                  + [c for c in ("company_name", "ticker") if c in pl.read_parquet_schema(cfg["data_files"]["8k"])])
          .with_columns(pl.col("filing_date").cast(pl.Date))
          .filter(pl.col("filing_date") <= d).sort("filing_date", descending=True).head(a["max_filings"]).collect())
    # anonymise as in step 7 (firm names case-insensitive, tickers case-sensitive) so the LLM cannot recall what
    # happened to the firm later; names from the filings and from the panel for this permno
    names, ticks = set(), set()
    for path in (cfg["data_files"]["8k"], cfg["data_files"]["chars"]):
        sch = pl.read_parquet_schema(path)
        cols = [c for c in ("company_name", "ticker") if c in sch]
        if cols:
            pn = pl.scan_parquet(path).filter(pl.col("permno") == int(permno)).select(cols).unique().collect()
            names |= set(pn["company_name"].drop_nulls()) if "company_name" in cols else set()
            ticks |= set(pn["ticker"].drop_nulls()) if "ticker" in cols else set()
    return [{"document_id": str(r["document_id"]), "filing_date": str(r["filing_date"]), "items": str(r["items"]),
             "text": _txt_anonymise((r["text"] or "")[:a["max_chars"]], names, ticks)} for r in df.to_dicts()]


def get_profile(cfg, permno, eom):
    want = ["gics", "market_equity", "be_me", "ret_12_1", "beta_60m", "ret_exc"]   # no ticker/name: anonymised
    path = cfg["data_files"]["chars"]
    have = [c for c in want if c in pl.read_parquet_schema(path)]
    df = (pl.scan_parquet(path).filter(pl.col("permno") == int(permno))
          .select(["permno", "eom"] + have).with_columns(pl.col("eom").cast(pl.Date))
          .filter(pl.col("eom") == _agent_date(eom)).collect())
    if df.height == 0:
        return {}
    return {k: (str(v) if k in ("ticker", "company_name", "gics") else v) for k, v in df.select(have).to_dicts()[0].items()}


def get_score(cfg, permno, eom):
    p = Path(cfg["output_dir"]) / "predictions.parquet"
    if not p.exists():
        return {}
    df = (pl.scan_parquet(p).select(["permno", "eom", "score"]).with_columns(pl.col("eom").cast(pl.Date))
          .filter(pl.col("eom") == _agent_date(eom)).collect())
    if df.height == 0:
        return {}
    df = df.with_columns(((pl.col("score").rank() - 1) / max(df.height - 1, 1)).alias("pct"))
    r = df.filter(pl.col("permno") == int(permno))
    return {"score": r["score"][0], "percentile": r["pct"][0], "n_stocks": df.height} if r.height else {}


def _agent_dispatch(cfg, name, args, permno, formation_eom):
    """LLM args are untrusted: permno is forced to the held one, dates are capped at formation_eom."""
    cap = _agent_date(formation_eom)
    name = next((n for n in ("get_filings", "get_profile", "get_score") if str(name).startswith(n.rstrip("s"))), name)
    key = "before_eom" if name == "get_filings" else "eom"
    try:
        d = min(_agent_date(args.get(key)), cap)
    except Exception:
        d = cap
    if name == "get_filings":
        return get_filings(cfg, permno, d)
    if name == "get_profile":
        return get_profile(cfg, permno, d)
    if name == "get_score":
        return get_score(cfg, permno, d)
    return {"error": f"unknown tool {name}"}


def _agent_conn_err(e):
    import socket
    import urllib.error
    try:
        import requests
        if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
    except ImportError:
        pass
    return isinstance(e, (urllib.error.URLError, socket.timeout, ConnectionError))


def _agent_post(cfg, payload):
    url = _agent_cfg(cfg)["url"]
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        r = requests.post(url, json=payload, timeout=(5, 120))
        r.raise_for_status()
        return r.json()
    import urllib.request
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _agent_parse(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"thesis": _agent_thesis(text), "facts": [], "risks": [], "confidence": 0.0}


def _agent_thesis(text):
    """Unparseable/truncated JSON: pull out the thesis string if present, else the raw text."""
    t = str(text or "")
    m = re.search(r'"thesis"\s*:\s*"((?:[^"\\]|\\.)*)', t, re.S)
    return (m.group(1).replace('\\"', '"') if m else t)[:800]


def _agent_norm(s):
    s = str(s or "").translate({0x2018: 39, 0x2019: 39, 0x201A: 39, 0x2032: 39, 0x201C: 34, 0x201D: 34, 0x201E: 34,
                                0x2010: 45, 0x2011: 45, 0x2012: 45, 0x2013: 45, 0x2014: 45, 0x2015: 45, 0x2212: 45,
                                0xA0: 32, 0x202F: 32, 0x2009: 32, 0x2026: 46})   # LLMs emit typographic variants
    return re.sub(r"\s+", " ", s).strip().lower()


def _agent_verify(facts, filings, formation_eom=None):
    """Keep a fact only if its quote occurs in the cited filing text (or any fetched filing if id missing).
    Only filings with filing_date <= formation_eom are eligible."""
    if formation_eom is not None:
        cap = _agent_date(formation_eom)
        filings = [f for f in filings if _agent_date(f["filing_date"]) <= cap]
    by_id = {str(f["document_id"]): _agent_norm(f["text"]) for f in filings}
    kept, dropped = [], []
    for f in facts if isinstance(facts, list) else []:
        q = _agent_norm(f.get("quote")) if isinstance(f, dict) else ""
        did = str(f.get("document_id", "") or "") if isinstance(f, dict) else ""
        pool = [by_id[did]] if did in by_id else (list(by_id.values()) if not did else [])
        (kept if q and any(q in t for t in pool) else dropped).append(f)
    return kept, dropped


def _agent_explain(cfg, permno, eom, side):
    a = _agent_cfg(cfg)
    ask = f"Stock permno={permno}, side={side.upper()}, formation month-end {eom}. Explain the position."
    msgs = [{"role": "system", "content": _AGENT_SYSTEM}, {"role": "user", "content": ask}]
    calls, filings, steps, content = [], [], 0, None
    try:
        for _ in range(a["steps"]):
            msg = _agent_post(cfg, {"model": a["model"], "messages": msgs, "tools": _AGENT_TOOLS})["choices"][0]["message"]
            steps += 1
            tcs = msg.get("tool_calls") or []
            if not tcs:
                content = msg.get("content")
                break
            msgs.append(msg)
            for tc in tcs:
                fn = tc["function"]
                args = fn["arguments"]
                args = json.loads(args) if isinstance(args, str) else args
                out = _agent_dispatch(cfg, fn["name"], args, permno, eom)
                if fn["name"] == "get_filings":
                    filings += out
                calls.append({"name": fn["name"], "args": args})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(out, default=str)})
        if content is None:  # step budget used up: force an answer
            msgs.append({"role": "user", "content": "Give the final JSON now."})
            content = _agent_post(cfg, {"model": a["model"], "messages": msgs})["choices"][0]["message"].get("content")
            steps += 1
    except Exception as e:
        if _agent_conn_err(e):
            raise
        # no tool-calling support: prefetch all tools, ask once
        print(f"[agent] tool mode failed ({type(e).__name__}); prefetch fallback")
        filings = get_filings(cfg, permno, eom)
        calls = [{"name": "get_filings", "args": {"permno": permno, "before_eom": eom}},
                 {"name": "get_profile", "args": {"permno": permno, "eom": eom}},
                 {"name": "get_score", "args": {"permno": permno, "eom": eom}}]
        ctx = {"get_filings": filings, "get_profile": get_profile(cfg, permno, eom), "get_score": get_score(cfg, permno, eom)}
        m2 = [{"role": "system", "content": _AGENT_SYSTEM},
              {"role": "user", "content": ask + "\nTool outputs:\n" + json.dumps(ctx, default=str)}]
        content = _agent_post(cfg, {"model": a["model"], "messages": m2})["choices"][0]["message"].get("content")
        steps = 1
    res = _agent_parse(content)
    kept, dropped = _agent_verify(res.get("facts"), filings, eom)
    return {"thesis": res.get("thesis", ""), "facts": kept, "dropped_facts": dropped, "risks": res.get("risks", []),
            "confidence": res.get("confidence", 0.0), "tool_calls": calls, "steps": steps}


def _agent_pick(hold_path, top_n):
    h = pl.read_csv(hold_path, try_parse_dates=True).with_columns(pl.col("Date").cast(pl.Date))
    h = h.with_columns(pl.col("WEIGHT" if "WEIGHT" in h.columns else "WEIGHT %").cast(pl.Float64).alias("w"))
    g = h.group_by("PERMNO").agg(pl.col("w").abs().mean().alias("avg_abs"), pl.col("w").mean().alias("avg_w"),
                                 pl.col("TICKER").first(), pl.col("COMPANY NAME").first())
    out = []
    for side, sub in (("long", g.filter(pl.col("avg_w") > 0)), ("short", g.filter(pl.col("avg_w") < 0))):
        for r in sub.sort("avg_abs", descending=True).head(top_n).to_dicts():
            peak = h.filter(pl.col("PERMNO") == r["PERMNO"]).sort(pl.col("w").abs(), descending=True)["Date"][0]
            out.append({"permno": int(r["PERMNO"]), "ticker": r["TICKER"], "name": r["COMPANY NAME"], "side": side,
                        "avg_weight": r["avg_w"], "formation_eom": str(peak.replace(day=1) - timedelta(days=1))})
    return out


def run_agent(cfg):
    a = _agent_cfg(cfg)
    out_dir, cache_dir = Path(cfg["output_dir"]), Path(cfg["cache_dir"])
    cache_p = cache_dir / "agent_cache.json"
    try:
        cache = json.loads(cache_p.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    rows = []
    for h in _agent_pick(out_dir / "holdings.csv", a["top_n"]):
        key = f"{h['permno']}|{h['formation_eom']}|{a['model']}"
        if key not in cache:
            try:
                cache[key] = _agent_explain(cfg, h["permno"], h["formation_eom"], h["side"])
            except Exception as e:
                if _agent_conn_err(e):
                    print(f"[agent] LLM endpoint unreachable ({type(e).__name__}); stopping agent run")
                    break
                print(f"[agent] {h['ticker']} failed: {e}")
                continue
        r = {**h, **cache[key]}
        if str(r.get("thesis", "")).lstrip().startswith("{"):
            r["thesis"] = _agent_thesis(r["thesis"])
        rows.append(r)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(cache, default=str), encoding="utf-8")
    kept = sum(len(r["facts"]) for r in rows)
    drop = sum(len(r["dropped_facts"]) for r in rows)
    summ = {"n_holdings": len(rows), "verification_rate": kept / (kept + drop) if kept + drop else None,
            "verified_facts": kept, "dropped_facts": drop, "model": a["model"],
            "timestamp": datetime.now().isoformat(timespec="seconds"), "note": _AGENT_NOTE}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agent_rationales.json").write_text(
        json.dumps({"summary": summ, "note": _AGENT_NOTE, "holdings": rows}, indent=2, default=str), encoding="utf-8")
    print(f"[agent] {len(rows)} holdings explained; verified {kept}/{kept + drop} facts; model {a['model']}")
    return rows
