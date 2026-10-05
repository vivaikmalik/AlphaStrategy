import json
from datetime import date

import polars as pl

import src.agent as ag

TEXT = "The Company announced record revenue growth.\nGuidance was “raised” for FY."


def _cfg(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    pl.DataFrame({
        "document_id": ["d1", "d2"], "permno": [1, 1], "filing_date": [date(2020, 1, 15), date(2020, 3, 10)],
        "items": ["2.02", "8.01"], "text": [TEXT, "later filing"]}).write_parquet(d / "8k.parquet")
    pl.DataFrame({"permno": [1], "eom": [date(2020, 1, 31)], "ticker": ["AAA"], "company_name": ["Aaa Inc"],
                  "gics": ["10"], "market_equity": [1.0], "be_me": [0.5], "ret_12_1": [0.1], "beta_60m": [1.0],
                  "ret_exc": [0.01], "ret_exc_lead1m": [0.99]}).write_parquet(d / "chars.parquet")
    out = tmp_path / "output"
    out.mkdir()
    pl.DataFrame({"permno": [1, 2], "eom": [date(2020, 1, 31)] * 2, "score": [0.9, 0.1],
                  "ret_hat": [0.0, 0.0]}).write_parquet(out / "predictions.parquet")
    (out / "holdings.csv").write_text(
        "Date,PERMNO,TICKER,COMPANY NAME,WEIGHT\n2020-02-29,1,AAA,Aaa Inc,5.0\n2020-03-31,1,AAA,Aaa Inc,3.0\n"
        "2020-02-29,2,BBB,Bbb Inc,-4.0\n")
    return {"data_files": {"8k": d / "8k.parquet", "chars": d / "chars.parquet"}, "output_dir": out,
            "cache_dir": tmp_path / "cache", "agent_top_n": 2}


def test_no_lookahead(tmp_path):
    cfg = _cfg(tmp_path)
    r = ag.get_filings(cfg, 1, "2020-01-31")
    assert [f["document_id"] for f in r] == ["d1"]
    assert all(date.fromisoformat(f["filing_date"]) <= date(2020, 2, 15) for f in ag.get_filings(cfg, 1, "2020-02-15"))
    assert "ret_exc_lead1m" not in ag.get_profile(cfg, 1, "2020-01-31")


def test_verifier():
    fl = [{"document_id": "d1", "text": TEXT}]
    facts = [{"quote": "record revenue growth", "document_id": "d1"},
             {"quote": 'Guidance was "raised" for  FY.', "document_id": "d1"},
             {"quote": "record\nrevenue growth", "document_id": ""},
             {"quote": "profits tripled", "document_id": "d1"},
             {"quote": "record revenue growth", "document_id": "zzz"}]
    kept, dropped = ag._agent_verify(facts, fl)
    assert len(kept) == 3 and len(dropped) == 2


def test_run_agent_mocked(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    answer = {"thesis": "t", "facts": [{"fact": "f", "quote": "record revenue growth", "document_id": "d1"},
                                       {"fact": "x", "quote": "made up", "document_id": "d1"}],
              "risks": ["r"], "confidence": 0.5}
    seq = {}

    def fake_post(c, payload):
        n = seq.setdefault(payload["messages"][1]["content"], 0)
        seq[payload["messages"][1]["content"]] = n + 1
        if n == 0:
            tc = {"id": "1", "function": {"name": "get_filings",
                                          "arguments": json.dumps({"permno": 1, "before_eom": "2020-01-31"})}}
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [tc]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "```json\n" + json.dumps(answer) + "\n```"}}]}

    monkeypatch.setattr(ag, "_agent_post", fake_post)
    rows = ag.run_agent(cfg)
    res = json.loads((cfg["output_dir"] / "agent_rationales.json").read_text())
    assert len(rows) == 2 and res["summary"]["n_holdings"] == 2 and "post-hoc" in res["note"]
    h1 = [h for h in res["holdings"] if h["permno"] == 1][0]
    assert h1["formation_eom"] == "2020-01-31" and h1["side"] == "long"
    assert len(h1["facts"]) == 1 and len(h1["dropped_facts"]) == 1
    assert (tmp_path / "cache" / "agent_cache.json").exists()


def test_dispatch_enforces_formation(tmp_path):
    cfg = _cfg(tmp_path)
    seen = []
    orig = ag.get_filings
    ag.get_filings = lambda c, p, d: seen.append((p, d)) or orig(c, p, d)
    try:
        out = ag._agent_dispatch(cfg, "get_filings", {"permno": 999, "before_eom": "2021-01-31"}, 1, "2020-01-31")
    finally:
        ag.get_filings = orig
    assert seen == [(1, date(2020, 1, 31))] and [f["document_id"] for f in out] == ["d1"]
    kept, dropped = ag._agent_verify([{"quote": "later filing", "document_id": "d2"}],
                                     [{"document_id": "d2", "filing_date": "2020-03-10", "text": "later filing"}],
                                     "2020-01-31")
    assert not kept and len(dropped) == 1
