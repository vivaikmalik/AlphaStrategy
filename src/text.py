"""
src/text.py - Steps 6-7: 8-K event flags and FinBERT tone features per (permno, eom).
"""
import re
import numpy as np
import polars as pl
import torch


def _txt_idx(col="eom"):
    """Calendar month index (year*12 + month) of a Date expression."""
    c = pl.col(col)
    return c.dt.year().cast(pl.Int32) * 12 + c.dt.month().cast(pl.Int32)


def _txt_trailing(keys, monthly, offsets, aggs):
    """For each key (permno, eom): aggregate `monthly` (permno, idx, cols) over calendar months idx-o, o in offsets."""
    k = keys.select("permno", "eom").with_columns(_txt_idx().alias("idx"))
    parts = [k.with_columns((pl.col("idx") - o).alias("_j")) for o in offsets]
    k = pl.concat(parts).join(monthly.rename({"idx": "_j"}), on=["permno", "_j"], how="inner")
    return k.group_by("permno", "eom").agg(*aggs)


def event_flags(filings, keys, cfg):
    """Step 6: has_filing, n_filings, item_* (month t) and hist_* (12-month trailing counts, months t-11..t)."""
    ic = lambda it: it.replace(".", "_")
    f = filings.select("permno", "items", pl.col("filing_date").dt.month_end().alias("eom"))
    f = f.with_columns(
        [pl.col("items").list.contains(it).fill_null(False).cast(pl.Int8).alias(f"item_{ic(it)}") for it in cfg["items_flag"]]
        + [pl.col("items").list.contains(it).fill_null(False).cast(pl.Int8).alias(f"hist_{ic(it)}")
           for it in cfg["items_hist"]]
    )
    m = f.group_by("permno", "eom").agg(
        pl.len().alias("n_filings"),
        *[pl.col(f"item_{ic(it)}").max() for it in cfg["items_flag"]],
        *[pl.col(f"hist_{ic(it)}").sum().alias(f"hist_{ic(it)}") for it in cfg["items_hist"]],
    )
    cur = ["n_filings"] + [f"item_{ic(it)}" for it in cfg["items_flag"]]
    out = keys.select("permno", "eom").join(m.select("permno", "eom", *cur), on=["permno", "eom"], how="left")
    out = out.with_columns(pl.col("n_filings").is_not_null().cast(pl.Int8).alias("has_filing"))
    # history: sum monthly counts over t-11..t (calendar months); 0 when no filings in the window
    hc = [f"hist_{ic(it)}" for it in cfg["items_hist"]]
    h = _txt_trailing(keys, m.with_columns(_txt_idx().alias("idx")).select("permno", "idx", *hc), range(12),
                      [pl.col(c).sum() for c in hc])
    out = out.join(h, on=["permno", "eom"], how="left").with_columns([pl.col(c).fill_null(0) for c in hc])
    return out.select("permno", "eom", "has_filing", *cur, *hc)


# ---------------------------------------------------------------- FinBERT (step 7)
def _txt_load_model(cfg):
    """Returns (tokenizer, model in eval mode on cfg['device']). Separate so tests can replace it."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(cfg["finbert_model"], use_fast=True)
    tok.model_max_length = int(1e9)  # we chunk manually; silence the length warning
    model = AutoModelForSequenceClassification.from_pretrained(cfg["finbert_model"]).to(cfg["device"]).eval()
    return tok, model


def _txt_anonymise(text, names, tickers):
    """Remove company names/tickers (word boundaries; names case-insensitive, tickers ALWAYS case-sensitive so ALL/NOW/FOR keep common words)."""
    def pat(words, flags):
        words = sorted({w.strip() for w in words if w and w.strip()}, key=len, reverse=True)
        if not words:
            return None
        return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)", flags)
    ci = pat(list(names), re.IGNORECASE)
    cs = pat(list(tickers), 0)
    for p in (ci, cs):
        if p is not None:
            text = p.sub(" ", text)
    return text


def _txt_score_docs(docs, tok, model, cfg):
    """docs: list of clean texts -> list of (tone_mean, tone_min) (None, None if no tokens). Chunks sorted by length."""
    size, max_ch, bs = cfg["finbert_chunk_tokens"] - 2, cfg["finbert_max_chunks"], cfg["finbert_batch"]
    id2l = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
    pos = next(i for i, l in id2l.items() if l.startswith("pos"))
    neg = next(i for i, l in id2l.items() if l.startswith("neg"))
    chunks = []  # (doc index, token ids)
    for s in range(0, len(docs), 64):
        enc = tok(docs[s:s + 64], add_special_tokens=False, truncation=False)["input_ids"]
        for j, ids in enumerate(enc):
            for c in range(0, min(len(ids), size * max_ch), size):
                chunks.append((s + j, ids[c:c + size]))
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][1]))
    tones = np.zeros(len(chunks))
    dev = cfg["device"]
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(chunks[i][1]) for i in idx) + 2
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            seq = [tok.cls_token_id] + chunks[i][1] + [tok.sep_token_id]
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        tones[idx] = p[:, pos] - p[:, neg]
    per = [[] for _ in docs]
    for (d, _), t in zip(chunks, tones):
        per[d].append(t)
    return [(float(np.mean(t)), float(np.min(t))) if t else (None, None) for t in per]


def finbert_doc_tones(filings, panel, cfg, save_every=2000):
    """Step 7: per-filing FinBERT tone (mean, min over <= 8 chunks), cached by document_id."""
    path = cfg["cache_dir"] / ("finbert_docs_smoke.parquet" if cfg["smoke"] else "finbert_docs.parquet")
    schema = {"document_id": pl.String, "tone_mean": pl.Float64, "tone_min": pl.Float64}
    cache = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=schema)
    f = filings.with_columns(pl.col("filing_date").dt.month_end().alias("eom"))
    pk = panel.select("permno", "eom").unique()
    f = f.join(pk, on=["permno", "eom"], how="semi")
    todo = f.filter(~pl.col("document_id").is_in(cache["document_id"].implode())).sort("document_id")
    if cfg.get("finbert_max_filings") is not None:
        todo = todo.head(cfg["finbert_max_filings"])
    if todo.height:
        # panel names/tickers for that permno-eom, used as extra anonymisation terms
        extra = {}
        if {"company_name", "ticker"} <= set(panel.columns):
            for r in panel.select("permno", "eom", "company_name", "ticker").iter_rows():
                extra[(r[0], r[1])] = (r[2], r[3])
        tok, model = _txt_load_model(cfg)
        todo = todo.select("document_id", "permno", "eom", "text", "company_name", "ticker")
        for s in range(0, todo.height, save_every):
            rows = todo.slice(s, save_every).to_dicts()
            docs = []
            for r in rows:
                pn, pt = extra.get((r["permno"], r["eom"]), (None, None))
                docs.append(_txt_anonymise(r["text"] or "", [r["company_name"], pn], [r["ticker"], pt]))
            sc = _txt_score_docs(docs, tok, model, cfg)
            new = pl.DataFrame({"document_id": [r["document_id"] for r in rows],
                                "tone_mean": [x[0] for x in sc], "tone_min": [x[1] for x in sc]}, schema=schema)
            cache = pl.concat([cache, new])
            cache.write_parquet(path)  # incremental save: a crash keeps progress
    return cache.filter(pl.col("document_id").is_in(f["document_id"].implode()))


def tone_features(filings, doc_tones, keys, cfg):
    """Per stock-month tone_mean, tone_min, tone_surprise = tone_mean - mean of monthly tone_mean over t-12..t-1."""
    f = (filings.select("document_id", "permno", pl.col("filing_date").dt.month_end().alias("eom"))
         .join(doc_tones, on="document_id", how="inner").drop_nulls("tone_mean"))
    m = f.group_by("permno", "eom").agg(pl.col("tone_mean").mean(), pl.col("tone_min").min())
    h = _txt_trailing(keys, m.with_columns(_txt_idx().alias("idx")).select("permno", "idx", "tone_mean"),
                      range(1, 13), [pl.col("tone_mean").mean().alias("_hist")])
    out = keys.select("permno", "eom").join(m, on=["permno", "eom"], how="left").join(h, on=["permno", "eom"], how="left")
    return out.select("permno", "eom", "tone_mean", "tone_min", (pl.col("tone_mean") - pl.col("_hist")).alias("tone_surprise"))
