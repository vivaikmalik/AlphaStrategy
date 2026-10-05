"""Synthetic tests for src/text.py (FinBERT mocked, no internet)."""
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest
import torch

import src.text as T
from src.text import event_flags, finbert_doc_tones, tone_features

CFG = {"smoke": False, "items_flag": ["1.01", "4.02", "8.01"], "items_hist": ["4.02", "2.06"], "device": "cpu",
       "finbert_model": "x", "finbert_chunk_tokens": 8, "finbert_max_chunks": 3, "finbert_batch": 2,
       "finbert_max_filings": None}


def _filings():
    return pl.DataFrame({
        "document_id": ["a", "b", "c", "d"],
        "permno": [1, 1, 1, 2],
        "filing_date": [date(2020, 1, 15), date(2020, 1, 31), date(2020, 5, 2), date(2020, 1, 3)],
        "items": [["1.01", "9.01"], ["4.02"], ["2.06", "4.02"], None],
    })


def _keys():
    return pl.DataFrame({"permno": [1, 1, 1, 1, 2], "eom": [date(2020, 1, 31), date(2020, 2, 29), date(2020, 5, 31),
                                                            date(2021, 1, 31), date(2020, 3, 31)]})


def test_event_flags():
    out = event_flags(_filings(), _keys(), CFG).sort("permno", "eom")
    r = {(a, b): row for a, b, row in zip(out["permno"], out["eom"], out.iter_rows(named=True))}
    jan = r[(1, date(2020, 1, 31))]
    assert (jan["has_filing"], jan["n_filings"], jan["item_1_01"], jan["item_4_02"], jan["item_8_01"]) == (1, 2, 1, 1, 0)
    assert jan["hist_4_02"] == 1 and jan["hist_2_06"] == 0
    feb = r[(1, date(2020, 2, 29))]  # no filing: current cols null, history still computed
    assert feb["has_filing"] == 0 and feb["n_filings"] is None and feb["item_1_01"] is None and feb["hist_4_02"] == 1
    may = r[(1, date(2020, 5, 31))]
    assert may["hist_4_02"] == 2 and may["hist_2_06"] == 1
    assert r[(1, date(2021, 1, 31))]["hist_4_02"] == 1  # Jan 2020 dropped, May 2020 kept (window Feb 2020..Jan 2021)
    assert r[(2, date(2020, 3, 31))]["hist_4_02"] == 0  # null items handled
    assert out.height == 5 and out.select(["permno", "eom"]).is_unique().all()


def test_event_flags_window_edge():
    f = pl.DataFrame({"document_id": ["a"], "permno": [1], "filing_date": [date(2020, 1, 31)], "items": [["4.02"]]})
    k = pl.DataFrame({"permno": [1, 1], "eom": [date(2020, 12, 31), date(2021, 1, 31)]})
    out = event_flags(f, k, CFG).sort("eom")
    assert out["hist_4_02"].to_list() == [1, 0]  # Dec 2020 = t-11 still counts, Jan 2021 does not


class _Tok:
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        # one token per whitespace word; word "good" -> id 1, "bad" -> id 2, else 3
        m = {"good": 1, "bad": 2}
        return {"input_ids": [[m.get(w, 3) for w in t.split()] for t in texts]}


class _Model:
    config = SimpleNamespace(id2label={0: "positive", 1: "negative", 2: "neutral"})  # not ProsusAI's real order on purpose
    seen = []

    def __call__(self, input_ids, attention_mask):
        _Model.seen.append(input_ids.clone())
        n = ((input_ids == 1) & (attention_mask == 1)).sum(1).float()   # good words
        b = ((input_ids == 2) & (attention_mask == 1)).sum(1).float()   # bad words
        logits = torch.stack([n * 20, b * 20, torch.zeros_like(n) + 5], 1)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def fake(monkeypatch):
    _Model.seen = []
    monkeypatch.setattr(T, "_txt_load_model", lambda cfg: (_Tok(), _Model()))


def _docs():
    f = pl.DataFrame({
        "document_id": ["a", "b", "c"], "permno": [1, 1, 2],
        "filing_date": [date(2020, 1, 15), date(2020, 2, 10), date(2020, 1, 20)],
        "text": ["Acme good good good", " ".join(["good"] * 6 + ["bad"] * 6 + ["good"] * 20), "bad bad"],
        "company_name": ["Acme", "Acme", "Zed"], "ticker": ["ACM", "ACM", "ZD"], "items": [[], [], []],
    })
    p = pl.DataFrame({"permno": [1, 1], "eom": [date(2020, 1, 31), date(2020, 2, 29)]})  # permno 2 not in panel
    return f, p


def test_finbert_chunks_cache_and_label_order(fake, tmp_path):
    cfg = {**CFG, "cache_dir": tmp_path}
    f, p = _docs()
    out = finbert_doc_tones(f, p, cfg).sort("document_id")
    assert out["document_id"].to_list() == ["a", "b"]           # c not in panel keys -> not scored
    a, b = out.row(0, named=True), out.row(1, named=True)
    assert a["tone_mean"] == pytest.approx(1.0, abs=1e-3)       # "Acme" removed; 3 good words only; pos idx via id2label
    # doc b: 32 words, chunk size 6 content tokens, max 3 chunks -> first 18 words: good*6 | bad*6 | good*6
    assert a["tone_min"] == pytest.approx(1.0, abs=1e-3)
    assert b["tone_mean"] == pytest.approx((1 - 1 + 1) / 3, abs=1e-3) and b["tone_min"] == pytest.approx(-1.0, abs=1e-3)
    assert max(s.shape[1] for s in _Model.seen) <= 8             # 6 content + CLS/SEP
    assert all(s.shape[0] <= 2 for s in _Model.seen)             # batch size respected
    # second call: everything cached, model must not be invoked again
    n = len(_Model.seen)
    out2 = finbert_doc_tones(f, p, cfg).sort("document_id")
    assert len(_Model.seen) == n and out2.equals(out)
    # incremental: a new document is the only one scored
    f2 = pl.concat([f, pl.DataFrame({"document_id": ["e"], "permno": [1], "filing_date": [date(2020, 2, 11)],
                                     "text": ["bad"], "company_name": ["Acme"], "ticker": ["ACM"], "items": [[]]})])
    out3 = finbert_doc_tones(f2, p, cfg)
    assert out3.height == 3 and len(_Model.seen) == n + 1
    assert pl.read_parquet(tmp_path / "finbert_docs.parquet").height == 3


def test_finbert_max_filings(fake, tmp_path):
    f, p = _docs()
    out = finbert_doc_tones(f, p, {**CFG, "cache_dir": tmp_path, "finbert_max_filings": 1})
    assert out.height == 1


def test_anonymise():
    f = T._txt_anonymise
    assert f("Acme Corp (ACME) said acme corp. AAA on", ["Acme Corp"], ["ACME"]).split() == ["(", ")", "said", ".", "AAA", "on"]
    # short ticker: case-sensitive only
    assert "A" not in f("A big deal with A and a", [], ["A"]).split() and "a" in f("A and a", [], ["A"]).split()
    assert f("on ON", [], ["ON"]).split() == ["on"]
    assert f("all ALL now", [], ["ALL"]).split() == ["all", "now"]  # long tickers are case-sensitive too
    assert f("Acmes stay", ["Acme"], []) == "Acmes stay"  # word boundary


def test_tone_features():
    f = pl.DataFrame({"document_id": list("abcd"), "permno": [1, 1, 1, 1],
                      "filing_date": [date(2020, 1, 5), date(2020, 1, 6), date(2020, 2, 3), date(2021, 2, 3)]})
    dt = pl.DataFrame({"document_id": list("abcd"), "tone_mean": [0.2, 0.4, 0.9, 0.5], "tone_min": [0.0, -0.5, 0.9, 0.5]})
    k = pl.DataFrame({"permno": [1] * 4, "eom": [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31), date(2021, 2, 28)]})
    o = tone_features(f, dt, k, CFG).sort("eom")
    assert o["tone_mean"][0] == pytest.approx(0.3) and o["tone_min"][0] == pytest.approx(-0.5)
    assert o["tone_surprise"][0] is None                          # no history before Jan 2020
    assert o["tone_surprise"][1] == pytest.approx(0.9 - 0.3)      # uses months before t only
    assert o["tone_mean"][2] is None and o["tone_surprise"][2] is None  # no filing in March
    # Feb 2021: t-12..t-1 = Feb 2020..Jan 2021 -> only Feb 2020 (0.9); Jan 2020 is outside
    assert o["tone_surprise"][3] == pytest.approx(0.5 - 0.9)
