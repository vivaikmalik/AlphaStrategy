"""Test temporal boundaries and TabPFN integration without model downloads."""
import sys
import types

import numpy as np
import pytest
import polars as pl

from src.ranker import _rk_context_indices, fit_predict_year, windows
from test_ranker import CFG, FEATS, _panel, _join
from src.ranker import rank_ic


@pytest.fixture
def fake_tabpfn(monkeypatch):
    fits = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            fits.append((X.copy(), y.copy()))
            assert np.isfinite(y).all() and np.max(np.abs(y)) <= 1
            self.coef = np.linalg.lstsq(np.nan_to_num(X), y, rcond=None)[0]
            return self

        def predict(self, X):
            return np.nan_to_num(X) @ self.coef

    monkeypatch.setitem(sys.modules, "tabpfn", types.SimpleNamespace(TabPFNRegressor=FakeRegressor))
    return fits


def test_context_balanced_deterministic():
    e = pl.Series(np.repeat(np.arange(6), 20))
    ix = _rk_context_indices(e, 30, 42)
    assert len(ix) == 30 and len(np.unique(ix)) == 30
    assert np.array_equal(ix, _rk_context_indices(e, 30, 42))
    assert np.array_equal(np.bincount(e.to_numpy()[ix]), np.full(6, 5))
    with pytest.raises(ValueError):
        _rk_context_indices(e, 0, 42)


def test_tabpfn_temporal_splits_and_null_targets(fake_tabpfn):
    df = _panel()
    cfg = {**CFG, "ranker_model": "tabpfn", "tabpfn_max_train_samples": 120,
           "tabpfn_predict_batch": 73}
    w = windows(cfg)[0]
    te, va, info = fit_predict_year(df, FEATS, w, cfg)
    assert info["model"] == "tabpfn" and info["best_trees"] is None
    assert info["refit_context_rows"] == 120 and len(fake_tabpfn) == 2
    assert rank_ic(_join(te, df))["ic"].mean() > 0.2
    assert te.height == df.filter(pl.col("target_month") >= w["test"][0]).height
    assert te["ret_hat"].is_finite().all()
    before = [(X.copy(), y.copy()) for X, y in fake_tabpfn]
    junk = df.with_columns([
        pl.when(pl.col("target_month") >= w["test"][0]).then(9999.0).otherwise(pl.col(c)).alias(c)
        for c in FEATS + ["ret_exc_lead1m"]])
    _, va2, info2 = fit_predict_year(junk, FEATS, w, cfg)
    assert va.equals(va2) and info == info2
    for a, b in zip(before, fake_tabpfn[2:]):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_tabpfn_cfi_and_shuffle(fake_tabpfn):
    cfg = {**CFG, "ranker_model": "tabpfn", "tabpfn_max_train_samples": 120,
           "cfi_cuts": [0.75], "cfi_shares": [0.5], "cfi_repeats": 1}
    df = _panel(n=12)
    _, _, info = fit_predict_year(df, FEATS, windows(cfg)[0], cfg, cfi_factors=FEATS)
    assert info["cfi"]["candidates"] and info["features"]
    fake_tabpfn.clear()
    fit_predict_year(df, FEATS, windows(cfg)[0], cfg)
    original = fake_tabpfn[0][1].copy()
    fake_tabpfn.clear()
    fit_predict_year(df, FEATS, windows(cfg)[0], cfg, shuffle=True)
    assert not np.array_equal(original, fake_tabpfn[0][1])
