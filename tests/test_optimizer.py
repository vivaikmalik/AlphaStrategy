"""Optimizer step 9 on synthetic months: every constraint, candidate sets, relaxation, turnover."""
import time
from datetime import date
import numpy as np
import polars as pl
from src.config import CONFIG
from src.optimizer import optimize_month, run_book

CFG = dict(CONFIG)


def _month(n=1500, seed=0, eom=date(2020, 1, 31), beta_spread=0.3):
    r = np.random.default_rng(seed)
    return pl.DataFrame({
        "permno": np.arange(n, dtype=np.int64) + 1, "eom": pl.Series([eom] * n, dtype=pl.Date), "score": r.normal(size=n),
        "short_eligible": r.random(n) < 0.6, "beta_kf": 1 + beta_spread * r.normal(size=n),
        "beta_var": r.uniform(0.005, 0.05, n), "sector": [str(10 + x) for x in r.integers(0, 11, n)]})


def _check(mdf, w, info, tol=CFG["beta_tol"]):
    d = mdf.filter(pl.col("permno").is_in(list(w))).with_columns(pl.col("permno").replace_strict(w, return_dtype=pl.Float64).alias("w"))
    ww = d["w"].to_numpy()
    assert abs(np.abs(ww).sum() - 2) < 1e-5
    assert abs(ww.sum()) <= 0.2 + 1e-6
    assert abs((d["beta_kf"].to_numpy() * ww).sum()) <= info["beta_tol"] + 1e-6
    assert np.abs(ww).max() <= 0.015 + 1e-7 and len(w) >= 120
    for _, g in d.group_by("sector"):
        assert abs(g["w"].sum()) <= 0.1 + 1e-6
    top = set(mdf.sort("score", descending=True).head(250)["permno"])
    bot = set(mdf.filter(pl.col("short_eligible")).sort("score").head(250)["permno"])
    assert all(p in top for p, x in w.items() if x > 0) and all(p in bot for p, x in w.items() if x < 0)


def test_constraints_and_speed():
    m = _month()
    t0 = time.time()
    w, info = optimize_month(m, {}, 0.1, 10.0, CFG)
    assert time.time() - t0 < 5
    _check(m, w, info)
    assert info["status"] in ("optimal", "optimal_inaccurate") and info["beta_tol"] == CFG["beta_tol"]
    assert info["n_long"] > 0 and info["n_short"] > 0 and abs(info["turnover"] - 2) < 1e-4  # from cash: sum|w| = 2


def test_turnover_counts_dropped_names_and_tc_reduces_it():
    m = _month(seed=1)
    w0, _ = optimize_month(m, {}, 0.0, 0.0, CFG)
    m2 = _month(seed=2)                                   # new scores: some old names leave the candidate set
    wa, ia = optimize_month(m2, w0, 0.0, 0.0, CFG)
    wb, ib = optimize_month(m2, w0, 0.5, 0.0, CFG)
    assert ib["turnover"] < ia["turnover"]
    gone = sum(abs(v) for p, v in w0.items() if p not in set(m2.sort("score", descending=True).head(250)["permno"]) and v > 0)
    assert ia["turnover"] >= gone - 1e-6


def test_beta_relaxation_logged():
    m = _month(seed=3).with_columns(beta_kf=pl.when(pl.col("score") > 0).then(1.3).otherwise(0.9))   # longs 1.3 vs shorts 0.9: beta ~0.18 unavoidable
    w, info = optimize_month(m, {}, 0.0, 0.0, CFG)
    _check(m, w, info)
    assert info["beta_tol"] > CFG["beta_tol"]
    # relaxation lands on the tolerance grid
    k = (info["beta_tol"] - CFG["beta_tol"]) / CFG["beta_tol_step"]
    assert abs(k - round(k)) < 1e-6


def test_null_betas_and_run_book():
    ms = [_month(seed=i, eom=e) for i, e in enumerate([date(2020, 1, 31), date(2020, 2, 29)])]
    ms[0] = ms[0].with_columns(beta_kf=pl.when(pl.col("permno") % 50 == 0).then(None).otherwise(pl.col("beta_kf")))
    wdf, log = run_book(pl.concat(ms), 0.1, 10.0, CFG)
    assert wdf["eom"].n_unique() == 2 and len(log) == 2 and set(log[0]) >= {"beta_tol", "status", "n_long", "n_short", "turnover"}
    assert log[1]["turnover"] > 0
