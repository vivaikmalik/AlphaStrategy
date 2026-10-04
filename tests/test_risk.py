"""Risk model on a synthetic panel: no look-ahead, PD covariance factor, min-months rule."""
from datetime import date
import numpy as np
import polars as pl
from src.config import CONFIG
from src.risk import risk_model, risk_exposures

CFG = dict(CONFIG, risk_window=12, risk_min_months=6)


def _frame(months=20, n=150, seed=0):
    r = np.random.default_rng(seed)
    eoms = pl.date_range(date(2019, 1, 31), date(2019, 1, 31), eager=True)
    eoms = pl.Series([date(2019 + (i // 12), i % 12 + 1, 1) for i in range(months)]).dt.month_end()
    df = pl.concat([pl.DataFrame({"permno": np.arange(n), "eom": pl.Series([e] * n, dtype=pl.Date),
                                  "sector": [str(10 + x) for x in r.integers(0, 4, n)],
                                  "ret_exc_lead1m": r.normal(0, 0.1, n)}) for e in eoms])
    return df.with_columns([pl.Series(c, r.uniform(-1, 1, df.height)) for c in CFG["risk_factors"]])


def test_no_lookahead():
    f = _frame()
    ts = sorted(f["eom"].unique())
    t = ts[12]
    m1 = risk_model(f, CFG)
    f2 = f.with_columns(ret_exc_lead1m=pl.when(pl.col("eom") >= t).then(pl.col("ret_exc_lead1m") * 5 + 1).otherwise(pl.col("ret_exc_lead1m")))
    m2 = risk_model(f2, CFG)
    assert np.allclose(m1["L"][t], m2["L"][t])        # f_s for s = t is realised after t; s < t unchanged
    assert not np.allclose(m1["L"][ts[-1]], m2["L"][ts[-1]])


def test_cholesky_pd_and_min_months():
    f = _frame()
    ts = sorted(f["eom"].unique())
    m = risk_model(f, CFG)
    k = 1 + len(CFG["risk_factors"]) + len(m["sectors"])
    assert risk_exposures(f.filter(pl.col("eom") == ts[0]), m["sectors"], CFG).shape == (150, k)
    # F_t needs risk_min_months factor returns with s <= t-1 month -> first entry at index risk_min_months
    assert min(m["L"]) == ts[CFG["risk_min_months"]] and ts[CFG["risk_min_months"] - 1] not in m["L"]
    for L in m["L"].values():
        F = L @ L.T
        assert L.shape == (k, k) and np.linalg.eigvalsh(F).min() > 0 and np.allclose(L, np.tril(L))
