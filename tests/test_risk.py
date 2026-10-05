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


def _market(f, seed=1):
    ts = sorted(f["eom"].unique())
    r = np.random.default_rng(seed)
    return pl.DataFrame({"eom": ts, "sp500_ret": r.normal(0.01, 0.04, len(ts)), "tb3ms": np.full(len(ts), 0.12)})


def _planted(months=30, n=150):
    """Panel where the intercept factor return in month s+1 is 2 * market excess of month s+1 (+ noise)."""
    f = _frame(months, n)
    mk = _market(f)
    ts = sorted(f["eom"].unique())
    ex = dict(zip(ts, (mk["sp500_ret"] - mk["tb3ms"] / 1200).to_list()))
    nxt = {ts[i]: ts[i + 1] for i in range(len(ts) - 1)}
    r = np.random.default_rng(5)
    f = f.with_columns(pl.col("eom").map_elements(lambda e: 2 * ex.get(nxt.get(e), 0.0), return_dtype=pl.Float64).alias("_m"))
    f = f.with_columns((pl.col("_m") + pl.Series(r.normal(0, 0.001, f.height))).alias("ret_exc_lead1m")).drop("_m")
    return f.filter(pl.col("eom") < ts[-1]), mk


def test_no_lookahead_spec_xsvol_b():
    f = _frame()
    mk = _market(f)
    ts = sorted(f["eom"].unique())
    t = ts[14]
    m1 = risk_model(f, CFG, mk)
    f2 = f.with_columns(ret_exc_lead1m=pl.when(pl.col("eom") >= t).then(pl.col("ret_exc_lead1m") * 5 + 1).otherwise(pl.col("ret_exc_lead1m")))
    mk2 = mk.with_columns(sp500_ret=pl.when(pl.col("eom") > t).then(0.5).otherwise(pl.col("sp500_ret")))
    m2 = risk_model(f2, CFG, mk2)
    assert np.isclose(m1["xs_vol"][t], m2["xs_vol"][t]) and not np.isclose(m1["xs_vol"][ts[-1]], m2["xs_vol"][ts[-1]])
    assert np.allclose(m1["b"][t], m2["b"][t]) and not np.allclose(m1["b"][ts[-1]], m2["b"][ts[-1]])
    s1, s2 = (m["spec"].filter(pl.col("eom") == t).sort("permno") for m in (m1, m2))
    assert np.allclose(s1["spec_var"], s2["spec_var"])
    assert "b" not in risk_model(f, CFG)


def test_planted_market_beta():
    f, mk = _planted()
    m = risk_model(f, CFG, mk)
    t = sorted(m["b"])[-1]
    X = risk_exposures(f.filter(pl.col("eom") == t), m["sectors"], CFG)
    assert np.allclose(np.abs(X @ m["b"][t]).mean(), 2.0, atol=0.2)   # every stock's predicted beta ~ planted 2 (intercept/sector collinear)


def test_spec_fallback_median():
    f = _frame()
    ts = sorted(f["eom"].unique())
    f = f.filter(~((pl.col("permno") == 0) & (pl.col("eom") < ts[12])))      # stock 0 has too few residuals at t
    t = ts[14]
    sp = risk_model(f, dict(CFG, risk_spec_min=6))["spec"].filter(pl.col("eom") == t)
    ok = sp.filter(pl.col("permno") != 0)["spec_var"]
    assert np.isclose(sp.filter(pl.col("permno") == 0)["spec_var"][0], ok.median()) and sp["spec_var"].null_count() == 0
