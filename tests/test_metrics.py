"""Tests for src/metrics.py on small synthetic data."""
import datetime as dt
import numpy as np
import polars as pl
import pytest
from src.config import CONFIG
from src.metrics import book_returns, summary_stats, oos_r2, ic_summary, load_market

CFG = CONFIG
EOMS = [dt.date(2021, 1, 31), dt.date(2021, 2, 28), dt.date(2021, 3, 31), dt.date(2021, 4, 30)]
TARGET = [dt.date(2021, 2, 28), dt.date(2021, 3, 31), dt.date(2021, 4, 30), dt.date(2021, 5, 31)]


def _market():
    # distinct TB3MS per month so a wrong month alignment is detected
    d = [dt.date(2021, m, 28 if m == 2 else 30 if m in (4, 6) else 31) for m in range(1, 7)]
    return pl.DataFrame({"eom": d, "tb3ms": [0.12, 0.24, 0.36, 0.48, 0.60, 0.72], "sp500_ret": [.01, .02, -.01, .03, .0, .01]})


def _inputs():
    rows, pan = [], []
    for e in EOMS:
        for p, w, r in [(1, 0.6, 0.05), (2, 0.4, -0.01), (3, -0.5, 0.02), (4, -0.5, -0.04)]:
            rows.append((p, e, w))
            pan.append((p, e, r if e != EOMS[2] else (None if p == 1 else r)))
    return (pl.DataFrame(rows, schema=["permno", "eom", "weight"], orient="row"),
            pl.DataFrame(pan, schema=["permno", "eom", "ret_exc_lead1m"], orient="row"))


def test_book_returns_identities():
    w, pan = _inputs()
    rets, nm = book_returns(w, pan, _market(), CFG)
    assert nm == 1 and rets.height == 4
    r = rets.row(0, named=True)
    assert r["long_leg"] == pytest.approx(0.6 * 0.05 + 0.4 * -0.01)
    assert r["short_leg"] == pytest.approx(-0.5 * 0.02 + -0.5 * -0.04)
    assert r["excess"] == pytest.approx(r["long_leg"] + r["short_leg"])
    rf = 0.24 / 1200  # TB3MS of the TARGET month (Feb 2021), not January
    assert r["total"] == pytest.approx(rf + r["excess"])
    assert r["benchmark"] == pytest.approx(rf + 0.04 / 12)
    assert r["active"] == pytest.approx(r["excess"] - 0.04 / 12)
    assert r["sp500"] == pytest.approx(0.02)
    assert r["Date"] == dt.date(2021, 2, 1)
    assert (r["gross"], r["n_long"], r["n_short"]) == (pytest.approx(2.0), 2, 2) and r["net"] == pytest.approx(0.0)
    assert r["turnover"] == pytest.approx(2.0)  # first month: from cash
    assert rets["turnover"][1] == pytest.approx(0.0)  # same weights


def test_turnover_dropped_names():
    w, pan = _inputs()
    w = w.filter(~((pl.col("eom") == EOMS[1]) & (pl.col("permno") == 1)))  # name 1 absent in month 2
    rets, _ = book_returns(w, pan, _market(), CFG)
    assert rets["turnover"][1] == pytest.approx(0.6)
    assert rets["turnover"][2] == pytest.approx(0.6)  # re-entering


def test_summary_stats_ir_and_beta():
    rng = np.random.default_rng(0)
    n = 60
    sp_ex = rng.normal(0.01, 0.04, n)
    rf = np.full(n, 0.002)
    exc = 0.003 + 0.5 * sp_ex + rng.normal(0, 0.01, n)
    ll, sl = exc * 0.7, exc * 0.3
    prem = 0.04 / 12
    rets = pl.DataFrame({
        "Date": pl.date_range(dt.date(2021, 1, 1), dt.date(2025, 12, 1), "1mo", eager=True), "total": rf + exc, "excess": exc,
        "active": exc - prem, "long_leg": ll, "short_leg": sl, "benchmark": rf + prem, "sp500": sp_ex + rf,
        "gross": np.full(n, 2.0), "net": np.zeros(n), "n_long": np.full(n, 250), "n_short": np.full(n, 250),
        "turnover": np.full(n, 0.5), "rf": rf})
    s = summary_stats(rets, CFG)
    a = exc - prem
    assert s["ir"] == pytest.approx(np.sqrt(12) * a.mean() / a.std(ddof=1))
    assert s["beta"] == pytest.approx(0.5, abs=0.15) and s["alpha_monthly"] == pytest.approx(0.003, abs=0.003)
    assert s["strategy"]["cumulative"] == pytest.approx(np.prod(1 + rf + exc) - 1)
    assert s["strategy"]["cagr"] == pytest.approx((np.prod(1 + rf + exc)) ** (1 / 5) - 1)
    assert s["best_month"] == pytest.approx((rf + exc).max()) and s["worst_month"] == pytest.approx((rf + exc).min())
    assert set(s["calendar_years"]) == {"2021", "2022", "2023", "2024", "2025"}
    assert -1 < s["max_drawdown"] <= 0 and s["turnover"]["avg"] == 0.5 and s["gross"]["max"] == 2.0


def test_oos_r2_and_ic():
    p = pl.DataFrame({"eom": [dt.date(2021, 1, 31)] * 4 + [dt.date(2021, 2, 28)] * 4, "permno": list(range(4)) * 2,
                      "ret_exc_lead1m": [0.1, -0.1, 0.2, None, 0.1, 0.0, -0.1, 0.2],
                      "ret_hat": [0.0] * 8, "score": [1.0, 2, 3, 4, 4, 3, 2, 1]})
    assert oos_r2(p) == pytest.approx(0.0)  # zero forecast -> zero R2 (null target row ignored)
    q = p.with_columns(ret_hat=pl.col("ret_exc_lead1m").fill_null(0.0))
    assert oos_r2(q) == pytest.approx(1.0)
    ic = ic_summary(p, CFG)
    assert ic["n_months"] == 2 and -1 <= ic["mean_ic"] <= 1


def test_load_market_real_files():
    m = load_market(CFG)
    assert m.columns == ["eom", "tb3ms", "sp500_ret"] and m["eom"].dtype == pl.Date
