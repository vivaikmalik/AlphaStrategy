"""Kalman step 8 on simulated data: MLE recovers phi/q/R, missing months handled, fallbacks."""
import numpy as np
import polars as pl
from src.config import CONFIG
from src.kalman import kalman_betas

CFG = dict(CONFIG)
OBS = CFG["kalman_obs"]
TRUE_R = [0.08, 0.05, 0.03, 0.02]


def _sim(n=300, T=72, phi=0.9, q=0.02, seed=0, p_miss=0.1):
    rng = np.random.default_rng(seed)
    months = pl.date_range(pl.date(2015, 1, 31), pl.date(2020, 12, 31), "1mo", eager=True).dt.month_end()[:T]
    sec = rng.integers(10, 14, n)
    m = np.ones((n, T)) * (0.8 + 0.1 * (sec % 4))[:, None]
    d = rng.normal(0, np.sqrt(q / (1 - phi ** 2)), n)
    rows = []
    beta = np.zeros((n, T))
    for t in range(T):
        d = phi * d + rng.normal(0, np.sqrt(q), n) if t else d
        beta[:, t] = m[:, t] + d
    for i in range(n):
        for t in range(T):
            if rng.random() < p_miss:
                continue                                                    # missing stock-month
            obs = [beta[i, t] + rng.normal(0, np.sqrt(r)) if rng.random() > 0.15 else None for r in TRUE_R]
            rows.append((1000 + i, months[t], f"{sec[i]}101010", *obs))
    df = pl.DataFrame(rows, schema=["permno", "eom", "gics", *OBS], orient="row")
    return df.with_columns(pl.col("permno").cast(pl.Int64)), beta


def test_mle_recovers_params_and_shapes():
    df, beta = _sim()
    out, p = kalman_betas(df, CFG)
    assert out.height == df.height and out.select("permno", "eom").equals(df.select("permno", "eom"))
    assert abs(p["phi"] - 0.9) < 0.07 and abs(p["q"] - 0.02) < 0.01
    for r, c in zip(TRUE_R, OBS):
        assert 0.4 * r < p["R"][c] < 2.5 * r, (c, p["R"][c])
    assert out["beta_kf"].null_count() == 0 and out["beta_var"].min() > 0


def test_missing_months_widen_variance_and_fallback():
    df, _ = _sim(n=40, T=40, p_miss=0.0)
    pid = df["permno"].min()
    gap = df.filter(~((pl.col("permno") == pid) & (pl.col("eom").is_between(pl.date(2016, 1, 31), pl.date(2016, 6, 30)))))
    full, _ = kalman_betas(df, CFG)
    holed, _ = kalman_betas(gap, CFG)
    a = full.filter((pl.col("permno") == pid) & (pl.col("eom") == pl.date(2016, 7, 31)))["beta_var"][0]
    b = holed.filter((pl.col("permno") == pid) & (pl.col("eom") == pl.date(2016, 7, 31)))["beta_var"][0]
    assert b >= a - 1e-12
    # a stock with no observations at all falls back to the sector median (m_t)
    blank = df.filter(pl.col("permno") == pid).with_columns([pl.lit(None, pl.Float64).alias(c) for c in OBS])
    out, _ = kalman_betas(pl.concat([df.filter(pl.col("permno") != pid), blank]), CFG)
    assert out.filter(pl.col("permno") == pid)["beta_kf"].null_count() == 0


def test_r_floor_bounds_R_and_none_is_unchanged():
    df, _ = _sim(n=100, T=48)
    _, p0 = kalman_betas(df, CFG)
    _, pn = kalman_betas(df, CFG, r_floor=None)
    assert p0["R"] == pn["R"] and p0["phi"] == pn["phi"] and p0["q"] == pn["q"] and pn["r_floor"] is None
    _, pf = kalman_betas(df, CFG, r_floor=0.05)
    assert pf["r_floor"] == 0.05 and all(r >= 0.05 - 1e-9 for r in pf["R"].values())
