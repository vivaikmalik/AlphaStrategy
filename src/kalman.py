"""
src/kalman.py - Step 8: Kalman-filtered stock betas (4 noisy beta measures -> one state).
beta_t - m_t = phi (beta_{t-1} - m_{t-1}) + N(0,q);  m_t = sector median of beta_60m.
Sequential scalar updates per observation (own R_j); phi, q, R_j by exact Gaussian MLE on eom <= 2018-12, then frozen.
"""
from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import minimize


def _kalman_grid(panel, cfg):
    """Dense arrays: permnos, month-ends, Y (n,T,J) observations, m (n,T) prior mean, first (n,) first month index, row index."""
    obs = cfg["kalman_obs"]
    df = panel.select("permno", "eom", "gics", *obs).with_columns(
        pl.col("gics").str.slice(0, 2).alias("sector"))
    months = pl.date_range(df["eom"].min(), df["eom"].max(), "1mo", eager=True).dt.month_end()
    permnos = df["permno"].unique().sort()
    pi = permnos.search_sorted(df["permno"]).to_numpy()
    ti = months.search_sorted(df["eom"]).to_numpy()
    n, T = len(permnos), len(months)
    Y = np.full((n, T, len(obs)), np.nan)
    for j, c in enumerate(obs):
        Y[pi, ti, j] = df[c].cast(pl.Float64).fill_nan(None).to_numpy(allow_copy=True)
    # prior mean: sector median of beta_60m per month, all-stock median where the sector median is null
    med = df.group_by("eom", "sector").agg(pl.col("beta_60m").median().alias("m_sec")).join(
        df.group_by("eom").agg(pl.col("beta_60m").median().alias("m_all")), on="eom")
    sec_of = (df.sort("eom").group_by("permno").agg(pl.col("sector").drop_nulls().first())
              .sort("permno"))["sector"]                                  # stock's first known sector
    secs = sorted(set(sec_of.drop_nulls().to_list()))
    sidx = np.array([secs.index(s) if s is not None else -1 for s in sec_of.to_list()])
    M_sec = np.full((len(secs) + 1, T), np.nan)                           # last row (idx -1) = no sector
    M_all = np.full(T, np.nan)
    mi = months.search_sorted(med["eom"]).to_numpy()
    M_all[mi] = med["m_all"].to_numpy()
    for s, t, v in zip(med["sector"].to_list(), mi, med["m_sec"].to_numpy()):
        if s is not None:
            M_sec[secs.index(s), t] = v
    M_all = np.where(np.isnan(M_all), np.nanmedian(M_all), M_all)
    m = np.where(np.isnan(M_sec[sidx]), M_all[None, :], M_sec[sidx])
    first = np.full(n, T)
    np.minimum.at(first, pi, ti)
    return permnos, months, Y, m, first, pi, ti


def _kalman_filter(Y, m, first, theta, t_end=None, full=False):
    """Vectorised-over-stocks filter. Returns loglik (and state, var, has_obs grids if full)."""
    phi = 1 / (1 + np.exp(-theta[0])); q = np.exp(theta[1]); R = np.exp(theta[2:])
    n, T, J = Y.shape
    T = T if t_end is None else t_end
    d = np.zeros(n); P = np.full(n, q / (1 - phi ** 2)); seen = np.zeros(n, bool)
    ll = 0.0
    out = np.zeros((3, n, T)) if full else None
    for t in range(T):
        if t > 0:                                                         # predict (also bridges missing months)
            d = phi * d                                                   # deviations from m_t
            P = phi ** 2 * P + q
        start = first == t                                                # initialise at first month: d=0, P=q/(1-phi^2)
        d = np.where(start, 0.0, d); P = np.where(start, q / (1 - phi ** 2), P)
        active = first <= t
        for j in range(J):
            y = Y[:, t, j] - m[:, t]
            ok = active & ~np.isnan(y)
            S = P + R[j]
            v = np.where(ok, y - d, 0.0)
            ll += -0.5 * np.sum(np.where(ok, np.log(2 * np.pi * S) + v ** 2 / S, 0.0))
            K = np.where(ok, P / S, 0.0)
            d = d + K * v; P = (1 - K) * P; seen |= ok
        if full:
            out[0, :, t], out[1, :, t], out[2, :, t] = d + m[:, t], P, seen
    return (ll, out) if full else ll


def kalman_betas(panel, cfg):
    """-> (DataFrame[permno, eom, beta_kf, beta_var], params). Panel = raw (unranked) betas."""
    permnos, months, Y, m, first, pi, ti = _kalman_grid(panel, cfg)
    J = Y.shape[2]
    t_fit = int((months <= date.fromisoformat(cfg["kalman_fit_end_eom"])).sum())
    lo, hi = np.nanpercentile(Y[:, :t_fit], 1, axis=(0, 1)), np.nanpercentile(Y[:, :t_fit], 99, axis=(0, 1))
    Y = np.clip(Y, lo, hi)                       # winsorise at the fit-sample 1/99 pct: raw betas have +-1e4 outliers (not in spec)
    x0 = np.array([2.0, np.log(0.02)] + [np.log(0.05)] * J)
    res = minimize(lambda th: -_kalman_filter(Y, m, first, th, t_fit), x0, method="L-BFGS-B",
                   bounds=[(-8, 8)] + [(-12, 3)] * (J + 1))
    th = res.x
    params = {"phi": float(1 / (1 + np.exp(-th[0]))), "q": float(np.exp(th[1])),
              "R": {c: float(np.exp(r)) for c, r in zip(cfg["kalman_obs"], th[2:])}, "loglik": float(-res.fun)}
    _, out = _kalman_filter(Y, m, first, th, full=True)
    kf, var, seen = out[0, pi, ti], out[1, pi, ti], out[2, pi, ti].astype(bool)
    b60 = Y[pi, ti, cfg["kalman_obs"].index("beta_60m")]
    kf = np.where(seen, kf, np.where(np.isnan(b60), m[pi, ti], b60))      # no data yet -> beta_60m, else m_t
    df = panel.select("permno", "eom").with_columns(pl.Series("beta_kf", kf), pl.Series("beta_var", var))
    return df, params
