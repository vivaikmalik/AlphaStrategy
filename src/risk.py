"""
src/risk.py - factor risk model (not in spec): monthly cross-sectional OLS factor returns on
[intercept, ranked characteristics, sector dummies]; F_t = trailing covariance of factor returns known at eom t.
"""
import numpy as np
import polars as pl


def risk_exposures(mdf, sectors, cfg):
    """One month -> n x k exposures: intercept, cfg risk_factors (nulls -> 0), sector dummies for `sectors`."""
    cols = mdf.select([pl.col(c).fill_nan(None).fill_null(0.0) for c in cfg["risk_factors"]]).to_numpy().astype(float)
    sec = mdf["sector"].fill_null("NA").to_numpy()
    D = (sec[:, None] == np.array(sectors)[None, :]).astype(float)
    return np.hstack([np.ones((len(mdf), 1)), cols, D])


def _risk_spec(res, t_frame, t, months, cfg):
    """Specific variance of the stocks listed in t_frame at t from residuals of `months` (known months <= t)."""
    r = res.filter(pl.col("eom").is_in(months)).group_by("permno").agg(
        pl.col("resid").var().alias("v"), pl.col("resid").count().alias("n"))
    r = r.with_columns(pl.when(pl.col("n") >= cfg["risk_spec_min"]).then(pl.col("v")).otherwise(None).alias("v"))
    out = t_frame.select("permno").join(r.select("permno", "v"), on="permno", how="left")
    med = out["v"].median()
    if med is None:                                         # nothing estimable: pooled residual variance
        med = res.filter(pl.col("eom").is_in(months))["resid"].var()
    return out.select("permno", pl.lit(t).cast(pl.Date).alias("eom"),
                      pl.col("v").fill_nan(None).fill_null(med).alias("spec_var"))


def risk_model(frame, cfg, market=None):
    """frame: permno, eom, sector, ret_exc_lead1m, risk factors ->
    {"sectors", "L": {eom: chol(F_t)}, "xs_vol": {eom: mean xs return std}, "spec": permno/eom/spec_var,
     "b": {eom: factor betas to S&P excess return} (only if `market` = eom, sp500_ret, tb3ms)}. Only f_s known <= t used."""
    sectors = sorted(frame["sector"].fill_null("NA").unique().to_list())
    fs, vol, res = {}, {}, []                                               # eom s -> factor return realised in month s+1
    for (s,), g in frame.filter(pl.col("ret_exc_lead1m").is_not_null()).sort("eom").group_by("eom", maintain_order=True):
        X = risk_exposures(g, sectors, cfg)
        y = g["ret_exc_lead1m"].to_numpy().astype(float)
        fs[s] = np.linalg.lstsq(X, y, rcond=None)[0]
        vol[s] = y.std(ddof=1)
        res.append(pl.DataFrame({"permno": g["permno"], "eom": pl.Series([s] * len(y), dtype=pl.Date), "resid": y - X @ fs[s]}))
    eoms = sorted(fs)
    known = dict(zip(eoms, pl.Series(eoms).dt.offset_by("1mo").dt.month_end().to_list()))   # f_s known at eom s+1
    res = pl.concat(res)
    mk = {} if market is None else {e: r - b / 1200 for e, r, b in market.select("eom", "sp500_ret", "tb3ms").drop_nulls().iter_rows()}
    L, xs_vol, spec, b = {}, {}, [], {}
    for t in sorted(frame["eom"].unique().to_list()):
        kn = [s for s in eoms if known[s] <= t]
        past = [fs[s] for s in kn][-cfg["risk_window"]:]
        if kn:
            spec.append(_risk_spec(res, frame.filter(pl.col("eom") == t), t, kn[-cfg["risk_spec_window"]:], cfg))
        if len(past) < cfg["risk_min_months"]:
            continue
        xs_vol[t] = float(np.mean([vol[s] for s in kn[-cfg["risk_window"]:]]))
        F = np.cov(np.array(past), rowvar=False)
        k = F.shape[0]                                     # few months vs ~22 factors: shrink toward avg variance
        F = 0.9 * F + 0.1 * np.trace(F) / k * np.eye(k) + 1e-8 * np.eye(k)
        L[t] = np.linalg.cholesky(F)
        ks = [s for s in kn[-cfg["risk_window"]:] if known[s] in mk]
        if market is not None and len(ks) >= cfg["risk_min_months"]:
            m = np.array([mk[known[s]] for s in ks])
            m = m - m.mean()
            b[t] = (m @ (np.array([fs[s] for s in ks]) - np.mean([fs[s] for s in ks], axis=0))) / (m @ m)
    out = {"sectors": sectors, "L": L, "xs_vol": xs_vol, "spec": pl.concat(spec)}
    if market is not None:
        out["b"] = b
    return out


def _risk_midx(col):
    """Date column -> integer calendar-month index (year*12 + month-1)."""
    return col.dt.year().cast(pl.Int64) * 12 + col.dt.month().cast(pl.Int64) - 1


def lw_factor(rets, eom, window, permnos):
    """Ledoit-Wolf factor form (port of the teammate's ledoit_wolf_factor): Sigma = (1-delta)*Xc'Xc/T + delta*mu*I.
    rets: pl.DataFrame[permno, eom, ret] (monthly returns, all stocks). Uses months eom-window+1..eom only, and only
    `permnos` with `window` consecutive finite returns (full_history_mask). -> dict(Xc, delta, mu, T, permnos)."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    m1 = eom.year * 12 + eom.month - 1
    m0 = m1 - window + 1
    pm = np.unique(np.asarray(list(permnos), dtype=np.int64))
    if "_m" not in rets.columns:
        rets = rets.with_columns(_m=_risk_midx(pl.col("eom")))
    sub = rets.filter((pl.col("_m") >= m0) & (pl.col("_m") <= m1) & pl.col("permno").is_in(pm.tolist()))
    X = np.full((window, len(pm)), np.nan)
    if len(sub):
        X[sub["_m"].to_numpy() - m0, np.searchsorted(pm, sub["permno"].to_numpy())] = sub["ret"].to_numpy().astype(float)
    ok = np.isfinite(X).all(axis=0)
    X, pm = X[:, ok], pm[ok]
    if X.shape[1] == 0:
        return {"Xc": X, "delta": 0.0, "mu": 0.0, "T": window, "permnos": pm}
    Xc = X - X.mean(axis=0)
    delta = float(ledoit_wolf_shrinkage(X, assume_centered=False))
    mu = float((Xc ** 2).sum() / (X.shape[0] * X.shape[1]))
    return {"Xc": Xc, "delta": delta, "mu": mu, "T": window, "permnos": pm}


def _risk_variance(fac, w, idx):
    """Ex-ante variance w' Sigma w for weights w on factor columns idx."""
    x = fac["Xc"][:, idx] @ w
    return float((1 - fac["delta"]) / fac["T"] * (x @ x) + fac["delta"] * fac["mu"] * (w @ w))
