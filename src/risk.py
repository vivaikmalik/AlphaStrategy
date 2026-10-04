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
