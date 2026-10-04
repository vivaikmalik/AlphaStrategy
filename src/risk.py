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


def risk_model(frame, cfg):
    """frame: permno, eom, sector, ret_exc_lead1m, risk factors -> {"sectors": [...], "L": {eom: k x k Cholesky of F_t}}."""
    sectors = sorted(frame["sector"].fill_null("NA").unique().to_list())
    fs = {}                                                                 # eom s -> factor return, realised in month s+1
    for (s,), g in frame.filter(pl.col("ret_exc_lead1m").is_not_null()).sort("eom").group_by("eom", maintain_order=True):
        X = risk_exposures(g, sectors, cfg)
        fs[s] = np.linalg.lstsq(X, g["ret_exc_lead1m"].to_numpy().astype(float), rcond=None)[0]
    eoms = sorted(fs)
    known = dict(zip(eoms, pl.Series(eoms).dt.offset_by("1mo").dt.month_end().to_list()))   # f_s known at eom s+1
    L = {}
    for t in sorted(frame["eom"].unique().to_list()):
        past = [fs[s] for s in eoms if known[s] <= t][-cfg["risk_window"]:]
        if len(past) < cfg["risk_min_months"]:
            continue
        F = np.cov(np.array(past), rowvar=False)
        k = F.shape[0]                                     # few months vs ~22 factors: shrink toward avg variance
        F = 0.9 * F + 0.1 * np.trace(F) / k * np.eye(k) + 1e-8 * np.eye(k)
        L[t] = np.linalg.cholesky(F)
    return {"sectors": sectors, "L": L}
