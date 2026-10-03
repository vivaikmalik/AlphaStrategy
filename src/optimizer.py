"""
src/optimizer.py - Step 9: long/short portfolio optimiser (cvxpy + Clarabel), one convex problem per month.
w = wL - wS over the UNION of long candidates (top n by score) and short candidates (bottom n of the short-eligible subset).
"""
import cvxpy as cp
import numpy as np
import polars as pl


def optimize_month(mdf, w_prev, lam_tc, lam_beta, cfg):
    """mdf: one month with permno, score, short_eligible, beta_kf, beta_var, sector -> (weights dict, info dict)."""
    d = mdf.with_columns(pl.col("sector").fill_null("NA")).sort("permno").with_columns(
        ((pl.col("score") - pl.col("score").mean()) / pl.col("score").std()).alias("s"),
        pl.col("beta_kf").fill_nan(None).fill_null(pl.col("beta_kf").median()),
        pl.col("beta_var").fill_nan(None).fill_null(pl.col("beta_var").median()))
    long_ids = set(d.sort(["s", "permno"], descending=[True, False]).head(cfg["n_long_cand"])["permno"])
    short_ids = set(d.filter(pl.col("short_eligible")).sort(["s", "permno"]).head(cfg["n_short_cand"])["permno"])
    c = d.filter(pl.col("permno").is_in(long_ids | short_ids))
    ids = c["permno"].to_list(); n = len(ids)
    s, bk, bv = c["s"].to_numpy(), c["beta_kf"].to_numpy(), c["beta_var"].to_numpy()
    ubL = np.array([cfg["max_weight"] if p in long_ids else 0.0 for p in ids])
    ubS = np.array([cfg["max_weight"] if p in short_ids else 0.0 for p in ids])
    G = (c["sector"].to_numpy()[None, :] == np.unique(c["sector"].to_numpy())[:, None]).astype(float)
    wp = np.array([w_prev.get(p, 0.0) for p in ids])
    dropped = sum(abs(v) for p, v in w_prev.items() if p not in set(ids))   # names that left the candidate set: sold to 0

    wL, wS, tol = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True), cp.Parameter(nonneg=True)
    w = wL - wS
    obj = s @ w - lam_tc * cp.norm1(w - wp) - lam_beta * (bv @ cp.square(w))   # dropped-name sales are a constant
    cons = [cp.sum(wL) + cp.sum(wS) == cfg["gross"], cp.abs(cp.sum(w)) <= cfg["net_band"], wL <= ubL, wS <= ubS,
            cp.abs(bk @ w) <= tol, cp.abs(G @ w) <= cfg["sector_band"]]
    prob = cp.Problem(cp.Maximize(obj), cons)
    t = cfg["beta_tol"]
    while True:
        tol.value = t
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            pass                                                           # treated as a failed attempt; relaxed below
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
            break
        t += cfg["beta_tol_step"]
        if t > 1.0:
            raise RuntimeError(f"optimizer infeasible even with beta tol {t:.2f}: status {prob.status}")
    wv = np.where(np.abs(w.value) < 1e-7, 0.0, w.value)
    weights = {p: float(x) for p, x in zip(ids, wv) if x != 0.0}
    turnover = float(np.abs(wv - wp).sum() + dropped)
    return weights, {"beta_tol": round(t, 6), "status": prob.status, "n_long": int((wv > 0).sum()),
                     "n_short": int((wv < 0).sum()), "turnover": turnover}


def run_book(scored, lam_tc, lam_beta, cfg):
    """scored: all months (permno, eom, score, short_eligible, beta_kf, beta_var, sector) -> (weights df, log)."""
    w_prev, rows, log = {}, [], []
    for (eom,), mdf in scored.sort("eom").group_by("eom", maintain_order=True):
        w, info = optimize_month(mdf, w_prev, lam_tc, lam_beta, cfg)
        if info["beta_tol"] > cfg["beta_tol"]:
            print(f"[optimizer] {eom}: beta tolerance relaxed to {info['beta_tol']:.2f}")
        log.append({"eom": eom, **info})
        rows += [(p, eom, x) for p, x in w.items()]
        w_prev = w
    wdf = pl.DataFrame(rows, schema={"permno": pl.Int64, "eom": pl.Date, "weight": pl.Float64}, orient="row")
    return wdf, log
