"""
src/optimizer.py - Step 9: long/short portfolio optimiser (cvxpy + Clarabel), one convex problem per month.
w = wL - wS over the UNION of long candidates (top n by score) and short candidates (bottom n of the short-eligible subset).
"""
import cvxpy as cp
import numpy as np
import polars as pl


def optimize_month(mdf, w_prev, lam_tc, lam_beta, cfg, lam_risk=0.0):
    """mdf: one month with permno, score, short_eligible, beta_kf, beta_var, sector [, size_z] -> (weights dict, info dict)."""
    d = mdf.with_columns(pl.col("sector").fill_null("NA")).sort("permno").with_columns(
        ((pl.col("score") - pl.col("score").mean()) / pl.col("score").std()).fill_nan(0.0).fill_null(0.0).alias("s"),
        pl.col("beta_kf").fill_nan(None).fill_null(pl.col("beta_kf").median()),
        pl.col("beta_var").fill_nan(None).fill_null(pl.col("beta_var").median()))
    has_size = "size_z" in d.columns                                       # not in spec: size neutrality only if present
    if has_size:
        d = d.with_columns(pl.col("size_z").fill_nan(None).fill_null(0.0))
    long_ids = set(d.sort(["s", "permno"], descending=[True, False]).head(cfg["n_long_cand"])["permno"])
    short_ids = set(d.filter(pl.col("short_eligible") & ~pl.col("permno").is_in(long_ids))     # no name on both sides:
                    .sort(["s", "permno"]).head(cfg["n_short_cand"])["permno"])               # wL=wS would fake gross
    c = d.filter(pl.col("permno").is_in(long_ids | short_ids))
    ids = c["permno"].to_list(); n = len(ids)
    s, bk, bv = c["s"].to_numpy(), c["beta_kf"].to_numpy(), c["beta_var"].to_numpy()
    sz = c["size_z"].to_numpy() if has_size else None
    ubL = np.array([cfg["max_weight"] if p in long_ids else 0.0 for p in ids])
    ubS = np.array([cfg["max_weight"] if p in short_ids else 0.0 for p in ids])
    G = (c["sector"].to_numpy()[None, :] == np.unique(c["sector"].to_numpy())[:, None]).astype(float)
    wp = np.clip(np.nan_to_num(np.array([w_prev.get(p, 0.0) for p in ids])), -cfg["max_weight"], cfg["max_weight"])
    dropped = sum(abs(v) for p, v in w_prev.items() if p not in set(ids))   # names that left the candidate set: sold to 0

    wL, wS = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    tol, band, sband = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
    w = wL - wS
    obj = s @ w - lam_tc * cp.norm1(w - wp) - lam_beta * (bv @ cp.square(w)) - lam_risk * cp.sum_squares(w)   # L2 risk penalty (not in spec); dropped-name sales are a constant
    cons = [cp.sum(wL) + cp.sum(wS) == cfg["gross"], cp.abs(cp.sum(w)) <= cfg["net_band"], wL <= ubL, wS <= ubS,
            cp.abs(bk @ w) <= tol, cp.abs(G @ w) <= band]
    if has_size:
        cons.append(cp.abs(sz @ w) <= sband)                               # not in spec: size-neutral book
    prob = cp.Problem(cp.Maximize(obj), cons)

    def solve(t_, b_, sb_):
        tol.value, band.value, sband.value = t_, b_, sb_
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return False                                                   # treated as a failed attempt
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            return False
        v = w.value                                                        # inaccurate solutions can hold garbage weights
        return bool(np.all(np.isfinite(v)) and np.abs(v).max() <= cfg["max_weight"] + 1e-4
                    and abs(np.abs(v).sum() - cfg["gross"]) <= 1e-3)

    t, b, sb, no_beta = cfg["beta_tol"], cfg["sector_band"], cfg["size_band"], 1e6
    if not solve(t, b, sb):
        # Not in spec: if the month is infeasible even with NO beta limit, the sector nets bind,
        # so widen the sector (and size) band in 0.05 steps first (logged), then relax beta from 0.02 as the spec says.
        while not solve(no_beta, b, sb):
            b += 0.05
            sb += 0.05
            if b > 2.0:                                                    # should not happen: save inputs, hold last book
                eom = mdf["eom"][0]
                path = cfg["cache_dir"] / f"optimizer_fail_{eom}.parquet"
                mdf.write_parquet(path)
                bad = {k: int((~np.isfinite(v)).sum()) for k, v in (("s", s), ("beta_kf", bk), ("beta_var", bv))}
                print(f"[optimizer] WARNING {eom}: infeasible ({prob.status}) even with sector band 2.0 and no beta limit; "
                      f"rows={d.height} long_cand={len(long_ids)} short_cand={len(short_ids)} non-finite={bad}; "
                      f"holding previous weights; inputs saved to {path}")
                return dict(w_prev), {"beta_tol": None, "sector_band": b, "size_band": sb, "status": "fallback_hold",
                                      "n_long": sum(v > 0 for v in w_prev.values()),
                                      "n_short": sum(v < 0 for v in w_prev.values()), "turnover": 0.0}
        while not solve(t, b, sb):                                           # spec: relax beta tolerance in 0.01 steps
            t += cfg["beta_tol_step"]                                      # terminates: feasible without a beta limit
    wv = np.where(np.abs(w.value) < 1e-7, 0.0, w.value)
    weights = {p: float(x) for p, x in zip(ids, wv) if x != 0.0}
    turnover = float(np.abs(wv - wp).sum() + dropped)
    return weights, {"beta_tol": round(t, 6), "sector_band": round(b, 6), "size_band": round(sb, 6), "status": prob.status, "n_long": int((wv > 0).sum()),
                     "n_short": int((wv < 0).sum()), "turnover": turnover}


def run_book(scored, lam_tc, lam_beta, cfg, lam_risk=0.0):
    """scored: all months (permno, eom, score, short_eligible, beta_kf, beta_var, sector) -> (weights df, log)."""
    w_prev, rows, log = {}, [], []
    for (eom,), mdf in scored.sort("eom").group_by("eom", maintain_order=True):
        w, info = optimize_month(mdf, w_prev, lam_tc, lam_beta, cfg, lam_risk)
        if info["beta_tol"] is not None and info["beta_tol"] > cfg["beta_tol"]:
            print(f"[optimizer] {eom}: beta tolerance relaxed to {info['beta_tol']:.2f}")
        if info["sector_band"] > cfg["sector_band"]:
            print(f"[optimizer] {eom}: sector band relaxed to {info['sector_band']:.2f} (infeasible at 0.10)")
        if info["size_band"] > cfg["size_band"]:
            print(f"[optimizer] {eom}: size band relaxed to {info['size_band']:.2f}")
        log.append({"eom": eom, **info})
        rows += [(p, eom, x) for p, x in w.items()]
        w_prev = w
    wdf = pl.DataFrame(rows, schema={"permno": pl.Int64, "eom": pl.Date, "weight": pl.Float64}, orient="row")
    return wdf, log
