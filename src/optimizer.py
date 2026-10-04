"""
src/optimizer.py - Step 9: long/short portfolio optimiser (cvxpy + Clarabel), one convex problem per month.
w = wL - wS over the UNION of long candidates (top n by score) and short candidates (bottom n of the short-eligible subset).
"""
import cvxpy as cp
import numpy as np
import polars as pl
from src.risk import risk_exposures


def optimize_month(mdf, w_prev, lam_tc, lam_beta, cfg, lam_risk=0.0, lam_fac=0.0, risk=None):
    """mdf: one month with permno, score, short_eligible, beta_kf, beta_var, sector [, size_z] -> (weights dict, info dict); risk = (sectors, L) adds lam_fac * w'X F X'w."""
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
    if lam_fac > 0 and risk is not None:
        obj = obj - lam_fac * cp.sum_squares(risk[1].T @ (risk_exposures(c, risk[0], cfg).T @ w))   # factor-risk penalty (not in spec)
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


def run_book(scored, lam_tc, lam_beta, cfg, lam_risk=0.0, lam_fac=0.0, rm=None):
    """scored: all months (permno, eom, score, short_eligible, beta_kf, beta_var, sector) -> (weights df, log)."""
    w_prev, rows, log = {}, [], []
    for (eom,), mdf in scored.sort("eom").group_by("eom", maintain_order=True):
        risk = (rm["sectors"], rm["L"][eom]) if rm is not None and eom in rm["L"] else None
        w, info = optimize_month(mdf, w_prev, lam_tc, lam_beta, cfg, lam_risk, lam_fac, risk)
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


def _opt_ladder(cfg):
    """Relaxation steps (sector/size band, beta tol, vol multiplier): bands first, then beta, then vol (each stage keeps earlier relaxations)."""
    b0, s0, t0 = cfg["sector_band"], cfg["size_band"], cfg["v2_beta_tol"]
    steps = [(b0, s0, t0, 1.0)]
    b, s = b0, s0
    for _ in range(9):
        b, s = b + 0.05, s + 0.05
        steps.append((b, s, t0, 1.0))
    t = t0
    for _ in range(int(round((0.5 - t0) / 0.01))):
        t += 0.01
        steps.append((b, s, t, 1.0))
    m = 1.0
    while m < 4.0 - 1e-9:
        m = min(m * 1.25, 4.0)
        steps.append((b, s, t, m))
    return steps


def optimize_month_v2(mdf, w_prev, cfg, risk, ic):
    """One month -> (weights dict, info dict). Max alpha.w - v2_cost*|w - w_prev|_1 s.t. gross/net/sector/size/beta/ex-ante vol caps.
    risk = (sectors, L, b, xs_vol); alpha = max(ic, floor) * xs_vol * z (monthly expected excess return). L None -> vol constraint off."""
    sectors, L, b_vec, xs_vol = risk
    cap = cfg["v2_max_weight"]
    d = mdf.with_columns(pl.col("sector").fill_null("NA")).sort("permno").with_columns(
        ((pl.col("score") - pl.col("score").mean()) / pl.col("score").std()).fill_nan(0.0).fill_null(0.0).alias("s"),
        pl.col("spec_var").fill_nan(None).fill_null(pl.col("spec_var").median()).fill_null(0.0).clip(lower_bound=0.0),
        pl.col("size_z").fill_nan(None).fill_null(0.0))
    long_ids = set(d.sort(["s", "permno"], descending=[True, False]).head(cfg["n_long_cand"])["permno"])
    short_ids = set(d.filter(pl.col("short_eligible") & ~pl.col("permno").is_in(long_ids))
                    .sort(["s", "permno"]).head(cfg["n_short_cand"])["permno"])
    c = d.filter(pl.col("permno").is_in(long_ids | short_ids))
    ids = c["permno"].to_list(); n = len(ids)
    alpha = max(ic, cfg["v2_ic_floor"]) * xs_vol * c["s"].to_numpy()
    sz, spec = c["size_z"].to_numpy(), np.sqrt(c["spec_var"].to_numpy())
    ubL = np.array([cap if p in long_ids else 0.0 for p in ids])
    ubS = np.array([cap if p in short_ids else 0.0 for p in ids])
    G = (c["sector"].to_numpy()[None, :] == np.unique(c["sector"].to_numpy())[:, None]).astype(float)
    X = risk_exposures(c, sectors, cfg) if (L is not None or b_vec is not None) else None
    if b_vec is not None:
        beta = X @ np.asarray(b_vec, dtype=float)
    elif "beta_kf" in c.columns:
        beta = c["beta_kf"].fill_nan(None).fill_null(0.0).to_numpy()
    else:
        beta = None
    LtXt = L.T @ X.T if L is not None else None
    wp = np.clip(np.nan_to_num(np.array([w_prev.get(p, 0.0) for p in ids])), -cap, cap)
    dropped = sum(abs(v) for p, v in w_prev.items() if p not in set(ids))   # names that left the candidate set: sold to 0

    wL, wS = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    band, sband, tol, vol = (cp.Parameter(nonneg=True) for _ in range(4))
    w = wL - wS
    cons = [cp.sum(wL) + cp.sum(wS) <= cfg["gross"], cp.abs(cp.sum(w)) <= cfg["net_band"], wL <= ubL, wS <= ubS,
            cp.abs(G @ w) <= band, cp.abs(sz @ w) <= sband]
    if beta is not None:
        cons.append(cp.abs(beta @ w) <= tol)
    if LtXt is not None:
        cons.append(cp.norm(cp.hstack([LtXt @ w, cp.multiply(spec, w)])) <= vol)
    prob = cp.Problem(cp.Maximize(alpha @ w - cfg["v2_cost"] * cp.norm1(w - wp)), cons)
    vol0 = cfg["v2_vol_target"] / np.sqrt(12)

    def solve(bd, sb, t_, vm):
        band.value, sband.value, tol.value, vol.value = bd, sb, t_, vol0 * vm
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return False
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            return False
        v = w.value
        return bool(np.all(np.isfinite(v)) and np.abs(v).max() <= cap + 1e-4 and np.abs(v).sum() <= cfg["gross"] + 1e-3
                    and abs(v.sum()) <= cfg["net_band"] + 1e-3)                  # inaccurate solutions must still obey net

    for bd, sb, t_, vm in _opt_ladder(cfg):
        if solve(bd, sb, t_, vm):
            break
    else:
        print(f"[optimizer-v2] WARNING {mdf['eom'][0]}: infeasible even after full relaxation; holding previous weights")
        return dict(w_prev), {"status": "fallback_hold", "n_long": sum(v > 0 for v in w_prev.values()),
                              "n_short": sum(v < 0 for v in w_prev.values()), "gross": float(sum(abs(v) for v in w_prev.values())),
                              "net": float(sum(w_prev.values())), "turnover": 0.0, "vol": None, "beta_pred": None, "relaxed": ["hold"]}
    wv = np.where(np.abs(w.value) < 1e-7, 0.0, w.value)
    relaxed = []
    if bd > cfg["sector_band"] + 1e-9:
        relaxed.append(f"sector/size band {bd:.2f}/{sb:.2f}")
    if beta is not None and t_ > cfg["v2_beta_tol"] + 1e-9:
        relaxed.append(f"beta tol {t_:.2f}")
    if vm > 1.0 + 1e-9:
        relaxed.append(f"vol target x{vm:.2f}")
    var = float(np.sum((LtXt @ wv) ** 2) + np.sum((spec * wv) ** 2)) if LtXt is not None else None
    weights = {p: float(x) for p, x in zip(ids, wv) if x != 0.0}
    return weights, {"status": prob.status, "n_long": int((wv > 0).sum()), "n_short": int((wv < 0).sum()),
                     "gross": float(np.abs(wv).sum()), "net": float(wv.sum()), "turnover": float(np.abs(wv - wp).sum() + dropped),
                     "vol": float(np.sqrt(12 * var)) if var is not None else None,
                     "beta_pred": float(beta @ wv) if beta is not None else None, "relaxed": relaxed}


def run_book_v2(scored, cfg, rm, ic):
    """scored: all months -> (weights df, log). ic: float or dict eom -> float; rm: risk model dict (sectors, L, xs_vol, optional b)."""
    w_prev, rows, log = {}, [], []
    for (eom,), mdf in scored.sort("eom").group_by("eom", maintain_order=True):
        if eom in rm["L"] and eom in rm["xs_vol"]:
            risk = (rm["sectors"], rm["L"][eom], rm.get("b", {}).get(eom), rm["xs_vol"][eom])
        else:
            print(f"[optimizer-v2] WARNING {eom}: no risk model, vol constraint off")
            risk = (rm["sectors"], None, None, rm["xs_vol"].get(eom, 0.0))
        w, info = optimize_month_v2(mdf, w_prev, cfg, risk, ic[eom] if isinstance(ic, dict) else ic)
        if info["relaxed"]:
            print(f"[optimizer-v2] {eom}: relaxed {', '.join(info['relaxed'])}")
        log.append({"eom": eom, **info})
        rows += [(p, eom, x) for p, x in w.items()]
        w_prev = w
    wdf = pl.DataFrame(rows, schema={"permno": pl.Int64, "eom": pl.Date, "weight": pl.Float64}, orient="row")
    return wdf, log


def _opt_cap(w, cap, total):
    """Scale positive array w to sum `total`, cap each at `cap`, redistribute the excess to uncapped names. -> (w, capped?)"""
    w = w / w.sum() * total
    capped = False
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        capped = True
        w = np.where(over, cap, w)
        free = ~over & (w < cap - 1e-12)
        if not free.any() or total - w[~free].sum() <= 0:
            break
        w[free] *= (total - w[~free].sum()) / w[free].sum()
    return w, capped


def run_book_v3(scored, cfg, smooth):
    """Sector-neutral equal-weight tail book on smoothed z-scores, beta-neutral legs. scored: permno, eom, score, short_eligible,
    sector, beta_kf -> (weights df[permno, eom, weight], log list of dicts)."""
    prev_s, w_prev, rows, log = {}, {}, [], []
    for (eom,), mdf in scored.sort("eom").group_by("eom", maintain_order=True):
        sc = mdf["score"].to_numpy().astype(float)
        sc = np.where(np.isfinite(sc), sc, np.nan)
        sd = np.nanstd(sc, ddof=1) if np.isfinite(sc).sum() > 1 else 0.0
        z = np.nan_to_num((sc - np.nanmean(sc)) / sd) if sd > 0 else np.zeros(len(sc))
        pm = mdf["permno"].to_list()
        s = np.array([smooth * zi + (1 - smooth) * prev_s.get(p, zi) for p, zi in zip(pm, z)])
        prev_s = dict(zip(pm, s))
        beta = mdf["beta_kf"].to_numpy().astype(float)
        beta = np.where(np.isfinite(beta), beta, np.nanmedian(beta) if np.isfinite(beta).any() else 1.0)
        elig = mdf["short_eligible"].fill_null(False).to_numpy()
        sec = np.array(mdf["sector"].fill_null("NA").cast(pl.Utf8).to_list())
        tail = cfg["v3_tail"]
        while True:                               # competition rule: <= 500 names -> shrink the tail until it fits
            longs, shorts, budget = [], [], []    # per kept sector: index arrays + stock count
            for sname in np.unique(sec):
                idx = np.where(sec == sname)[0]
                nl = int(np.ceil(tail * len(idx)))
                top = idx[np.argsort(-s[idx], kind="stable")[:nl]]
                cand = idx[elig[idx] & ~np.isin(idx, top)]
                ns = int(np.ceil(tail * elig[idx].sum()))
                bot = cand[np.argsort(s[cand], kind="stable")[:ns]]
                if len(top) and len(bot):
                    longs.append(top); shorts.append(bot); budget.append(len(idx))
            if sum(len(x) for x in longs + shorts) <= 500 or tail <= 0.02:
                break
            tail -= 0.01
        flags = []
        if not longs:
            log.append({"eom": eom, "n_long": 0, "n_short": 0, "gross": 0.0, "net": 0.0, "beta_exposure": 0.0, "turnover": sum(abs(x) for x in w_prev.values()), "flags": ["empty"]})
            w_prev = {}
            continue
        budget = np.array(budget, float) / sum(budget)
        li, si = np.concatenate(longs), np.concatenate(shorts)
        lw = np.concatenate([np.full(len(a), b / len(a)) for a, b in zip(longs, budget)])
        sw = np.concatenate([np.full(len(a), b / len(a)) for a, b in zip(shorts, budget)])
        bl, bs = float(lw @ beta[li]), float(sw @ beta[si])
        g = cfg["gross"]
        L = g * bs / (bl + bs) if bl + bs > 1e-9 else g / 2
        if abs(2 * L - g) > cfg["net_band"]:
            L = (g + np.sign(2 * L - g) * cfg["net_band"]) / 2
            flags.append("net_clipped")
        S = g - L
        lw, c1 = _opt_cap(lw, cfg["max_weight"], L)
        sw, c2 = _opt_cap(sw, cfg["max_weight"], S)
        if c1 or c2:
            flags.append("max_weight_capped")
        n_names = len(li) + len(si)
        if n_names > 500:
            flags.append("names_gt_500")
            print(f"[optimizer-v3] {eom}: {n_names} names > 500")
        w = {int(pm[i]): float(x) for i, x in zip(li, lw)}
        w.update({int(pm[i]): -float(x) for i, x in zip(si, sw)})
        turn = sum(abs(w.get(p, 0.0) - w_prev.get(p, 0.0)) for p in set(w) | set(w_prev))
        log.append({"eom": eom, "n_long": len(li), "n_short": len(si), "gross": float(lw.sum() + sw.sum()), "net": float(lw.sum() - sw.sum()),
                    "beta_exposure": float(lw @ beta[li] - sw @ beta[si]), "turnover": float(turn), "flags": flags})
        rows += [(p, eom, x) for p, x in w.items()]
        w_prev = w
    wdf = pl.DataFrame(rows, schema={"permno": pl.Int64, "eom": pl.Date, "weight": pl.Float64}, orient="row")
    return wdf, log
