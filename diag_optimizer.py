import sys, glob, numpy as np, polars as pl, cvxpy as cp
f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("cache/optimizer_fail_*.parquet"))[0]
m = pl.read_parquet(f); print(f, m.shape, dict(m.schema))
print("dup permnos:", m.height - m["permno"].n_unique(), "| short_eligible:", m["short_eligible"].sum(), "| null score:", m["score"].null_count())
print(m.select("score", "beta_kf", "beta_var").describe())
d = m.with_columns(pl.col("sector").fill_null("NA")).sort("permno").with_columns(
    ((pl.col("score") - pl.col("score").mean()) / pl.col("score").std()).fill_nan(0.0).fill_null(0.0).alias("s"))
L = set(d.sort(["s", "permno"], descending=[True, False]).head(250)["permno"])
S = set(d.filter(pl.col("short_eligible") & ~pl.col("permno").is_in(L)).sort(["s", "permno"]).head(250)["permno"])
c = d.filter(pl.col("permno").is_in(L | S)); ids = c["permno"].to_list(); n = len(ids)
print("cands:", len(L), len(S), "rows:", n)
ubL = np.array([0.015 if p in L else 0.0 for p in ids]); ubS = np.array([0.015 if p in S else 0.0 for p in ids])
print("capacity long/short:", ubL.sum(), ubS.sum())
wL, wS = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True); w = wL - wS
base = [cp.sum(wL) + cp.sum(wS) == 2, wL <= ubL, wS <= ubS]
for name, cons in [("gross+caps", base), ("+net", base + [cp.abs(cp.sum(w)) <= 0.2]),
                   ("+beta 1e6", base + [cp.abs(cp.sum(w)) <= 0.2, cp.abs(c["beta_kf"].to_numpy() @ w) <= 1e6])]:
    for solver in [cp.CLARABEL, cp.SCS]:
        p = cp.Problem(cp.Maximize(c["s"].to_numpy() @ w), cons)
        try: p.solve(solver=solver); st = p.status
        except Exception as e: st = f"error {e}"
        print(f"{name:12s} {solver:9s} -> {st}")
from src.optimizer import optimize_month
from src.config import CONFIG
for lt, lb in [(0.0, 1000.0), (0.0, 0.0)]:
    w_, info = optimize_month(m, {}, lt, lb, CONFIG)
    print(f"optimize_month lam_tc={lt} lam_beta={lb} -> {info}")
