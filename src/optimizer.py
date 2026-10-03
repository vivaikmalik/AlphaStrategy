import cvxpy as cp
import numpy as np
import polars as pl

def optimize_month(df: pl.DataFrame, w_prev: dict, lambda_tc: float = 0.01, lambda_beta: float = 0.01) -> dict:
    """Returns optimal long/short weights bounded by spec constraints[cite: 2, 3]."""
    df = df.with_columns(((pl.col("score") - pl.col("score").mean()) / pl.col("score").std()).alias("s"))
    candidates = pl.concat([
        df.sort("s", descending=True).head(250),
        df.filter(pl.col("short_eligible") == 1).sort("s", descending=False).head(250)
    ]).unique(subset=["permno"])
    
    n, permnos = len(candidates), candidates["permno"].to_list()
    s, beta_kf, beta_var = candidates["s"].to_numpy(), candidates["beta_kf"].to_numpy(), candidates["beta_var"].to_numpy()
    sectors = candidates["sector"].to_numpy()
    w_prev_arr = np.array([w_prev.get(p, 0.0) for p in permnos])
    
    wL, wS = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    w = wL - wS
    
    objective = cp.Maximize(s @ w - lambda_tc * cp.norm1(w - w_prev_arr) - lambda_beta * cp.sum(cp.multiply(beta_var, cp.square(w))))
    
    constraints = [
        cp.sum(wL) + cp.sum(wS) == 2.0, 
        cp.sum(w) >= -0.20, cp.sum(w) <= 0.20,
        wL <= 0.015, wS <= 0.015
    ]
    for sec in np.unique(sectors):
        sec_mask = (sectors == sec).astype(float)
        constraints.extend([sec_mask @ w >= -0.10, sec_mask @ w <= 0.10])
        
    beta_tol, solved = 0.02, False
    while beta_tol <= 1.0 and not solved:
        prob = cp.Problem(objective, constraints + [beta_kf @ w >= -beta_tol, beta_kf @ w <= beta_tol])
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]: solved = True
            else: beta_tol += 0.01
        except: beta_tol += 0.01
            
    if not solved: return {}
    weights = np.round(w.value, 6)
    return {permnos[i]: weights[i] for i in range(n) if abs(weights[i]) > 1e-6}