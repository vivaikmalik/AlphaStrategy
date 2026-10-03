import polars as pl
import numpy as np
from src.optimizer import optimize_month

def test_optimizer():
    np.random.seed(42)
    n = 600
    df = pl.DataFrame({
        "permno": np.arange(1000, 1000 + n),
        "score": np.random.randn(n),
        "beta_kf": np.random.normal(1.0, 0.2, n),
        "beta_var": np.abs(np.random.normal(0.05, 0.01, n)),
        "sector": np.random.choice([10, 20, 30, 40], n),
        "short_eligible": np.random.choice([0, 1], n, p=[0.2, 0.8])
    })
    
    w_prev = {1000: 0.01}
    weights = optimize_month(df, w_prev)
    
    assert len(weights) > 0, "Optimizer failed to find a solution"
    
    total_long = sum(w for w in weights.values() if w > 0)
    total_short = sum(abs(w) for w in weights.values() if w < 0)
    net_exposure = total_long - total_short
    gross_exposure = total_long + total_short
    
    assert abs(gross_exposure - 2.0) < 1e-4, f"Gross exposure {gross_exposure} != 200%"
    assert -0.2001 <= net_exposure <= 0.2001, f"Net exposure {net_exposure} outside 20% bounds"
    assert all(abs(w) <= 0.01501 for w in weights.values()), "Weight exceeds 1.5% limit"
    print("✅ CVXPY Clarabel Optimizer (Step 9) Verified.")

if __name__ == "__main__":
    test_optimizer()