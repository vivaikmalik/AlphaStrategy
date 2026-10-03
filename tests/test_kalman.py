import polars as pl
import numpy as np
from src.kalman import run_kalman_filter

def test_kalman():
    print("Running Kalman Filter Test...")
    df = pl.DataFrame({
        "permno": [1, 1, 1],
        "eom": [pl.date(2015, 1, 31), pl.date(2015, 2, 28), pl.date(2015, 3, 31)],
        "sector": [10, 10, 10],
        "beta_dimson_21d": [1.1, np.nan, 1.3],
        "betadown_252d": [1.05, 1.15, np.nan],
        "betabab_1260d": [1.2, 1.2, 1.2],
        "beta_60m": [1.15, 1.1, 1.2]
    })
    
    df_filtered = run_kalman_filter(df)
    
    assert "beta_kf" in df_filtered.columns, "Missing state estimate"
    assert "beta_var" in df_filtered.columns, "Missing variance estimate"
    assert df_filtered["beta_kf"].is_null().sum() == 0, "Kalman filter left NaNs"
    print("✅ Kalman Filter (Step 8) Verified.")

if __name__ == "__main__":
    test_kalman()