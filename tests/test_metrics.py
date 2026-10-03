import polars as pl
import numpy as np
from src.metrics import calculate_metrics

def test_metrics():
    print("Running Performance Metrics Test...")
    returns_df = pl.DataFrame({
        "portfolio_return": [0.05, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, -0.02],
        "tb3ms_rate": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "sp500_return": [0.04, -0.03, 0.02, 0.02, 0.00, 0.01, -0.01, 0.03]
    })
    
    predictions_df = pl.DataFrame({
        "ret_exc_lead1m": [0.04, -0.01, 0.02, 0.03, -0.02, 0.01, 0.00, 0.02],
        "ret_hat": [0.03, -0.01, 0.01, 0.02, -0.01, 0.01, 0.01, 0.01]
    })
    
    metrics = calculate_metrics(returns_df, predictions_df)
    
    assert "OOS_R2" in metrics, "OOS R2 missing"
    assert "Information_Ratio" in metrics, "Information Ratio missing"
    assert "Beta" in metrics, "Beta missing"
    assert "Alpha_Annualized" in metrics, "Alpha missing"
    print(f"Metrics Sample -> IR: {metrics['Information_Ratio']:.2f} | Beta: {metrics['Beta']:.2f} | R2: {metrics['OOS_R2']:.4f}")
    print("✅ Performance Metrics (Step 13) Verified.")

if __name__ == "__main__":
    test_metrics()