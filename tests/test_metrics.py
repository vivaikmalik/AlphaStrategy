import polars as pl
import numpy as np
from src.metrics import calculate_metrics

def test_metrics():
    df = pl.DataFrame({
        "portfolio_return": [0.05, -0.02, 0.03, 0.01, -0.01],
        "tb3ms_rate": [3.0, 3.0, 3.0, 3.0, 3.0], # 3% annualized
        "sp500_return": [0.04, -0.03, 0.02, 0.02, 0.00]
    })
    
    metrics = calculate_metrics(df)
    
    assert "IR" in metrics, "Information Ratio missing"
    assert "Beta" in metrics, "Realized Beta missing"
    assert "Max_Drawdown" in metrics, "Max Drawdown missing"
    print(f"Metrics: IR={metrics['IR']:.2f}, Beta={metrics['Beta']:.2f}, DD={metrics['Max_Drawdown']:.2f}")
    print("✅ Performance Metrics (Step 13) Verified.")

if __name__ == "__main__":
    test_metrics()