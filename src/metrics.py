import polars as pl
import numpy as np
import statsmodels.api as sm

def calculate_metrics(returns_df: pl.DataFrame, predictions_df: pl.DataFrame) -> dict:
    """Calculates strictly required Hackathon metrics[cite: 3, 12, 18]."""
    
    # 1. Zero-Benchmark OOS R2[cite: 3, 18]
    r = predictions_df["ret_exc_lead1m"].to_numpy()
    r_hat = predictions_df["ret_hat"].to_numpy()
    oos_r2 = 1 - (np.sum((r - r_hat)**2) / np.sum(r**2))
    
    # 2. Returns and Hurdle[cite: 12]
    df = returns_df.with_columns(
        (pl.col("tb3ms_rate") / 1200 + 0.04 / 12).alias("benchmark_return")
    ).with_columns(
        (pl.col("portfolio_return") - pl.col("benchmark_return")).alias("active_return"),
        (pl.col("sp500_return") - (pl.col("tb3ms_rate") / 1200)).alias("sp500_excess"),
        (pl.col("portfolio_return") - (pl.col("tb3ms_rate") / 1200)).alias("portfolio_excess")
    )
    
    active_returns = df["active_return"].to_numpy()
    ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(12)
    
    # 3. Newey-West Alpha and Beta (3 Lags)[cite: 3]
    y = df["portfolio_excess"].to_numpy()
    X = sm.add_constant(df["sp500_excess"].to_numpy())
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    alpha, beta = results.params[0], results.params[1]
    alpha_se, beta_se = results.bse[0], results.bse[1]
    
    # 4. Drawdown
    cum_returns = (1 + df["portfolio_return"]).cum_prod()
    max_drawdown = (cum_returns / cum_returns.cum_max() - 1).min()
    
    return {
        "OOS_R2": oos_r2,
        "Information_Ratio": ir,
        "Alpha_Annualized": alpha * 12,
        "Alpha_SE": alpha_se,
        "Beta": beta,
        "Beta_SE": beta_se,
        "Max_Drawdown": max_drawdown
    }