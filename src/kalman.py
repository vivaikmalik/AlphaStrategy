import polars as pl
import numpy as np

def run_kalman_filter(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates state-space beta and variance per stock-month[cite: 2]."""
    obs_cols = ["beta_dimson_21d", "betadown_252d", "betabab_1260d", "beta_60m"]
    df = df.with_columns(pl.col("beta_60m").median().over(["sector", "eom"]).fill_null(pl.col("beta_60m").median().over("eom")).alias("m_t"))
    df = df.sort(["permno", "eom"])
    
    phi, q = 0.95, 0.02
    R = {"beta_dimson_21d": 0.05, "betadown_252d": 0.04, "betabab_1260d": 0.03, "beta_60m": 0.02}
    
    permnos, m_t, beta_60m = df["permno"].to_numpy(), df["m_t"].to_numpy(), df["beta_60m"].to_numpy()
    obs_matrix = df.select(obs_cols).to_numpy()
    beta_kf, beta_var = np.zeros(len(df)), np.zeros(len(df))
    
    state_est, cov_est = m_t[0], q
    
    for i in range(len(df)):
        if i == 0 or permnos[i] != permnos[i-1]:
            state_est, cov_est = m_t[i], q
        else:
            state_est = m_t[i] + phi * (state_est - m_t[i-1])
            cov_est = (phi ** 2) * cov_est + q
            
        has_data = False
        for j, col in enumerate(obs_cols):
            y_j = obs_matrix[i, j]
            if not np.isnan(y_j):
                has_data, R_j = True, R[col]
                innovation, S = y_j - state_est, cov_est + R_j
                K = cov_est / S
                state_est, cov_est = state_est + K * innovation, (1 - K) * cov_est
                
        if not has_data and not np.isnan(beta_60m[i]):
            beta_kf[i], beta_var[i] = beta_60m[i], R["beta_60m"]
        else:
            beta_kf[i], beta_var[i] = state_est, cov_est

    return df.with_columns([pl.Series("beta_kf", beta_kf), pl.Series("beta_var", beta_var)]).drop("m_t")