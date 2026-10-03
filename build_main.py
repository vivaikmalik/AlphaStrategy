import os

def build_main():
    """Compiles src/ into a single MAIN.py and appends the execution orchestrator."""
    modules = ["config.py", "data.py", "gru.py", "text.py", "kalman.py", "ranker.py", "optimizer.py", "metrics.py"]
    
    with open("MAIN.py", "w") as outfile:
        outfile.write("# MAIN.py - Auto-compiled AlphaStrategy Submission\n\n")
        
        # 1. Concatenate all source modules
        for module in modules:
            path = os.path.join("src", module)
            if os.path.exists(path):
                with open(path, "r") as infile:
                    inside_multiline_import = False
                    
                    for line in infile.readlines():
                        # Start of a local import
                        if line.startswith("from src."):
                            if "(" in line and ")" not in line:
                                inside_multiline_import = True
                            continue
                        
                        # Inside a multi-line import block
                        if inside_multiline_import:
                            if ")" in line:
                                inside_multiline_import = False
                            continue
                            
                        # Normal lines
                        outfile.write(line)
                    outfile.write("\n\n")
                    
        # 2. Append the actual execution engine
        execution_block = """
if __name__ == '__main__':
    import polars as pl
    import pandas as pd
    import os
    
    print('Starting AlphaBERT End-to-End Pipeline...')
    
    # 1. Load Data (Steps 1-3)
    print('Loading datasets...')
    df = pl.read_parquet("cache/preprocessed_features.parquet") 
    df_8k = pl.read_parquet("data/8k_20150101_20260831_identified.parquet")
    
    # 2. Text Features & FinBERT (Steps 6-7)
    print('Processing Text Features via FinBERT...')
    df_text = process_text_features(df_8k, device="cuda")
    df = df.join(df_text, on=["permno", "eom"], how="left")
    
    # 3. Kalman Filter Betas (Step 8)
    print('Running Kalman Filter for market betas...')
    df = run_kalman_filter(df)
    
    # 4. XGBoost Ranker (Steps 5 & 10)
    print('Training XGBoost Ranker...')
    exclude_cols = ["permno", "eom", "ret_exc_lead1m", "label_decile", "score", "ret_hat"]
    features = [col for col in df.columns if col not in exclude_cols]
    df_scored, final_model = train_and_rank(df, features=features, target_col="ret_exc_lead1m")
    
    # 5. CVXPY Optimizer (Step 9)
    print('Running CVXPY Clarabel Optimizer (2021-2026)...')
    final_holdings = []
    w_prev = {}
    
    # Filter for OOS testing years (2021+)
    test_months = df_scored.filter(pl.col("eom") >= pl.date(2021, 1, 1))["eom"].unique().sort()
    
    for month in test_months:
        month_df = df_scored.filter(pl.col("eom") == month)
        weights = optimize_month(month_df, w_prev)
        for permno, w in weights.items():
            final_holdings.append({"Date": month, "PERMNO": permno, "WEIGHT": w})
        w_prev = weights
        
    # 6. Export Submission Files (Step 14)
    print('Exporting output/holdings.csv...')
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(final_holdings).to_csv("output/holdings.csv", index=False)
    print('✅ Pipeline complete. Holdings saved to output/holdings.csv')
"""
        outfile.write(execution_block)
        
    print("✅ MAIN.py successfully built with execution loop.")

if __name__ == "__main__":
    build_main()