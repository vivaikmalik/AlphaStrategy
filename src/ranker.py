"""
src/ranker.py - Step 5: XGBoost Ranker
Trains an XGBoost regression model on original features + GRU embeddings.
Outputs a cross-sectionally ranked alpha signal [-1, 1] for portfolio construction.
"""

import polars as pl
import xgboost as xgb
import numpy as np

def train_and_rank(df: pl.DataFrame, factors: list, d: int, target_col: str = "ret_exc_lead1m"):
    """
    Trains XGBoost using early stopping on the 2019-2020 validation set.
    Generates a cross-sectional alpha signal ranked strictly between -1 (Short) and 1 (Long).
    """
    # 1. Feature Assembly
    gru_cols = [f"gru_{i+1}" for i in range(d)]
    features = factors + gru_cols
    
    # 2. Data Cleaning - drop rows missing the target or GRU embeddings
    df_clean = df.drop_nulls(subset=[target_col] + gru_cols)
    
    # 3. Chronological Splits
    train_mask = df_clean["eom"] <= pl.date(2018, 11, 30)
    val_mask = (df_clean["eom"] >= pl.date(2019, 1, 1)) & (df_clean["eom"] <= pl.date(2020, 12, 31))
    
    train_df = df_clean.filter(train_mask)
    val_df = df_clean.filter(val_mask)
    
    X_train = train_df.select(features).to_pandas()
    y_train = train_df.select(target_col).to_numpy().ravel()
    
    X_val = val_df.select(features).to_pandas()
    y_val = val_df.select(target_col).to_numpy().ravel()
    
    print(f"Training XGBoost on {len(X_train)} samples, validating on {len(X_val)} samples...")

    # 4. Hyperparameter Grid Search
    best_model = None
    best_val_loss = float('inf')
    best_params = {}
    
    for lr in [0.01, 0.05]:
        for depth in [3, 5]:
            # Bind to Grace Blackwell GPU via device="cuda"
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=lr,
                max_depth=depth,
                tree_method="hist",
                device="cuda", 
                eval_metric="rmse",
                early_stopping_rounds=20
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            val_loss = model.best_score
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_params = {'lr': lr, 'depth': depth, 'rmse': val_loss}

    print(f"Best XGBoost Params -> LR: {best_params['lr']} | Max Depth: {best_params['depth']} | Val RMSE: {best_params['rmse']:.4f}")

    # 5. Generate Raw Predictions for the entire dataset
    print("Scoring universe and cross-sectionally ranking signals...")
    X_all = df_clean.select(features).to_pandas()
    df_clean = df_clean.with_columns(
        pl.Series("raw_prediction", best_model.predict(X_all))
    )
    
    # 6. Cross-Sectional Ranking (The Alpha Signal)
    # Ranks the predictions dynamically within each month (eom)
    # Scales the rank to exactly [-1, 1] where 1 is the highest predicted return
    ranked = df_clean.with_columns([
        pl.col("raw_prediction").rank(method="dense", descending=False).over("eom").alias("rank_temp")
    ])
    
    final_df = ranked.with_columns([
        ((pl.col("rank_temp") - 1) / (pl.col("rank_temp").max().over("eom") - 1) * 2 - 1).alias("alpha_signal")
    ]).drop("rank_temp")

    return final_df, best_model