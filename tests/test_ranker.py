"""
src/ranker.py - Step 5: XGBoost Ranker (Production Spec)
"""
import polars as pl
import xgboost as xgb
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

def compute_rank_ic(model, X_val, df_val):
    """Evaluates validation mean monthly rank IC[cite: 2]."""
    preds = model.predict(X_val)
    # XGBRanker output doesn't have a fixed scale, we rank it cross-sectionally
    df_temp = df_val.with_columns(pl.Series("score", preds))
    
    # Calculate Spearman rank correlation (IC) per month
    ic_list = []
    for eom, group in df_temp.group_by("eom"):
        ic, _ = spearmanr(group["score"], group["ret_exc_lead1m"])
        if not np.isnan(ic):
            ic_list.append(ic)
            
    return np.mean(ic_list)

def train_and_rank(df: pl.DataFrame, features: list, target_col: str = "ret_exc_lead1m"):
    """
    Trains pairwise XGBRanker using exact hackathon specifications[cite: 2].
    """
    # 1. Label Assignment: Within-month decile (0-9)[cite: 2]
    df = df.drop_nulls(subset=[target_col] + features)
    df = df.with_columns(
        (pl.col(target_col).rank("ordinal").over("eom") / pl.col(target_col).count().over("eom") * 10)
        .floor().clip(0, 9).cast(pl.Int32).alias("label_decile")
    )
    
    # Sort rows by eom (required for pairwise ranking query groups)[cite: 2]
    df = df.sort("eom")
    
    # Chronological Splits
    train_mask = df["eom"] <= pl.date(2018, 11, 30)
    val_mask = (df["eom"] >= pl.date(2019, 1, 1)) & (df["eom"] <= pl.date(2020, 12, 31))
    
    train_df = df.filter(train_mask)
    val_df = df.filter(val_mask)
    
    # Extract arrays
    X_train = train_df.select(features).to_pandas()
    y_train = train_df["label_decile"].to_numpy()
    qid_train = train_df["eom"].dt.epoch().to_numpy() # Grouping IDs for ranking[cite: 2]
    
    X_val = val_df.select(features).to_pandas()
    y_val = val_df["label_decile"].to_numpy()
    qid_val = val_df["eom"].dt.epoch().to_numpy()
    
    # 2. Grid Search[cite: 2]
    best_ic = -float('inf')
    best_params = {}
    best_trees = 0
    best_model_val = None
    
    print("Executing hyperparameter grid search on max_depth [3, 4, 5]...")
    for depth in [3, 4, 5]:
        # Fixed settings from spec[cite: 2]
        model = xgb.XGBRanker(
            objective="rank:pairwise",
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.5,
            min_child_weight=200,
            reg_lambda=10,
            tree_method="hist",
            device="cuda",
            max_depth=depth,
            n_estimators=2000,
            early_stopping_rounds=50,
            eval_metric="ndcg"
        )
        
        # Fit with evaluation every 50 trees
        model.fit(
            X_train, y_train, qid=qid_train,
            eval_set=[(X_val, y_val)], eval_qid=[qid_val],
            verbose=False
        )
        
        current_ic = compute_rank_ic(model, X_val, val_df)
        print(f"Depth {depth} -> Trees: {model.best_iteration + 1} | Val IC: {current_ic:.4f}")
        
        if current_ic > best_ic:
            best_ic = current_ic
            best_params = {"max_depth": depth}
            best_trees = model.best_iteration + 1
            best_model_val = model

    print(f"Optimal XGBoost Params: Depth {best_params['max_depth']}, Trees {best_trees}")
    
    # 3. Return Forecast Regression (for OOS R²)[cite: 2]
    val_preds = best_model_val.predict(X_val)
    val_df_pred = val_df.with_columns(pl.Series("raw_score", val_preds))
    
    # Rank and scale to [-1, 1][cite: 2]
    val_df_pred = val_df_pred.with_columns(
        ((pl.col("raw_score").rank("dense").over("eom") - 1) / 
         (pl.col("raw_score").count().over("eom") - 1) * 2 - 1).alias("scaled_rank")
    )
    
    # Fit linear regression on validation years[cite: 2]
    lr = LinearRegression()
    lr.fit(val_df_pred["scaled_rank"].to_numpy().reshape(-1, 1), val_df_pred[target_col].to_numpy())
    
    # 4. Refit on Train + Validation[cite: 2]
    print("Refitting optimal model on combined Train + Validation data...")
    X_train_val = df.filter(train_mask | val_mask).select(features).to_pandas()
    y_train_val = df.filter(train_mask | val_mask)["label_decile"].to_numpy()
    qid_train_val = df.filter(train_mask | val_mask)["eom"].dt.epoch().to_numpy()
    
    final_model = xgb.XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=200,
        reg_lambda=10,
        tree_method="hist",
        device="cuda",
        max_depth=best_params["max_depth"],
        n_estimators=best_trees
    )
    
    final_model.fit(X_train_val, y_train_val, qid=qid_train_val, verbose=False)
    
    # 5. Predict Test Year and Generate final signals
    print("Scoring universe and applying return forecast...")
    X_all = df.select(features).to_pandas()
    df = df.with_columns(pl.Series("score", final_model.predict(X_all)))
    
    df = df.with_columns(
        ((pl.col("score").rank("dense").over("eom") - 1) / 
         (pl.col("score").count().over("eom") - 1) * 2 - 1).alias("scaled_test_rank")
    )
    
    # Apply forecast regression to get ret_hat[cite: 2]
    df = df.with_columns(
        pl.Series("ret_hat", lr.predict(df["scaled_test_rank"].to_numpy().reshape(-1, 1)))
    )
    
    # The optimizer requires the column to be exactly 'score'
    return df, final_model