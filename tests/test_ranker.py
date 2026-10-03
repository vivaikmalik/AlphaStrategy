import polars as pl
from src.config import CACHE_DIR, DATA_DIR
from src.gru import train_and_embed
from src.ranker import train_and_rank
import torch

def test_ranker():
    # 1. Load Data
    cache_path = CACHE_DIR / "preprocessed_features.parquet"
    df = pl.read_parquet(cache_path)
    factors = pl.read_csv(DATA_DIR / "factor_char_list.csv")["variable"].drop_nulls().to_list()
    
    # 2. Get GRU Embeddings (using the tuned pipeline)
    print("Generating GRU embeddings...")
    torch.backends.cudnn.enabled = False 
    embed_df = train_and_embed(d=8, df=df, factors=factors, device="cuda")
    
    # Join embeddings to main dataframe
    df_merged = df.join(embed_df, on=["permno", "eom"], how="left")
    
    # 3. Train XGBoost and Rank
    # UPDATE "target_return" to match your exact target column name!
    ranked_df, _ = train_and_rank(df_merged, factors, d=8, target_col="ret_exc_lead1m")
    
    # 4. Verify the Alpha Signal Math
    max_signal = ranked_df["alpha_signal"].max()
    min_signal = ranked_df["alpha_signal"].min()
    
    print(f"Signal Bounds Check -> Min: {min_signal:.2f}, Max: {max_signal:.2f}")
    assert max_signal == 1.0, "Signal not bounded to 1"
    assert min_signal == -1.0, "Signal not bounded to -1"
    
    print("✅ XGBoost Ranker and Cross-Sectional Alpha Signal Verified.")

if __name__ == "__main__":
    test_ranker()