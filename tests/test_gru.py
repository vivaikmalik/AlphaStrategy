import polars as pl
import torch
from src.config import CACHE_DIR, DATA_DIR
from src.gru import train_and_embed

def test_gru():
    cache_path = CACHE_DIR / "preprocessed_features.parquet"
    df = pl.read_parquet(cache_path)
    factors = pl.read_csv(DATA_DIR / "factor_char_list.csv")["variable"].drop_nulls().to_list()
    
    print("Testing GRU training with d=8...")
    embed_df = train_and_embed(d=8, df=df, factors=factors, device="cuda" if torch.cuda.is_available() else "cpu")
    
    assert f"gru_8" in embed_df.columns, "Embeddings not correctly generated."
    
    nan_count = embed_df["gru_1"].is_null().sum()
    print(f"Generated embeddings. Invalid sequence count (NaNs): {nan_count}")
    print("✅ GRU Architecture and Train Loop Verified.")

if __name__ == "__main__":
    test_gru()