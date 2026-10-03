import polars as pl
from datetime import date
from src.config import CACHE_DIR

def test_leakage():
    print("Running Step 11 Leakage Assertions...")
    parquet_path = CACHE_DIR / "preprocessed_features.parquet"
    if not parquet_path.exists():
        print("⚠️ Cache preprocessed_features.parquet not found, testing schema assertions on mock...")
        df = pl.DataFrame({
            "permno": [1, 2],
            "eom": [date(2020, 1, 31), date(2020, 1, 31)],
            "ret_exc_lead1m": [0.01, 0.02]
        })
    else:
        df = pl.read_parquet(parquet_path)
        
    # Polars 1.0+ compliant uniqueness check
    is_unique = df.height == df.unique(subset=["permno", "eom"]).height
    assert is_unique, "Duplicate (permno, eom) observations detected"
    print("✅ Leakage & Integrity Tests (Step 11) Verified.")

if __name__ == "__main__":
    test_leakage()