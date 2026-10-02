import polars as pl
import numpy as np
from src.config import DATA_DIR, CACHE_DIR

def test_pipeline_integrity():
    # 1. Load the generated cache and the raw factor list
    cache_path = CACHE_DIR / "preprocessed_features.parquet"
    assert cache_path.exists(), "Cache file not found. Did src/data.py run successfully?"
    
    df = pl.read_parquet(cache_path)
    factors = pl.read_csv(DATA_DIR / "factor_char_list.csv")["variable"].drop_nulls().to_list()
    
    # 2. Spec Step 11: Assert (permno, eom) is strictly unique
    duplicates = df.select(["permno", "eom"]).is_duplicated().sum()
    assert duplicates == 0, f"Leakage Rule Failed: Found {duplicates} duplicate (permno, eom) pairs."
    
    # 3. Spec Step 2: Universe validation
    assert df["abs_prc"].min() >= 5.0, "Universe Rule Failed: Stocks with price < 5 slipped through."
    
    # 4. Spec Step 3: Rank-transformation bounds
    # Every factor must be scaled strictly between -1.0 and 1.0, with exactly 0 missing values
    for var in factors:
        var_min = df[var].min()
        var_max = df[var].max()
        null_count = df[var].is_null().sum()
        
        assert null_count == 0, f"Preprocessing Failed: Column {var} contains {null_count} NaNs."
        assert var_min >= -1.0001 and var_max <= 1.0001, \
            f"Scaling Failed: Column {var} is out of bounds [-1, 1]. Min: {var_min}, Max: {var_max}"
            
    # 5. Spec Step 11: Assert no feature is identical to the target
    # The target is ret_exc_lead1m. We must ensure no factor perfectly correlates with it.
    if "ret_exc_lead1m" in df.columns:
        target_series = df["ret_exc_lead1m"]
        for var in factors:
            # Check for exact equality
            is_identical = (df[var] == target_series).sum() == len(df)
            assert not is_identical, f"Leakage Rule Failed: Feature {var} is identical to the target!"
            
    # 6. Sector verification (first 2 digits of gics)
    assert df["sector"].str.len_bytes().max() == 2, "Preprocessing Failed: Sector string length exceeds 2 digits."

    print("✅ All Step 1-3 data integrity and leakage tests passed.")

if __name__ == "__main__":
    test_pipeline_integrity()