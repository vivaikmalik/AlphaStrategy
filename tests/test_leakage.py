import polars as pl

def test_leakage(df: pl.DataFrame):
    """Step 11 assertions[cite: 3]."""
    assert df.filter(pl.col("filing_date") > pl.col("eom")).height == 0, "Future filing leakage"
    assert df.is_unique(subset=["permno", "eom"]), "Duplicate permno-eom keys"
    assert "ret_exc_lead1m" not in df.columns[:147], "Target variable in features"
    print("✅ Leakage tests passed.")