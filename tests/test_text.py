import polars as pl
from src.text import process_text_features

def test_text():
    print("Testing FinBERT and 8-K parsing (Downloading model if first run)...")
    df_8k = pl.DataFrame({
        "permno": [10001, 10001],
        "eom": [pl.date(2020, 1, 31), pl.date(2020, 2, 29)],
        "document_id": ["d1", "d2"],
        "text_sha256": ["hash1", "hash2"],
        "items": ["Item 1.01, Item 4.02", "Item 8.01"],
        "text": ["Signed a material agreement but financials cannot be relied upon.", "Routine update."],
        "company_name": ["Apple", "Apple"],
        "ticker": ["AAPL", "AAPL"]
    })
    
    # Run on CPU just for the unit test to avoid locking GPU memory
    df_out = process_text_features(df_8k, device="cpu")
    
    assert "item_4_02" in df_out.columns, "Failed to parse item codes"
    assert "tone_mean" in df_out.columns, "Failed to extract FinBERT tone"
    assert "tone_surprise" in df_out.columns, "Failed to calculate tone surprise"
    print("✅ Text Processing (Steps 6 & 7) Verified.")

if __name__ == "__main__":
    test_text()