import polars as pl
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def process_text_features(df_8k: pl.DataFrame, device: str = "cuda") -> pl.DataFrame:
    """Extracts 8-K event flags and FinBERT tone per stock-month."""
    items_to_flag = ["1.01", "1.02", "1.03", "2.01", "2.02", "2.03", "2.05", "2.06", 
                     "3.01", "4.01", "4.02", "5.02", "7.01", "8.01"]
    history_items = ["4.02", "5.02", "2.06", "1.02", "2.05"]
    
    # FIX: Generate the 'eom' column from 'filing_date'
    df_8k = df_8k.with_columns(
        pl.col("filing_date").cast(pl.Date).dt.month_end().alias("eom")
    )
    
    # 1. Deduplicate and assign to filing month
    df = df_8k.unique(subset=["permno", "text_sha256"]).drop_nulls("text")
    
    # Unpacked with_columns and over (NO LIST BRACKETS)
    df = df.with_columns(
        pl.lit(1).alias("has_filing"),
        pl.col("document_id").len().over("permno", "eom").alias("n_filings")
    )
    
    for item in items_to_flag:
        df = df.with_columns(
            pl.col("items").str.contains(item).cast(pl.Int8).fill_null(0).alias(f"item_{item.replace('.', '_')}")
        )
        
    # 2. Anonymize text
    def anonymize(text, comp, tick):
        if not isinstance(text, str): return ""
        if isinstance(comp, str) and comp: text = re.sub(re.escape(comp), "[COMPANY]", text, flags=re.IGNORECASE)
        if isinstance(tick, str) and tick: text = re.sub(rf"\b{re.escape(tick)}\b", "[TICKER]", text, flags=re.IGNORECASE)
        return text
        
    df = df.with_columns(
        pl.struct(["text", "company_name", "ticker"]).map_elements(
            lambda x: anonymize(x["text"], x["company_name"], x["ticker"]), return_dtype=pl.Utf8
        ).alias("clean_text")
    )
    
    # 3. FinBERT inference
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()
    
    texts, doc_ids = df["clean_text"].to_list(), df["document_id"].to_list()
    tone_means, tone_mins = [], []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512, return_overflowing_tokens=False)
            outputs = model(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            tones = (probs[:, 0] - probs[:, 1]).cpu().numpy() # P(pos) - P(neg)
            tone_means.extend(tones)
            tone_mins.extend(tones)
            
    df_results = pl.DataFrame({"document_id": doc_ids, "tone_mean": tone_means, "tone_min": tone_mins})
    df = df.join(df_results, on="document_id")
    
    # 4. Aggregation and Trailing History
    agg_exprs = [pl.col("has_filing").first(), pl.col("n_filings").first(), pl.col("tone_mean").mean(), pl.col("tone_min").min()]
    for item in items_to_flag:
        agg_exprs.append(pl.col(f"item_{item.replace('.', '_')}").max())
        
    # Unpacked agg using *agg_exprs
    df_monthly = df.group_by("permno", "eom").agg(*agg_exprs).sort("permno", "eom")
    
    for item in history_items:
        df_monthly = df_monthly.with_columns(
            pl.col(f"item_{item.replace('.', '_')}").rolling_sum(window_size=12, min_periods=1).over("permno").alias(f"item_{item.replace('.', '_')}_12m")
        )
        
    # Unpacked with_columns chaining
    df_monthly = df_monthly.with_columns(
        pl.col("tone_mean").rolling_mean(window_size=12, min_periods=1).over("permno").alias("tone_12m_avg")
    ).with_columns(
        (pl.col("tone_mean") - pl.col("tone_12m_avg")).alias("tone_surprise")
    ).drop("tone_12m_avg")
    
    return df_monthly