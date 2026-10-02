"""
src/data.py - Steps 1, 2, & 3: High-throughput Ingestion, Universe Filter, & Vectorized Preprocessing
Optimized for ARM64 Grace Neoverse V2 architecture using multi-threaded Polars expressions.
"""

from pathlib import Path
from typing import List, Tuple
import polars as pl
import numpy as np

from src.config import (
    DATA_DIR,
    CACHE_DIR,
    UNIVERSE_PRC_MIN,
    UNIVERSE_ME_PCT,
    SHORT_ME_PCT,
    SHORT_DOLVOL_PCT,
    SHORT_ZEROTRADES_PCT,
    MISSING_FLAG_THRESHOLD,
)

def load_raw_inputs() -> Tuple[pl.DataFrame, List[str], pl.DataFrame]:
    """
    Step 1: Read parquet inputs, extract 147 factors, deduplicate 8-Ks on (permno, text_sha256),
    and construct target_month = eom + 1 month.
    """
    # 1. Load 147 factor variable names
    factor_path = DATA_DIR / "factor_char_list.csv"
    factors: List[str] = (
        pl.read_csv(factor_path)["variable"]
        .drop_nulls()
        .to_list()
    )

    # 2. Ingest numerical panel and compute target_month (eom + 1 month)
    chars_path = DATA_DIR / "chars_final_with_names.parquet"
    df = (
        pl.read_parquet(chars_path)
        .with_columns([
            pl.col("eom").cast(pl.Date),
            # Precise month-end shift: offset 1 month, anchor to month_end
            pl.col("eom").cast(pl.Date).dt.offset_by("1mo").dt.month_end().alias("target_month"),
            pl.col("prc").abs().alias("abs_prc")
        ])
    )

    # Step 11 check: Assert (permno, eom) is strictly unique
    assert df.select(["permno", "eom"]).is_duplicated().sum() == 0, \
        "Data Integrity Violation: (permno, eom) is not unique in panel."

    # 3. Ingest and deduplicate 8-K filings on (permno, text_sha256)
    eightk_path = DATA_DIR / "8k_20150101_20260831_identified.parquet"
    keep_cols = ["document_id", "permno", "filing_date", "items", "text", "text_sha256"]
    
    text_df = (
        pl.read_parquet(eightk_path, columns=keep_cols)
        .with_columns(pl.col("filing_date").cast(pl.Date))
        .unique(subset=["permno", "text_sha256"])
    )

    return df, factors, text_df


def filter_universe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Step 2: Define universe per eom using month-t data only.
    - Base universe: abs(prc) >= 5 and market_equity > 20th percentile of month t.
    - Short-eligible: market_equity >= 40th pct, dolvol_126d >= 30th pct,
      and zero_trades_126d <= median of the base universe.
    """
    # Base universe conditions evaluated cross-sectionally per eom
    df = df.with_columns([
        pl.col("market_equity").quantile(UNIVERSE_ME_PCT).over("eom").alias("me_p20")
    ]).with_columns([
        ((pl.col("abs_prc") >= UNIVERSE_PRC_MIN) & (pl.col("market_equity") > pl.col("me_p20"))).alias("in_universe")
    ])

    # Filter to base universe stocks only
    univ_df = df.filter(pl.col("in_universe"))

    # Compute short-eligibility thresholds strictly against the base universe cross-section
    univ_df = univ_df.with_columns([
        pl.col("market_equity").quantile(SHORT_ME_PCT).over("eom").alias("short_me_thresh"),
        pl.col("dolvol_126d").quantile(SHORT_DOLVOL_PCT).over("eom").alias("short_dolvol_thresh"),
        pl.col("zero_trades_126d").quantile(SHORT_ZEROTRADES_PCT).over("eom").alias("short_zero_thresh"),
    ]).with_columns([
        (
            (pl.col("market_equity") >= pl.col("short_me_thresh")) &
            (pl.col("dolvol_126d") >= pl.col("short_dolvol_thresh")) &
            (pl.col("zero_trades_126d") <= pl.col("short_zero_thresh"))
        ).alias("short_eligible")
    ]).drop(["me_p20", "short_me_thresh", "short_dolvol_thresh", "short_zero_thresh"])

    return univ_df


def preprocess_features(df: pl.DataFrame, factors: List[str]) -> Tuple[pl.DataFrame, List[str]]:
    """
    Step 3: Multi-threaded rank transformation, missingness imputation, and flag creation.
    - sector = first 2 digits of gics.
    - Missing flags created for features with >20% missingness in training period (eom <= 2018-12-31).
    - Features scaled to [-1, 1] per month via (dense_rank - 1) / max * 2 - 1, filled with 0 (median).
    """
    # 1. Sector extraction: first 2 digits of gics
    df = df.with_columns(
        pl.col("gics").cast(pl.Utf8).str.zfill(6).str.slice(0, 2).alias("sector")
    )

    # 2. Identify sparse features (>20% missing in training data eom <= 2018-12-31)
    train_split = df.filter(pl.col("eom") <= pl.date(2018, 12, 31))
    
    missing_flags: List[str] = []
    missing_exprs = []
    for var in factors:
        null_ratio = train_split.select(pl.col(var).is_null().mean()).item()
        if null_ratio > MISSING_FLAG_THRESHOLD:
            flag_name = f"{var}_missing"
            missing_flags.append(flag_name)
            missing_exprs.append(
                pl.col(var).is_null().cast(pl.Float32).alias(flag_name)
            )

    if missing_exprs:
        df = df.with_columns(missing_exprs)

    # 3. Vectorized rank-transformation and median-fill across all Grace cores simultaneously
    rank_exprs = []
    for var in factors:
        # Cross-sectional median per month
        med = pl.col(var).median().over("eom")
        imputed = pl.col(var).fill_null(med)
        
        # Dense rank (1-indexed) adjusted to 0-indexed
        rank_0 = (imputed.rank(method="dense").over("eom") - 1).cast(pl.Float64)
        rank_max = rank_0.max().over("eom")

        # Scale to [-1, 1], default to 0.0 if all missing/constant
        scaled = (
            pl.when(rank_max > 0)
            .then((rank_0 / rank_max) * 2.0 - 1.0)
            .otherwise(0.0)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(var)
        )
        rank_exprs.append(scaled)

    # Parallel batch execution across all 147 feature expressions
    df = df.with_columns(rank_exprs)

    return df, missing_flags


def run_data_pipeline() -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    Executes Steps 1-3 end-to-end and writes intermediate states to cache.
    """
    raw_df, factors, text_df = load_raw_inputs()
    
    # Cache deduplicated 8-K records
    text_cache = CACHE_DIR / "8k_deduped.parquet"
    text_df.write_parquet(text_cache)

    # Universe filtering
    univ_df = filter_universe(raw_df)

    # Preprocessing
    processed_df, missing_flags = preprocess_features(univ_df, factors)

    # Cache preprocessed features
    proc_cache = CACHE_DIR / "preprocessed_features.parquet"
    processed_df.write_parquet(proc_cache)

    return processed_df, factors, missing_flags