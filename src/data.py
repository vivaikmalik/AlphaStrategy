"""
src/data.py - Steps 1-3: load inputs, universe filter, per-month rank transform (spec: steps 1-3).
"""
import numpy as np
import polars as pl


def load_inputs(cfg):
    """Step 1: chars panel + target_month, 147 factor names, deduplicated 8-K filings."""
    f = cfg["data_files"]
    factors = pl.read_csv(f["factors"])["variable"].drop_nulls().to_list()
    panel = pl.read_parquet(f["chars"]).with_columns(
        pl.col("eom").cast(pl.Date),
        pl.col("eom").cast(pl.Date).dt.offset_by("1mo").dt.month_end().alias("target_month"))
    assert panel.select(["permno", "eom"]).is_duplicated().sum() == 0, "(permno, eom) not unique"
    cols = ["document_id", "permno", "filing_date", "items", "text", "text_sha256", "company_name", "ticker"]
    filings = (pl.read_parquet(f["8k"], columns=cols).drop_nulls("permno")
               .with_columns(pl.col("filing_date").cast(pl.Date))
               .sort(["filing_date", "document_id"])
               .unique(subset=["permno", "text_sha256"], keep="first", maintain_order=True))
    if cfg["smoke"]:  # seeded permno subsample
        ids = np.sort(panel["permno"].unique().to_numpy())
        keep = np.random.default_rng(cfg["seed"]).choice(ids, min(cfg["smoke_n_permnos"], len(ids)), replace=False)
        panel = panel.filter(pl.col("permno").is_in(keep))
        filings = filings.filter(pl.col("permno").is_in(keep))
    return panel, factors, filings


def short_eligible_mask(df, short_me_q, cfg):
    """Short-eligible expr: me_pct >= short_me_q, dolvol_pct >= short_dolvol_q, zero-trades <= universe median.
    Use as df.with_columns(short_eligible=short_eligible_mask(df, q, cfg)); df is unused (expression-based)."""
    return ((pl.col("me_pct") >= short_me_q) & (pl.col("dolvol_pct") >= cfg["short_dolvol_q"])
            & (pl.col("zt_raw") <= pl.col("zt_raw").median().over("eom"))).fill_null(False)


def build_universe(panel, cfg):
    """Step 2: per month, |prc| >= min and market_equity above the month's quantile (all stocks); raw copies kept."""
    me = pl.col("market_equity")
    df = panel.with_columns(
        prc_raw=pl.col("prc"), me_raw=me, dolvol_raw=pl.col("dolvol_126d"), zt_raw=pl.col("zero_trades_126d"),
        me_pct=me.rank("average").over("eom") / me.count().over("eom"),
        _cut=me.quantile(cfg["universe_me_q"]).over("eom"))
    df = df.filter((pl.col("prc").abs() >= cfg["min_abs_prc"]) & (me > pl.col("_cut"))).drop("_cut")
    return df.with_columns(
        dolvol_pct=pl.col("dolvol_raw").rank("average").over("eom") / pl.col("dolvol_raw").count().over("eom"),
        zt_pct=pl.col("zt_raw").rank("average").over("eom") / pl.col("zt_raw").count().over("eom"),
    ).with_columns(short_eligible=short_eligible_mask(None, cfg["short_me_q"], cfg))


def preprocess(univ, factors, cfg):
    """Step 3: missing flags from raw nulls (train rows), template rank to [-1,1] per month, nulls -> 0, sector."""
    train = univ.filter(pl.col("target_month") <= pl.lit("2018-12-31").str.to_date())
    miss = train.select([pl.col(c).is_null().mean() for c in factors]).row(0)
    flag_vars = [c for c, mm in zip(factors, miss) if mm is not None and mm > cfg["missing_flag_threshold"]]
    flag_cols = [c + "_miss" for c in flag_vars]
    out = univ.with_columns([pl.col(c).is_null().cast(pl.Int8).alias(c + "_miss") for c in flag_vars])

    def rk(c):  # template: dense rank - 1, / month max, *2 - 1; max == 0 -> 0; nulls -> 0
        r = pl.col(c).rank("dense").over("eom") - 1
        mx = r.max().over("eom")
        return pl.when(mx > 0).then(r / mx * 2 - 1).otherwise(0.0).fill_null(0.0).alias(c)
    out = out.with_columns([rk(c) for c in factors]).with_columns(sector=pl.col("gics").str.slice(0, 2).fill_null("NA"))
    return out, flag_cols
