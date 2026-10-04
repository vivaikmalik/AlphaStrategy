"""
src/config.py - Every setting of the pipeline in one CONFIG dict (spec: global conventions).
build_main.py puts this file first, so CONFIG sits at the top of MAIN.py.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # repo root when run as MAIN.py
if ROOT.name == "src":
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"
for _d in [CACHE_DIR, OUTPUT_DIR, OUTPUT_DIR / "figures"]:
    _d.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("ALPHA_SMOKE", "0") == "1"  # tiny CPU run for local testing

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

CONFIG = {
    # --- global ---
    "seed": 42,
    "device": _DEVICE,                      # GX10: cuda (GB10 Blackwell, unified memory)
    "n_jobs": 16,                           # CPU workers for the optimizer grid (GX10: 20 Grace cores)
    "smoke": SMOKE,
    "smoke_n_permnos": 1500,                # random permno subsample in smoke mode
    "data_files": {
        "chars": DATA_DIR / "chars_final_with_names.parquet",
        "8k": DATA_DIR / "8k_20150101_20260831_identified.parquet",
        "factors": DATA_DIR / "factor_char_list.csv",
        "tb3ms": DATA_DIR / "TB3MS.csv",    # columns: eom, tb3ms (annualized %)
        "sp500": DATA_DIR / "SP500.csv",    # columns: eom, sp500_ret, sp500_source
    },
    "cache_dir": CACHE_DIR,
    "output_dir": OUTPUT_DIR,

    # --- schedule (by target month) ---
    "first_train_month": "2015-02",
    "test_years": [2021, 2022, 2023, 2024, 2025, 2026],
    "last_test_month": "2026-08",           # 2026 covers January-August only
    "tune_val_start": "2019-01",            # step 10 settings fixed on 2019-2020 validation
    "tune_val_end": "2020-12",

    # --- step 2 universe (defaults; short-eligible ME pct tuned under step 10) ---
    "min_abs_prc": 5.0,
    "universe_me_q": 0.20,                  # market_equity strictly above this pct (all stocks that month)
    "short_me_q": 0.40,                     # market_equity >= this pct (all stocks that month)
    "short_dolvol_q": 0.30,                 # dolvol_126d >= this pct of the universe
    "short_zt_q": 0.50,                     # zero_trades_126d <= universe median

    # --- step 3 ---
    "missing_flag_threshold": 0.20,

    # --- step 4 GRU ---
    "gru_seq_len": 11,
    "gru_min_valid": 6,
    "gru_hidden": 64,
    "gru_wd_grid": [0.0, 1e-4, 1e-3],
    "gru_train_end_eom": "2018-11-30",
    "gru_val_eom": ("2019-01-31", "2020-12-31"),
    "gru_lr": 1e-3,
    "gru_batch": 2048,
    "gru_max_epochs": 2 if SMOKE else 100,
    "gru_patience": 5,
    "gru_clip": 1.0,

    # --- step 5 XGBoost ranker ---
    "xgb_fixed": dict(objective="rank:pairwise", learning_rate=0.03, subsample=0.7,
                      colsample_bytree=0.5, min_child_weight=200, reg_lambda=10, tree_method="hist",
                      # xgboost>=2 normalises pairwise gradients by default -> hessians so small that
                      # min_child_weight=200 blocks splits (all depths identical). Classic behaviour:
                      lambdarank_normalization=False,
                      # xgboost>=2 defaults to "topk" pairs (only the current top-ranked docs of each month).
                      # Classic pairwise over the whole cross-section, as the spec intends:
                      lambdarank_pair_method="mean"),
    "xgb_depth_grid": [3, 4, 5],
    "xgb_max_trees": 100 if SMOKE else 2000,
    "xgb_eval_every": 50,

    # --- step 6/7 8-K ---
    "items_flag": ["1.01", "1.02", "1.03", "2.01", "2.02", "2.03", "2.05", "2.06",
                   "3.01", "4.01", "4.02", "5.02", "7.01", "8.01"],
    "items_hist": ["4.02", "5.02", "2.06", "1.02", "2.05"],
    "finbert_model": "ProsusAI/finbert",
    "finbert_chunk_tokens": 512,
    "finbert_max_chunks": 8,
    "finbert_batch": 256,                   # chunks per forward pass (bf16 on GX10)
    "finbert_max_filings": 40 if SMOKE else None,

    # --- step 8 Kalman ---
    "kalman_obs": ["beta_dimson_21d", "betadown_252d", "betabab_1260d", "beta_60m"],
    "kalman_fit_end_eom": "2018-12-31",

    # --- step 9 optimizer ---
    "n_long_cand": 250,
    "n_short_cand": 250,
    "gross": 2.0,
    "net_band": 0.20,
    "beta_tol": 0.02,
    "beta_tol_step": 0.01,
    "sector_band": 0.10,
    "max_weight": 0.015,

    # --- additions after the first full run (not in spec; chosen on 2019-2020 only) ---
    "size_band": 0.05,                      # |sum size_z * w| <= band; size_z = monthly z-score of log(me_raw)
    "lambda_risk_grid": [0.0, 100.0],       # L2 penalty lam_risk * sum(w^2): specific-risk proxy
    # factor risk model (not in spec): exposures = intercept + these ranked characteristics + sector dummies;
    # factor returns from monthly cross-sectional OLS of realised returns; covariance over trailing months
    "risk_factors": ["market_equity", "be_me", "ret_12_1", "ret_1_0", "beta_60m", "ivol_capm_252d",
                     "ope_be", "at_gr1", "dolvol_126d"],
    "risk_window": 60,                      # trailing months of factor returns
    "risk_min_months": 24,
    "lambda_fac_grid": [0.0, 300.0, 1000.0, 3000.0, 10000.0],  # lam_fac * monthly factor variance of w
    "kalman_r_floor": 0.05,                 # variant "kf_rfloor": lower bound on every R_j
    "beta_shrink": 0.33,                    # variant "b60_shrunk": (1-0.33)*beta_60m + 0.33*1
    "beta_variants": ["kf", "kf_rfloor", "b60_shrunk"],
    "beta_check_t": 1.96,                   # step 10 beta check: |t| above this = clearly nonzero -> fix betas

    # --- book v2 (not in spec; added after the first full runs). Every number is a stated assumption or
    #     estimated on pre-test data, nothing is searched. The spec book is still run and reported for comparison.
    "book": "v2",                           # "v2" = main book below; "spec" = step-9 optimizer as the main book
    "v2_cost": 0.0012,                      # assumed one-way trading cost (12 bps) per unit of |trade|
    "v2_vol_target": 0.08,                  # ex-ante annual volatility cap (factor + specific risk)
    "v2_max_weight": 0.005,                 # 0.5% per name -> ~400-500 names at 200% gross (rules allow 500)
    "v2_beta_tol": 0.02,                    # |factor-model beta . w|
    "v2_ic_floor": 0.01,                    # IC used to scale alpha = IC * xs_vol * z (each window's validation IC)
    "risk_spec_window": 36,                 # trailing months of residuals for specific variance
    "risk_spec_min": 6,

    # --- step 10 grids (fixed once on 2019-2020, then frozen) ---
    "gru_d_grid": [8, 16, 32],
    "lambda_tc_grid": [0.0, 0.1, 0.25, 0.5],
    "lambda_beta_grid": [0.0, 10.0, 100.0, 1000.0],
    "short_me_q_grid": [0.30, 0.40, 0.50],

    # --- step 12 template Ridge baseline ---
    "ridge_alpha_grid": [1e-1, 1, 10, 100, 1000, 10000],

    # --- metrics ---
    "premium_annual": 0.04,
    "nw_lags": 3,
}
