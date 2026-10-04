# What `python MAIN.py` runs

This document describes the code as it is, not the plan. Sources: `src/*.py` (MAIN.py is generated from them by `build_main.py`, so MAIN.py and `src/` are the same code) and `Implementation_Spec_GRU_XGBoost.pdf`. The order of execution is `main()` in `src/pipeline.py`. Spec steps 1-14 ("First run") are implemented. The optional spec steps A-D ("After the first run") are not.

## 1. Overview

The strategy is a market-neutral long/short US equity portfolio, rebuilt monthly, holding 100-500 names. The benchmark is the 3-month T-bill plus 4% a year, and the main score is the information ratio (IR) of the active return (excess return minus 4%/12 per month). Each month, a model ranks stocks by expected next-month excess return, and a convex optimiser turns the ranking into weights that satisfy the competition rules (200% gross, net within +/-20%, beta near zero, sector-neutral, 1.5% cap per name).

Data flow in one line:

`147 characteristics + 8-K filings -> universe + per-month rank transform -> (Kalman betas, 8-K flags, FinBERT tone, GRU embeddings) -> XGBoost pairwise ranker (walk-forward) -> score -> CVXPY optimiser -> weights -> returns, performance pack, holdings.csv/returns.csv`

Conventions that hold everywhere:

- A row at `eom = t` uses month-t data only. The target `ret_exc_lead1m` is the excess return of month t+1. `target_month` = `eom` + 1 month (month-end).
- The holdings date is the first day of month t+1 (`Date` in `holdings.csv` and `returns.csv`).
- Weights are signed fractions of $1. Excess = sum of w * `ret_exc_lead1m`; total = TB3MS/1200 of month t+1 + excess; active = excess - 0.04/12.
- Seed 42. Every setting lives in the `CONFIG` dict (`src/config.py`, top of MAIN.py).

## 2. Execution order as it really runs

`main()` runs these blocks in this order. The spec step number is in brackets. Note the order differs from the numbering: Kalman (8) and text (6, 7) run before the GRU (4), and tuning (10) runs before the main schedule (5).

| # | Block | Spec step |
|---|---|---|
| 1 | Load, universe, preprocess | 1, 2, 3 |
| 2 | Kalman betas | 8 |
| 3 | 8-K event flags + FinBERT tone | 6, 7 |
| 4 | GRU embeddings for d = 8, 16, 32 | 4 |
| 5 | Step-10 tuning on 2019-2020 + beta check | 10 (uses 5, 9) |
| 6 | Full walk-forward XGBoost, book, performance, submission | 5, 9, 13, 14 |
| 7 | Ablations | 12 |
| 8 | Leakage tests incl. shuffled-label run | 11 |

### 2.1 Load (step 1) - `data.load_inputs`

- Inputs: `data/chars_final_with_names.parquet`, `data/8k_20150101_20260831_identified.parquet`, `data/factor_char_list.csv` (column `variable` = the 147 feature names), `data/TB3MS.csv`, `data/SP500.csv`.
- Adds `target_month`. Asserts `(permno, eom)` is unique.
- 8-K filings: keeps `document_id, permno, filing_date, items, text, text_sha256, company_name, ticker`; drops rows with null permno; deduplicates on `(permno, text_sha256)`, keeping the earliest filing.
- Why: the spec fixes the alignment (month-t features, t+1 target) once, here, so nothing downstream shifts the target again. Deduplication stops the same text from being counted twice.
- Smoke mode (`ALPHA_SMOKE=1`) keeps a seeded random 1500 permnos.

### 2.2 Universe and preprocessing (steps 2, 3) - `data.build_universe`, `data.preprocess`

- Output: `cache/universe.parquet`.
- Universe, per `eom`: `abs(prc) >= 5` and `market_equity` strictly above that month's 20th percentile (`universe_me_q`, computed over all stocks that month). Raw copies (`prc_raw, me_raw, dolvol_raw, zt_raw`) and percentiles are kept.
- Short-eligible flag: `me_pct >= short_me_q` (default 0.40, ranked among all stocks that month), `dolvol_126d` percentile within the universe `>= 0.30`, and `zero_trades_126d <=` the universe median that month. Nulls become not eligible.
- Missing flags: for each of the 147 features, compute the null share on universe rows with `target_month <= 2018-12-31`. If it is above 20%, add an Int8 column `<feature>_miss` (1 = missing). The cut-off date is hard-coded in `preprocess`, not in CONFIG.
- Rank transform: each month, each feature is dense-ranked among the universe, rescaled to [-1, 1] (`rank-1` over the month's max, times 2, minus 1; a constant column becomes 0), then nulls become 0 (the median). `sector` = first two digits of `gics`, "NA" if missing.
- Why: the rank transform makes features comparable across features and months and robust to the huge outliers in raw characteristics (the raw betas, for instance, reach +/-1e4). It also matches the template's method. The missing flags keep the information "this value was imputed to 0", which would otherwise be lost. The universe filters remove illiquid and tiny names (the cheap-to-hold and tradable set). The short-eligible subset is stricter because shorting small, thin names is costly and hard to borrow.

### 2.3 Kalman betas (step 8) - `kalman.kalman_betas`

- Input: the raw panel, all stocks, unranked. Output: `cache/kalman.parquet` (+ `kalman.json` with fitted parameters), columns `beta_kf`, `beta_var`, joined onto the universe.
- Observations per stock-month: `beta_dimson_21d`, `betadown_252d`, `betabab_1260d`, `beta_60m`. State: the true beta as an AR(1) deviation around a prior mean `m_t`, `beta_t - m_t = phi (beta_{t-1} - m_{t-1}) + noise(q)`. The prior `m_t` is the sector median of `beta_60m` in month t (all-stock median if the sector value is missing). Each available observation updates the state in sequence, with its own noise variance `R_j`.
- `phi`, `q` and the four `R_j` are fitted by exact Gaussian maximum likelihood on `eom <= 2018-12-31`, pooled across stocks, then frozen. The filter then runs forward over all months, so each month's value uses only past and current data.
- Fallbacks: a stock-month with no observation yet gets `beta_60m`, or `m_t` if that is also missing.
- Why: the four raw beta estimates are noisy and disagree. The optimiser enforces `|beta_kf . w| <= 0.02`, so it needs one stable beta per stock, and `beta_var` supplies a variance penalty on uncertain betas.

### 2.4 8-K event flags and FinBERT tone (steps 6, 7) - `text.py`

- Output: `cache/text_features.parquet` (event flags + tone), and `cache/finbert_docs.parquet` (per-filing tones, saved every 2000 filings, keyed by `document_id`).
- Details are in section 6. Why: 8-Ks are the only information in the data that is not a price/accounting characteristic and arrives between earnings dates.

### 2.5 GRU embeddings (step 4) - `gru.gru_embeddings`

- Output: `cache/gru_d8.parquet`, `gru_d16.parquet`, `gru_d32.parquet` (+ `.json` with the chosen weight decay and losses). Columns `gru_1..gru_d`.
- Input per `(permno, eom)`: the 11 month-to-month changes of the 147 ranked features over t-11..t, shape (11, 147). A change counts as valid only between calendar-consecutive months; windows with fewer than 6 valid changes get a NaN embedding.
- Model: 1-layer GRU encoder (hidden 64) -> linear layer to d; GRU decoder rebuilds the sequence from the embedding. Masked MSE loss (invalid steps do not count). AdamW, lr 1e-3, batch 2048, up to 100 epochs, early stopping with patience 5, gradient-norm clip 1.0.
- Training sequences: `eom <= 2018-11-30` only. Early stopping and the weight decay choice (from {0, 1e-4, 1e-3}) use the 2019-01 to 2020-12 reconstruction loss. The model is then frozen and used to embed every `(permno, eom)`.
- This is unsupervised: no returns enter, so there is no label leakage, and one frozen model serves all test years.
- Why: the 147 ranked features are a snapshot. The GRU summarises how a stock's characteristics have been changing over the last year (feature dynamics), compressed to d numbers. The ranker can use these without having to learn a 11 x 147 input.

### 2.6 Step-10 tuning on 2019-2020 (step 10) - `pipeline._pipe_tune`

Detailed in section 4. In short: for each d, fit the XGBoost ranker for the 2021 window (train through 2018-12, validate on 2019-2020), build the validation book for every combination of `lambda_tc`, `lambda_beta`, short-eligible ME percentile, `lambda_risk`, `lambda_fac`, and pick the combination with the highest 2019-2020 IR. Then `_pipe_beta_check` regresses the chosen validation book on the S&P 500; if `|t| > beta_check_t` (1.96) the beta is clearly nonzero and the beta variants are tried (section 4). The final choice is frozen for all test years. It writes `output/settings_log.json` (choice, beta check, beta fix, full grid, GRU logs).

### 2.7 Main walk-forward run (steps 5, 9, 13, 14)

- Feature set `full` for the chosen d = 147 ranked factors + `_miss` flags + `gru_1..gru_d` + event columns (21: `has_filing`, `n_filings`, 14 `item_*`, 5 `hist_*`) + 3 tone columns.
- `ranker.run_schedule` fits one XGBoost ranker per test year (section 3). Per window:
  - Model: `XGBRanker`, `objective="rank:pairwise"`, `qid = eom`, `learning_rate 0.03`, `subsample 0.7`, `colsample_bytree 0.5`, `min_child_weight 200`, `reg_lambda 10`, `tree_method="hist"`, `lambdarank_normalization=False`, seed 42.
  - Label: within-month decile (0-9) of `ret_exc_lead1m`, `floor(10 * (ordinal_rank - 1) / n)`.
  - Tuning: fit on the training months with `max_depth` in {3, 4, 5} and up to 2000 trees; every 50 trees, compute the mean monthly rank IC on the validation months; keep the best (depth, trees).
  - Refit on train + validation with that (depth, trees) and score the test year.
  - `ret_hat`: a linear regression of realised `ret_exc_lead1m` on the within-month score rank (scaled to [-1, 1]), fitted on the validation scores of the train-only model, applied to test-year score ranks. It exists only for the OOS R-squared.
  - Missing feature values (GRU/tone/event NaN) go to XGBoost as NaN.
- Output: `output/predictions.parquet` (`permno, eom, score, ret_hat`).
- Why a pairwise ranker: the portfolio is cross-sectional and market-neutral, so only the order of stocks within a month matters, not the level of predicted returns. A pairwise ranking loss on decile labels optimises order directly and is less sensitive to return outliers than a squared-error regression.
- Book (`optimizer.run_book`): one convex problem per month, solved with CVXPY/Clarabel:
  - Candidates: the top 250 stocks by score in the universe (longs) and the bottom 250 by score among short-eligible stocks (shorts). Scores are standardised within the month.
  - Variables `wL >= 0` (long candidates), `wS >= 0` (short candidates); `w = wL - wS`.
  - Objective: maximise `s.w - lambda_tc * |w - w_prev|_1 - lambda_beta * sum(beta_var_i * w_i^2) - lambda_risk * sum(w_i^2)`. Names that left the candidate set count as sold to 0 (a constant in the objective).
  - Constraints: `sum(wL) + sum(wS) = 2`; `|sum(w)| <= 0.20`; `|beta_kf . w| <= tol` with `tol = 0.02`; every sector's net exposure within +/-0.10; every weight `<= 0.015`; size neutrality `|sum(size_z_i * w_i)| <= size_band` (0.05; `size_z` = within-month z-score of log market cap, null -> 0). If the problem is infeasible, `tol` rises by 0.01 until it solves (logged per month; counted in the tuning grid as `n_relaxed_months`).
  - With the 0.015 cap and net within +/-0.20, each side holds at least 0.9 of capital, so at least 60 names per side (120 total), and at most 500.
  - The first month of every book starts from cash (`w_prev = 0`); the book is built separately for validation, test, ablations and the shuffled run.
  - Why these constraints: they are the competition rules plus the spec's neutrality targets. Sector and beta constraints are what make the book market-neutral rather than a disguised sector or market bet. The `lambda_tc` term lowers turnover. The `lambda_beta` term pushes weight away from stocks whose beta is poorly measured.
- Returns (`metrics.book_returns`): weight at month-end t times `ret_exc_lead1m`. If a held stock's `ret_exc_lead1m` is missing, 0 is used and the count is reported (`n_missing_returns`).
- `report.performance_pack` writes `performance.json` and the figures; `report.write_submission` writes `holdings.csv` and `returns.csv`.

### 2.7b Book v2 (alternative book, not the book of record) - `optimizer.run_book_v2`, `pipeline._pipe_book`

Outcome: the main book is chosen by 2019-2020 validation IR, a pre-test quantity: spec book 0.76 vs v2 0.38, so `cfg["book"] = "spec"`. (For the record, on test v2 also did worse: IR -0.20 vs 0.16; it de-levered to ~127% average gross and spread into weaker-scored names.) With `cfg["book"] = "v2"` the book of record (performance pack, `holdings.csv`, `returns.csv`, ablations, shuffled run) is v2. The spec book above still runs on the same test predictions and is reported next to it (`settings_log.json`: `spec_book`; the v2 book on 2019-2020 validation: `v2_validation`; one log line prints both).

- Why: the first full runs of the spec book held a concentrated tail book (about 160 names), carried an unhedged defensive/junk style tilt (S&P beta -0.18), had about 15% volatility and about 220% turnover.
- What: (1) alpha in return units = `IC * xs_vol * z`, with `z` the within-month standardised score, `xs_vol` the month's cross-sectional return volatility from the risk model and `IC` the validation IC of that test year's window (`info["val_ic"]`, known before the test year; validation books use the 2021 window's); (2) explicit one-way trading cost `v2_cost` = 12 bps; (3) ex-ante volatility capped at `v2_vol_target` = 8% using factor risk plus specific risk (`spec_var`, estimated over `risk_spec_window` months, at least `risk_spec_min`; null -> month median); (4) beta hedge on the factor-model beta (`b` from the risk model, tolerance `v2_beta_tol`); (5) weight cap `v2_max_weight` = 0.5%, giving roughly 400-500 names. Same candidates and rules (long/short candidate sets, short eligibility, sector and size bands, gross 2, net within +/-0.20). Months that need relaxed constraints are counted in the log.
- How the numbers were set: cost, vol target and cap are stated assumptions (12 bps is a typical one-way cost for liquid US stocks; 8% is a modest risk budget; 0.5% gives breadth); the IC floor and spec-risk window come from pre-2021 estimates. Nothing was searched or tuned on validation or test.

### 2.8 Ablations and leakage tests (steps 12, 11)

After the submission files exist, `main()` runs the five ablations (section 5), the shuffled-label run, and the leakage checks (section 4). The `assert` statements are the last thing that runs, after the output files have been written. A failing assert stops the script, but the submission files from step 14 are already on disk.

## 3. Walk-forward schedule

Windows are defined by target month (`ranker.windows`). A row's `eom` is one month before its target month; the book formed at `eom` is held during the target month.

| Test year | Train (target months) | Validation (target months) | Test (target months) |
|---|---|---|---|
| 2021 | 2015-02 to 2018-12 | 2019-01 to 2020-12 | 2021-01 to 2021-12 |
| 2022 | 2015-02 to 2019-12 | 2020-01 to 2021-12 | 2022-01 to 2022-12 |
| 2023 | 2015-02 to 2020-12 | 2021-01 to 2022-12 | 2023-01 to 2023-12 |
| 2024 | 2015-02 to 2021-12 | 2022-01 to 2023-12 | 2024-01 to 2024-12 |
| 2025 | 2015-02 to 2022-12 | 2023-01 to 2024-12 | 2025-01 to 2025-12 |
| 2026 | 2015-02 to 2023-12 | 2024-01 to 2025-12 | 2026-01 to 2026-08 |

Three dates per row, using the first 2021 test month as the example:

- `eom` = 2020-12-31: the month whose characteristics are the features.
- `target_month` = 2021-01-31: the month of `ret_exc_lead1m`, which the row predicts.
- Holdings date = 2021-01-01: when the position is held (first day of `target_month`).

Training and validation rows need a non-missing target; test rows do not. The test period is therefore return months 01/2021 to 08/2026, 68 months.

## 4. What is tuned and what is tested

Nothing global is chosen with 2021-2026 results. The global settings (d, `lambda_tc`, `lambda_beta`, short-eligible ME percentile, `lambda_risk`, `lambda_fac`, beta variant) are chosen on the 2019-2020 validation period. The only items that use data after 2020 are the per-window XGBoost depth/trees and ridge alpha, and each of those is chosen on validation months that lie strictly before that test year.

| Search | Where | Grid | Chosen by | Frozen? |
|---|---|---|---|---|
| GRU weight decay, per d | `gru.py` | {0, 1e-4, 1e-3} | 2019-2020 reconstruction loss | Yes, once; model trained on `eom <= 2018-11` |
| GRU epochs | `gru.py` | up to 100, patience 5 | 2019-2020 reconstruction loss | Yes |
| Kalman `phi, q, R_j` | `kalman.py` | continuous (L-BFGS-B MLE) | likelihood on `eom <= 2018-12` | Yes |
| XGB depth x trees, per window | `ranker.py` | depth {3,4,5} x trees {50, 100, ..., 2000} (3 x 40 = 120 points per window) | mean monthly rank IC on that window's validation months | Per window, then refit on train + val |
| d x `lambda_tc` x `lambda_beta` x short-eligible ME percentile x `lambda_risk` x `lambda_fac` | `pipeline._pipe_tune` | d {8,16,32} x `lambda_tc` {0, 0.1, 0.25, 0.5} x `lambda_beta` {0, 10, 100, 1000} x ME percentile {0.30, 0.40, 0.50} x `lambda_risk` {0, 100} x `lambda_fac` {0, 300, 1000, 3000, 10000} = 1440 combinations (rerun once if the beta variant changes) | 2019-2020 validation IR of the book | Yes, for all test years and all ablations |
| Ridge alpha (ablation 1 only), per window | `ranker.py` | {0.1, 1, 10, 100, 1000, 10000} | validation mean monthly rank IC | Per window |

Factor risk model (`src/risk.py`, built once per run from the modelling frame, no look-ahead): exposures X are an intercept, the 9 ranked characteristics in `risk_factors` and sector dummies. Factor returns come from a monthly cross-sectional OLS of realised next-month returns on X. The factor covariance F is the trailing 60-month (`risk_window`) covariance, at least 24 months (`risk_min_months`), using only returns realised by the formation date. The optimiser penalises `lambda_fac * w'XFX'w`, with `lambda_fac` chosen on 2019-2020 together with the other grid dimensions.

Notes on the step-10 grid:

- The XGBoost ranker is fitted once per d on the 2021 window (train-only fit; the validation predictions of that fit drive the books). It does not depend on `lambda_tc`, `lambda_beta` or the ME percentile, so those 384 combinations per d reuse the same predictions. The ME percentile changes only which stocks are short-eligible.
- The 1440 books run in parallel (`n_jobs = 16`, loky). The IR used is `sqrt(12) * mean(active) / std(active)` over the 24 validation months.
- The validation scores used here come from a model whose depth/trees were themselves picked on the same 2019-2020 months by IC. The validation IR is therefore optimistic. That only affects the choice, not the test period.
- `settings_log.json` stores every combination's `val_ir` and number of beta-relaxed months.

Tests and checks:

- Beta check (before the main run): OLS of the chosen validation book's monthly excess return on the S&P 500 excess return (Newey-West, 3 lags). Beta is clearly nonzero if `|t| > beta_check_t` (1.96). Then, before touching 2021-2026, `_pipe_beta_fix` builds three beta variants (`kf` = the Kalman betas; `kf_rfloor` = Kalman refit with every `R_j >= kalman_r_floor`, cached as `kalman_rfloor.parquet`; `b60_shrunk` = `(1 - beta_shrink) * beta_60m + beta_shrink * 1` from the raw panel, clipped at its 1st/99th percentile, `kf` where missing, `beta_var` from `kf`), runs the validation book with the current choice under each, and keeps the variant with the smallest `|beta|`. If that is not `kf`, the step-10 grid is rerun with it and the beta check repeated. Everything (each variant's validation beta and t, chosen variant, both choices) is in `settings_log.json` under `beta_fix`. Only validation months (2018-12 to 2020-11) are used.
- Structural leakage checks (`leakage_tests.json`, then asserted):
  - Every feature row flagged `has_filing = 1` has a filing in that calendar month, and no filing date is after its `eom`. (Filings are assigned to the month of their `filing_date`, so this holds by construction; the check guards against a later code change.)
  - Window ordering: train end < validation start <= validation end < test start, and train/validation end before the test year.
  - The latest target month actually used in any fit (`train_max_target`) is before 1 January of the test year.
  - No feature is named like the target or contains "lead", and none has absolute correlation above 0.999 with the target.
  - `(permno, eom)` is unique.
- Shuffled-label run: the same full pipeline, with `ret_exc_lead1m` permuted within each month in both train and validation. It reports test rank IC, ICIR, IR (with the chosen lambdas) and beta. The flags `ic_close_to_zero` (|IC| < 0.01) and `ir_close_to_zero` (|IR| < 1) are only printed as a warning if false; they are not asserted. Treat the shuffled IC as a noise baseline for comparison with the real IC, not as a leakage detector: the structural checks above are the leakage tests. (In the smoke-mode output currently in `output/`, the shuffled IC is -0.042, so `ic_close_to_zero` is false there while all structural checks pass. That is a 1500-stock subsample, so it says little about the full run.)

## 5. Ablations (step 12)

All five use the same schedule, the same frozen d, `lambda_tc`, `lambda_beta` and short-eligible ME percentile, and the same optimiser. `factors` = the 147 ranked characteristics; `flags` = the `_miss` columns.

| Name in code | Model | Features (as built in `main()`) | Question it answers |
|---|---|---|---|
| `1_ridge_147` | Ridge on raw `ret_exc_lead1m`, NaN -> 0, alpha grid | `factors` | What does the template's linear baseline earn? |
| `2_xgb_base` | XGB pairwise ranker | `factors` | Does the nonlinear ranker alone beat Ridge on the same 147 inputs? |
| `3_plus_gru` | XGB | `factors + flags + gru_1..gru_d` | Do missing-value flags and GRU embeddings add anything? |
| `4_plus_event` | XGB | `factors + flags + gru + 21 event columns` | Do 8-K event flags and history counts add anything? |
| `5_plus_tone` | XGB (this is the main run, not refitted) | `factors + flags + gru + event + tone_mean, tone_min, tone_surprise` | Does FinBERT tone add anything? |

Reported per ablation in `ablations.json`:

- `features`: number of features.
- `test`: rank IC, ICIR (mean/std x sqrt(12)) and number of months, over the test period.
- `val_2021`: the same for the 2019-2020 validation months of the 2021 window.
- `val_ic_by_year`: the best validation IC chosen in each of the six windows.
- `ir` and `beta` (with Newey-West t): of the test-period book. `val_ir`, `val_beta`: of the 2019-2020 validation book.
- `n_missing_returns`: held stock-months with no return (counted as 0).
- `test_has_filing_rows` (ablations 4 and 5 only): test IC restricted to stock-months with `has_filing = 1`. This isolates whether the 8-K columns help where an 8-K actually exists.

The missing-value flags first enter at ablation 3 (together with the GRU), not at ablation 2. So the ablation 2 to 3 step changes two things at once.

## 6. How the 8-K filings are used

Each filing is assigned only to the calendar month of its `filing_date` (month-end `eom`). Three groups of columns result, all in `cache/text_features.parquet`:

1. Event flags (month t only): `has_filing` (1 if at least one filing that month, else 0), `n_filings`, and 0/1 indicators `item_1_01, 1_02, 1_03, 2_01, 2_02, 2_03, 2_05, 2_06, 3_01, 4_01, 4_02, 5_02, 7_01, 8_01`. For stock-months without a filing, `n_filings` and `item_*` are null (NaN to XGBoost).
2. History counts: `hist_4_02, hist_5_02, hist_2_06, hist_1_02, hist_2_05` = the number of filings with that item over the trailing 12 calendar months t-11..t (including month t). These are 0 when there were none, for every stock-month (not NaN).
3. FinBERT tone (`ProsusAI/finbert`):
   - Anonymisation: the firm's `company_name` and `ticker` (from the filing row and from the panel row of that stock-month) are removed from the text with whole-word matching, replaced by a space. Names are matched case-insensitively. Tickers are matched case-sensitively, so tickers that are common words (ALL, NOW, FOR) do not delete ordinary text.
   - Chunking: the anonymised text is tokenised and cut into chunks of 510 tokens (+ CLS/SEP = 512), at most 8 chunks, so only about the first 4,080 tokens of a long filing are read. Per chunk, tone = P(positive) - P(negative). Per filing: `tone_mean` and `tone_min` over chunks (bf16 autocast on CUDA).
   - Per stock-month: `tone_mean` (mean over the month's filings), `tone_min` (min), and `tone_surprise` = `tone_mean` minus the average of the stock's monthly `tone_mean` over months t-12..t-1 (past months only; NaN if there is no history or no filing).
   - Only filings whose `(permno, month)` is in the universe are scored. Results are cached by `document_id`.

Is there an LLM agent reading the 8-Ks? No. The pipeline uses only item-code flags, filing counts and a FinBERT classifier score. It never extracts facts from the text. The local LLM agent (gpt-oss-20b served by vLLM on the GX10) is optional spec step A and is not built. It would add: a triage pass on every filing (material yes/no, event type, severity 0-3); for material filings, an item-specific JSON checklist of facts with verbatim supporting quotes and a history tool returning the same stock's earlier filings; a code verifier that drops any fact whose quote does not appear in the filing; and aggregation of the verified facts into stock-month columns for XGBoost. Spec steps B (cross-attention reader), C (Llama 3.1 8B look-ahead proof and fact-flip test) and D (extra ablations) depend on A. Step A is the natural next step if the 8-K columns show signal in ablations 4-5.

## 7. Outputs

`output/` (all written by one run of MAIN.py):

| File | Written by | Purpose |
|---|---|---|
| `holdings.csv` | `write_submission` | Submission. Columns `Date, PERMNO, TICKER, COMPANY NAME, WEIGHT`. `Date` = first day of the holding month; `WEIGHT` in percent of NAV, positive long, negative short. Missing ticker/name filled from the nearest-dated 8-K row; stocks still unlabeled are printed as a warning. |
| `returns.csv` | `write_submission` | Submission. Columns `Date, total, excess, active, long_leg, short_leg, benchmark, sp500`. |
| `performance.json` | `performance_pack` | Deck. Summary statistics: IR, Sharpe, arithmetic and geometric returns, calendar years, legs, drawdown, alpha/beta vs the S&P 500 (Newey-West, 3 lags), correlation, holdings counts, gross/net/turnover (average, min, max), largest weight, top-10 share, short-book market cap/dollar volume/small-cap share, top and bottom 10 contributors, OOS R-squared of `ret_hat`. |
| `figures/*.png` | `performance_pack` | Deck. `cumulative_returns`, `underwater` (drawdown), `rolling_active_return`, `rolling_ir`, `rolling_beta`, `return_histogram` (hurdle marked), `contributors`. Titles say "gross of trading costs". |
| `predictions.parquet` | `main` | Audit. `permno, eom, score, ret_hat` for all test months. |
| `settings_log.json` | `main` (written twice) | Audit. Step-10 choice, beta check, beta fix, all grid results, GRU logs, per-window XGBoost info (best depth/trees, validation IC, full IC grid), test IC/IR/beta. |
| `ablations.json` | `_pipe_ablations` | Audit and deck. Section 5. |
| `leakage_tests.json` | `_pipe_leakage` | Audit. Section 4. |

`cache/` holds the intermediate parquet files (section 9).

## 8. Judgment calls and deviations from the spec

Each item below was checked in the code.

1. `lambdarank_normalization=False` (`config.py`). With the xgboost 2.x default, pairwise gradients are normalised so the hessians become too small for `min_child_weight=200` to allow any split, and every depth gives the same model. The flag restores the classic behaviour. The spec lists no such setting.
1b. `lambdarank_pair_method="mean"` (`config.py`). xgboost 2.x defaults to `"topk"`, which builds pairs only among the top-ranked stocks of each month, so the model never learns the order of the rest of the cross-section. On synthetic data it reached a rank IC of 0.006 against 0.021 for `"mean"` (classic pairwise over all stocks). The first full run used the default: XGBoost test IC was negative, every variant lost to Ridge, and most windows stopped at 50 trees.
2. Kalman winsorising (`kalman.py`). The four raw betas are clipped at the 1st/99th percentile of the fit sample (`eom <= 2018-12`) before filtering, because raw values reach +/-1e4. Marked "not in spec" in the code.
3. Ticker anonymisation is case-sensitive; the spec says case-insensitive for both name and ticker. Names are case-insensitive. The reason is in section 6.
4. Short-eligible ME percentile is a fourth dimension of the step-10 grid (0.30/0.40/0.50). The spec says only "the universe thresholds" are tunable; the code tunes this one only. `universe_me_q` (0.20), the 30th dollar-volume percentile and the zero-trades median are fixed.
5. Missing-value flags enter the model at ablation 3 together with the GRU; ablation 2 uses the 147 features without flags (section 5).
6. `hist_*` counts are 0 (not NaN) for stock-months with no filings in the trailing window, defined for every stock-month. `has_filing = 0` rows keep `n_filings`, `item_*` and tone as NaN.
7. The shuffled-label IC is a noise baseline and only a warning (section 4). The spec says test IC and IR "must be close to 0"; the code prints a warning at |IC| >= 0.01 or |IR| >= 1 but never fails on it.
8. The beta check now acts, as the spec requires: if validation beta is clearly nonzero, the beta variants are compared and the step-10 grid rerun with the best (section 4). If no variant brings `|t|` under the threshold, the run still proceeds with the smallest-`|beta|` variant and a printed warning.
8b. Additions made after the first full run, all chosen or fixed on 2019-2020 only: (a) the validation beta check failed (spec-mandated fix, item 8); (b) a size-neutrality constraint `|sum size_z * w| <= size_band`, because the short universe was structurally lopsided (shorts must be large and liquid); (c) an L2 risk penalty `lambda_risk * sum(w^2)` as a fifth grid dimension, because the linear objective piled names at the 1.5% cap; (d) a factor-risk penalty `lambda_fac * w'XFX'w` (sixth grid dimension, see "Factor risk model"), added after the first full runs because realised volatility was about 21%/yr. The L2 penalty only spreads weight; it does not hedge factor bets.
9. The S&P 500 note. Spec step 13 assumes the FRED S&P 500 series, a price index without dividends. The supplied `SP500.csv` is a total return series instead (its `sp500_source` column is `sp500tr_total_return`), so `performance.json` correctly states "S&P 500 series is the total return index from SP500.csv". Say the same in the deck.
10. Rank IC and ICIR are computed on all rows with a non-null target (test rows with null targets are dropped). Months with no usable target are skipped.
11. Missing flag cut-off. The 20% missing threshold is computed on universe rows with `target_month <= 2018-12-31`, a date hard-coded in `data.preprocess`.
12. Missing returns. The spec asks to check the readme for delisting returns and otherwise use 0. The code uses 0 and reports the count; it has no delisting-return logic.
13. Two choices of the 2019-2020 period are slightly in-sample: the d/lambda/percentile selection uses the validation IR of a model whose depth/trees were selected on the same months.
14. The XGBoost depth/trees for each window (and the ridge alpha) are chosen by IC. `ret_hat` is a rank regression fitted on the validation scores of the train-only model, so OOS R-squared measures the calibration of that mapping, not the ranker's raw output.
15. The first month of each book (validation, test, each ablation) starts from cash, so turnover in that month is the full gross.
16. Tuning side effect. `_pipe_tune` calls `fit_predict_year` on the 2021 window, which also fits the final train+validation model and predicts 2021. Those test predictions are discarded in the tuning step (the grid is built only from validation predictions).
17. Only the first 8 chunks (about 4,000 tokens) of each filing are scored, so for long filings later sections (often exhibits) are never read.
18. Everything not cached is recomputed on every run (section 9), including the tuning grid and all XGBoost fits.
19. Book v2 was designed AFTER seeing the first full-run test results (concentration, beta -0.18, volatility, turnover). It is therefore not an untouched out-of-sample choice. Its parameters are stated assumptions or pre-2021 estimates and nothing was searched, but the decision to change the book was informed by test data. The spec book is still run and reported alongside (`spec_book`), and `v2_validation` shows v2 on 2019-2020 as a check.

## 9. How to run and rerun

Requirements: the five files in `data/` (not shipped in MAIN.py), plus polars, numpy, pandas, scipy, scikit-learn, xgboost, torch, transformers, cvxpy (Clarabel), statsmodels, joblib, matplotlib. `device` is `cuda` if available, else `cpu`.

Full run:

```
python MAIN.py
```

Smoke run (tiny CPU test, 1500 random permnos, GRU 2 epochs, 100 XGB trees, FinBERT on at most 40 filings):

```
ALPHA_SMOKE=1 python MAIN.py          # PowerShell: $env:ALPHA_SMOKE="1"; python MAIN.py
```

Smoke caches carry a `_smoke` suffix (e.g. `universe_smoke.parquet`) and do not collide with full-run caches. Smoke runs write to the same `output/` files as full runs, so rerun the full run before submitting. Smoke numbers say nothing about performance.

Editing code: change `src/*.py`, then run `python build_main.py` to regenerate MAIN.py. Do not edit MAIN.py by hand. Unit tests are in `tests/`.

Caching: `_pipe_cached` loads a parquet file if it exists and otherwise computes and writes it. There is no hash of the settings, so a changed setting does not invalidate a cache by itself. Delete the files below when you change what they depend on:

| Cache file (in `cache/`) | Contents | Delete when you change |
|---|---|---|
| `universe.parquet` | filtered, ranked panel with `_miss` flags, `short_eligible`, `sector` | `min_abs_prc`, `universe_me_q`, `missing_flag_threshold`, short-eligible thresholds, the input data. Then also delete `gru_d*.parquet` and `text_features.parquet` (they are built on the universe rows) |
| `kalman.parquet` + `kalman.json` | `beta_kf`, `beta_var` and fitted parameters | `kalman_obs`, `kalman_fit_end_eom`, the Kalman code |
| `kalman_rfloor.parquet` + `.json` | beta variant `kf_rfloor` (only built if the beta check fails) | `kalman_r_floor`, the Kalman code |
| `finbert_docs.parquet` | per-filing tone, keyed by `document_id` | the FinBERT model, chunking, or anonymisation code (cached tones are reused as they are) |
| `text_features.parquet` | event flags + stock-month tone features | `items_flag`, `items_hist`, tone aggregation, the universe, or after deleting `finbert_docs.parquet` |
| `gru_d8/16/32.parquet` + `.json` | embeddings and chosen weight decay | `gru_*` settings, `gru_d_grid`, the universe or factor list |

Not cached, so rerun on every execution: the step-10 grid, all XGBoost and Ridge fits, optimiser books, performance pack, submission files, ablations and the leakage tests. Changing `xgb_*`, `lambda_*_grid`, `short_me_q_grid`, optimiser constraints, or report code therefore needs no cache deletion. Changing `short_me_q` in CONFIG changes only the default used before tuning (the tuned value replaces it), and needs no cache deletion.

If you set `finbert_max_filings` (smoke default 40), only that many new filings are scored per run; the rest are scored on later runs, because the per-filing cache is incremental.

## Book v3 and book selection

Book v3 (`run_book_v3`) was designed after seeing test results. Per month: within-month z-score, exponentially smoothed with the past only (`v3_smooth_grid`, 1.0 = none); within each sector the top `v3_tail` of stocks are long and the bottom `v3_tail` of short-eligible stocks are short; sectors are weighted by stock count, names equal-weighted, leg sizes set so the book is beta-neutral on `beta_kf` (net clipped to `net_band`, weights capped at `max_weight`). With `book = "auto"` the pipeline picks the main book among spec, v2 and each v3 smoothing value by 2019-2020 validation IR only; the choice and all candidates' validation and test IR/beta are logged in `settings_log.json` under `book_selection` (test numbers are for transparency, never used to choose). Caveat: the spec book's validation IR is the maximum over a 1440-point grid, so it is optimistic relative to the untuned v3.
