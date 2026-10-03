"""
src/pipeline.py - Orchestration of steps 1-14: caching, step-10 tuning, step-11 leakage tests, step-12 ablations.
main(cfg) is called by MAIN.py; one short function per step.
"""
import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from joblib import Parallel, delayed

from src.data import load_inputs, build_universe, preprocess, short_eligible_mask
from src.gru import gru_embeddings
from src.text import event_flags, finbert_doc_tones, tone_features
from src.kalman import kalman_betas
from src.ranker import windows, fit_predict_year, run_schedule, rank_ic
from src.optimizer import run_book
from src.metrics import load_market, book_returns
from src.report import performance_pack, write_submission

_PIPE_T0 = time.time()
_PIPE_KEYS = ["permno", "eom"]
_PIPE_TONE = ["tone_mean", "tone_min", "tone_surprise"]


# ----------------------------------------------------------------------------- small helpers
def _pipe_log(msg):
    print(f"[{time.time() - _PIPE_T0:8.0f}s] {msg}", flush=True)


def _pipe_cached(path, fn):
    """Load parquet if present, else compute with fn() and write it."""
    path = Path(path)
    if path.exists():
        _pipe_log(f"cache hit {path.name}")
        return pl.read_parquet(path)
    out = fn()
    out.write_parquet(path)
    _pipe_log(f"cache written {path.name}")
    return out


def _pipe_cached_info(path, fn):
    """Like _pipe_cached, but fn() returns (frame, info) and info is kept in a sidecar JSON next to the parquet."""
    path = Path(path)
    side = path.with_suffix(".json")
    if path.exists():
        _pipe_log(f"cache hit {path.name}")
        return pl.read_parquet(path), (json.loads(side.read_text()) if side.exists() else None)
    out, info = fn()
    out.write_parquet(path)
    side.write_text(json.dumps(_pipe_clean(info), default=_pipe_jsonable))
    _pipe_log(f"cache written {path.name}")
    return out, info


def _pipe_path(cfg, name):
    """Cache path; smoke runs get their own files."""
    stem, ext = name.rsplit(".", 1)
    return Path(cfg["cache_dir"]) / f"{stem}{'_smoke' if cfg['smoke'] else ''}.{ext}"


def _pipe_jsonable(o):
    """json.dump default= hook: numpy / dates / paths / polars to plain Python."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, pl.DataFrame):
        return o.to_dicts()
    return str(o)


def _pipe_clean(o):
    """Recursively replace NaN floats by None (JSON has no NaN) and tuple keys by strings."""
    if isinstance(o, dict):
        return {str(k): _pipe_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_pipe_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


def _pipe_dump(obj, name, cfg):
    path = Path(cfg["output_dir"]) / name
    path.write_text(json.dumps(_pipe_clean(obj), indent=2, default=_pipe_jsonable))
    _pipe_log(f"wrote {path.name}")


def _pipe_with_short(df, q, cfg):
    """Recompute short_eligible for a given short_me_q (mask may be an Expr or a Series)."""
    m = short_eligible_mask(df, q, cfg)
    return df.with_columns(m.alias("short_eligible"))


def _pipe_attach_target(preds, frame):
    """Add ret_exc_lead1m (needed by rank_ic / oos_r2) to a predictions frame."""
    return preds.drop("ret_exc_lead1m", strict=False).join(
        frame.select(_PIPE_KEYS + ["ret_exc_lead1m"]), on=_PIPE_KEYS, how="left")


def _pipe_scored(preds, frame):
    """Predictions + optimizer inputs (short_eligible, betas, sector)."""
    cols = _PIPE_KEYS + ["short_eligible", "beta_kf", "beta_var", "sector"]
    return preds.select(_PIPE_KEYS + ["score"]).join(frame.select(cols), on=_PIPE_KEYS, how="left")


def _pipe_ir(rets):
    a = rets["active"].to_numpy()
    sd = a.std(ddof=1) if len(a) > 1 else np.nan
    return float(np.sqrt(12) * a.mean() / sd) if sd and sd > 0 else float("nan")


def _pipe_beta(rets, cfg):
    """OLS of monthly excess return on S&P excess (sp500 - rf), Newey-West; returns {beta, t, alpha_m}."""
    rf = rets["benchmark"].to_numpy() - cfg["premium_annual"] / 12
    x = rets["sp500"].to_numpy() - rf
    y = rets["excess"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return {"beta": float("nan"), "t": float("nan"), "alpha_m": float("nan")}
    r = sm.OLS(y[ok], sm.add_constant(x[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": cfg["nw_lags"]})
    return {"beta": float(r.params[1]), "t": float(r.tvalues[1]), "alpha_m": float(r.params[0])}


def _pipe_ic(preds, frame, cfg):
    """Mean rank IC, ICIR (mean/std*sqrt12) and month count of a predictions frame."""
    ic = rank_ic(_pipe_attach_target(preds, frame), cfg)["ic"].drop_nulls().to_numpy()
    sd = ic.std(ddof=1) if len(ic) > 1 else np.nan
    return {"ic": float(ic.mean()) if len(ic) else float("nan"),
            "icir": float(ic.mean() / sd * np.sqrt(12)) if len(ic) > 1 and sd > 0 else float("nan"),
            "n_months": int(len(ic))}


# ----------------------------------------------------------------------------- steps 1-8 (data, features)
def _pipe_data(cfg):
    """Steps 1-3: raw panel, 147 factors, filings, preprocessed universe (cached) and missing-flag columns."""
    panel, factors, filings = load_inputs(cfg)
    _pipe_log(f"loaded panel {panel.shape}, filings {filings.shape}")

    def build():
        return preprocess(build_universe(panel, cfg), factors, cfg)[0]
    df = _pipe_cached(_pipe_path(cfg, "universe.parquet"), build)
    flag_cols = [c for c in df.columns if c.endswith("_miss")]
    return panel, factors, filings, df, flag_cols


def _pipe_kalman(panel, df, cfg, logs):
    """Step 8 on the RAW panel (all stocks), joined onto the universe."""
    def build():
        return kalman_betas(panel, cfg)
    kal, logs["kalman"] = _pipe_cached_info(_pipe_path(cfg, "kalman.parquet"), build)
    return df.drop(["beta_kf", "beta_var"], strict=False).join(kal, on=_PIPE_KEYS, how="left")


def _pipe_text(filings, panel, df, cfg):
    """Steps 6-7: event flags + FinBERT tone features; returns (frame with text cols, event cols)."""
    keys = df.select(_PIPE_KEYS)

    def build():
        ev = event_flags(filings, keys, cfg)
        tones = finbert_doc_tones(filings, df, cfg)
        return ev.join(tone_features(filings, tones, keys, cfg), on=_PIPE_KEYS, how="left")
    txt = _pipe_cached(_pipe_path(cfg, "text_features.parquet"), build)
    event_cols = [c for c in txt.columns if c not in _PIPE_KEYS + _PIPE_TONE]
    return df.join(txt, on=_PIPE_KEYS, how="left"), event_cols


def _pipe_gru(df, factors, cfg, logs):
    """Step 4 for each d in the grid; returns {d: embedding frame}."""
    embs = {}
    for d in cfg["gru_d_grid"]:
        def build(d=d):
            return gru_embeddings(df, factors, d, cfg)
        embs[d], info = _pipe_cached_info(_pipe_path(cfg, f"gru_d{d}.parquet"), build)
        logs.setdefault("gru", {})[d] = info
    return embs


def _pipe_frame(df, emb, q, cfg):
    """Full modelling frame for one GRU d and short_me_q."""
    return _pipe_with_short(df.join(emb, on=_PIPE_KEYS, how="left"), q, cfg)


# ----------------------------------------------------------------------------- step 10 tuning
def _pipe_book_job(scored, ltc, lb, cfg):
    """Worker: one optimizer run over all months of `scored`."""
    return run_book(scored, ltc, lb, cfg)


def _pipe_tune(frames, feats, market, cfg):
    """Step 10: XGB on the 2021 window per d, then grid over (lam_tc, lam_beta, short_me_q) by validation IR."""
    win = windows(cfg)[0]
    vals, tasks = {}, []
    for d in cfg["gru_d_grid"]:
        fr = frames[d]
        fr = fr.filter(pl.col("target_month") <= win["test"][1])
        _, val_pred, info = fit_predict_year(fr, feats[d], win, cfg)
        vals[d] = val_pred
        _pipe_log(f"tune: d={d} xgb val IC {info.get('val_ic')}")
        for q in cfg["short_me_q_grid"]:
            sc = _pipe_scored(val_pred, _pipe_with_short(fr, q, cfg))
            tasks += [(d, ltc, lb, q, sc) for ltc in cfg["lambda_tc_grid"] for lb in cfg["lambda_beta_grid"]]
    _pipe_log(f"tune: {len(tasks)} optimizer runs on {cfg['n_jobs']} workers")
    res = Parallel(n_jobs=cfg["n_jobs"], backend="loky")(
        delayed(_pipe_book_job)(sc, ltc, lb, cfg) for d, ltc, lb, q, sc in tasks)
    grid, best = [], None
    for (d, ltc, lb, q, _), (w, log) in zip(tasks, res):
        rets, _ = book_returns(w, frames[d], market, cfg)
        ir = _pipe_ir(rets)
        grid.append({"d": d, "lam_tc": ltc, "lam_beta": lb, "short_me_q": q, "val_ir": ir,
                     "n_relaxed_months": sum(1 for r in log if (r.get("beta_tol") or 0) > cfg["beta_tol"] + 1e-12)})
        if np.isfinite(ir) and (best is None or ir > best[0]):
            best = (ir, grid[-1], rets)
    choice = {k: best[1][k] for k in ["d", "lam_tc", "lam_beta", "short_me_q"]} | {"val_ir": best[0]}
    return choice, grid, best[2]


def _pipe_beta_check(rets, cfg):
    """Chosen validation book vs S&P: loud warning when |beta| > 0.2 and significant."""
    b = _pipe_beta(rets, cfg)
    b["warning"] = bool(abs(b["beta"]) > 0.2 and abs(b["t"]) > 1.96)
    if b["warning"]:
        print(f"WARNING: validation book beta to S&P = {b['beta']:.3f} (t={b['t']:.2f}) is large and significant", flush=True)
    return b


# ----------------------------------------------------------------------------- full run + step 11 / 12 helpers
def _pipe_run_book_stats(preds, frame, choice, market, cfg):
    """Book with chosen lambdas on test predictions -> (weights, rets, IR, beta)."""
    w, _ = run_book(_pipe_scored(preds, frame), choice["lam_tc"], choice["lam_beta"], cfg)
    rets, n_miss = book_returns(w, frame, market, cfg)
    return w, rets, n_miss


def _pipe_leak_filings(filings, feats):
    """Number of stock-months flagged has_filing=1 without a filing in that calendar month, and filings used after their eom."""
    f = filings.select("permno", pl.col("filing_date").dt.month_end().alias("eom"), "filing_date")
    have = f.group_by(_PIPE_KEYS).agg(pl.col("filing_date").max().alias("last"))
    x = feats.filter(pl.col("has_filing") == 1).select(_PIPE_KEYS).join(have, on=_PIPE_KEYS, how="left")
    return {"flagged_without_filing": int(x["last"].is_null().sum()),
            "filing_after_eom": int((x["last"] > x["eom"]).sum())}


def _pipe_leak_windows(wins):
    """Violations of train < val < test ordering by target month, and of 'train ends before test year'."""
    bad = 0
    for w in wins:
        bad += not (w["train"][1] < w["val"][0] <= w["val"][1] < w["test"][0])
        bad += w["train"][1].year >= w["year"] or w["val"][1].year >= w["year"]
        bad += w["test"][0].year != w["year"]
    return int(bad)


def _pipe_leak_target(df, features, target="ret_exc_lead1m"):
    """Feature columns that are the target (by name) or numerically identical / perfectly correlated with it."""
    bad = [c for c in features if c == target or "lead" in c]
    num = [c for c in features if c not in bad and df.schema[c].is_numeric()]
    if num:
        cors = df.select([pl.corr(pl.col(c).cast(pl.Float64), pl.col(target)).alias(c) for c in num]).row(0)
        bad += [c for c, r in zip(num, cors) if r is not None and abs(r) > 0.999]
    return bad


def _pipe_leak_target_months(infos):
    """Windows whose fitted train/val rows reach a target month >= Y-01-01 (taken from the actual fits)."""
    return [{"year": i["year"], "max_target_month": i["train_max_target"]}
            for i in infos if i["train_max_target"] >= date(i["year"], 1, 1)]


def _pipe_leak_dups(df):
    return int(df.height - df.unique(subset=_PIPE_KEYS).height)


def _pipe_leakage(frame, filings, feats, event_cols, preds_shuf, choice, market, cfg, infos):
    """Step 11: structural checks plus shuffled-label run (test IC and IR should be ~0)."""
    ev = frame.filter(pl.col("has_filing") == 1) if "has_filing" in frame.columns else frame.head(0)
    out = {"filings": _pipe_leak_filings(filings, ev) if ev.height else {"flagged_without_filing": 0, "filing_after_eom": 0},
           "window_violations": _pipe_leak_windows(windows(cfg)),
           "target_like_features": _pipe_leak_target(frame, feats),
           "duplicate_permno_eom": _pipe_leak_dups(frame),
           "train_target_month_violations": _pipe_leak_target_months(infos),
           "shuffled_ic": _pipe_ic(preds_shuf, frame, cfg)}
    _, rets, _ = _pipe_run_book_stats(preds_shuf, frame, choice, market, cfg)
    out["shuffled_ir"] = _pipe_ir(rets)
    out["shuffled_beta"] = _pipe_beta(rets, cfg)
    out["passed_structural"] = (out["filings"]["flagged_without_filing"] == 0 and out["filings"]["filing_after_eom"] == 0
                                and out["window_violations"] == 0 and not out["target_like_features"]
                                and out["duplicate_permno_eom"] == 0 and not out["train_target_month_violations"])
    out["ic_close_to_zero"] = bool(abs(out["shuffled_ic"]["ic"]) < 0.01)
    out["ir_close_to_zero"] = bool(abs(out["shuffled_ir"]) < 1.0)
    if not (out["ic_close_to_zero"] and out["ir_close_to_zero"]):
        print(f"WARNING: shuffled-label run not ~0: IC {out['shuffled_ic']['ic']:.4f}, IR {out['shuffled_ir']:.2f}", flush=True)
    return out


def _pipe_ablations(frame, runs, choice, market, cfg):
    """Step 12: per ablation - validation/test rank IC, ICIR, IR, beta (chosen lambdas); IC on has_filing=1 rows for 4 and 5."""
    jobs = [r[k] for r in runs.values() for k in ("preds", "val")]
    books = Parallel(n_jobs=min(cfg["n_jobs"], len(jobs)), backend="loky")(
        delayed(_pipe_run_book_stats)(p, frame, choice, market, cfg) for p in jobs)
    out = {"features_note": "ablation 1 = 147 factors (ridge), 2 = the 147 factors only (XGB); missing-value flags enter from "
                            "ablation 3 onward; 5 = full main feature set. val_ir/val_beta: book on 2021-window val_pred."}
    for i, (name, r) in enumerate(runs.items()):
        _, rets, n_miss = books[2 * i]
        vrets = books[2 * i + 1][1]
        o = {"features": r["n_features"], "val_ir": _pipe_ir(vrets), "val_beta": _pipe_beta(vrets, cfg)["beta"], "test": _pipe_ic(r["preds"], frame, cfg), "val_2021": _pipe_ic(r["val"], frame, cfg),
             "val_ic_by_year": [i.get("val_ic") for i in r["infos"]],
             "ir": _pipe_ir(rets), "beta": _pipe_beta(rets, cfg), "n_missing_returns": n_miss}
        if r.get("has_filing_ic"):
            hf = frame.filter(pl.col("has_filing") == 1).select(_PIPE_KEYS)
            o["test_has_filing_rows"] = _pipe_ic(r["preds"].join(hf, on=_PIPE_KEYS, how="inner"), frame, cfg)
        out[name] = o
    return out


# ----------------------------------------------------------------------------- main
def main(cfg):
    """Run steps 1-14 end to end."""
    logs, out_dir = {}, Path(cfg["output_dir"])
    panel, factors, filings, df, flag_cols = _pipe_data(cfg)
    df = _pipe_kalman(panel, df, cfg, logs)
    df, event_cols = _pipe_text(filings, panel, df, cfg)
    embs = _pipe_gru(df, factors, cfg, logs)
    market = load_market(cfg)

    base = factors + flag_cols
    gru_cols = lambda d: [f"gru_{i}" for i in range(1, d + 1)]
    feats = {d: base + gru_cols(d) + event_cols + _PIPE_TONE for d in cfg["gru_d_grid"]}
    frames = {d: _pipe_frame(df, embs[d], cfg["short_me_q"], cfg) for d in cfg["gru_d_grid"]}

    # step 10: tuning on 2019-2020 validation
    choice, grid, val_rets = _pipe_tune(frames, feats, market, cfg)
    beta_chk = _pipe_beta_check(val_rets, cfg)
    _pipe_log(f"step 10 choice {choice}; validation beta {beta_chk}")
    d = choice["d"]
    frame = _pipe_with_short(frames[d], choice["short_me_q"], cfg)
    full = feats[d]
    _pipe_dump({"choice": choice, "beta_check": beta_chk, "grid": grid, "logs": logs}, "settings_log.json", cfg)

    # step 5 + 9: full schedule, predictions, book, performance, submission
    preds, val_2021, infos = run_schedule(frame, full, cfg)
    preds = preds.select(_PIPE_KEYS + ["score", "ret_hat"])
    preds.write_parquet(out_dir / "predictions.parquet")
    weights, rets, n_miss = _pipe_run_book_stats(preds, frame, choice, market, cfg)
    _pipe_log(f"full book: IR {_pipe_ir(rets):.3f}, missing returns {n_miss}")
    performance_pack(rets, weights, frame, filings, _pipe_attach_target(preds, frame), cfg)
    write_submission(weights, frame, filings, rets, cfg)
    _pipe_dump({"choice": choice, "beta_check": beta_chk, "grid": grid, "logs": logs, "xgb_infos": infos,
                "test_ic": _pipe_ic(preds, frame, cfg), "test_ir": _pipe_ir(rets), "test_beta": _pipe_beta(rets, cfg)},
               "settings_log.json", cfg)

    # step 12: ablations (each run reuses the chosen lambdas / short_me_q / d)
    sets = {"1_ridge_147": (factors, "ridge", False),
            "2_xgb_base": (factors, "xgb", False),
            "3_plus_gru": (base + gru_cols(d), "xgb", False),
            "4_plus_event": (base + gru_cols(d) + event_cols, "xgb", True)}
    runs = {}
    for name, (cols, model, hf) in sets.items():
        p, v, i = run_schedule(frame, cols, cfg, model=model)
        runs[name] = {"preds": p, "val": v, "infos": i, "n_features": len(cols), "has_filing_ic": hf}
        _pipe_log(f"ablation {name} done")
    runs["5_plus_tone"] = {"preds": preds, "val": val_2021, "infos": infos, "n_features": len(full), "has_filing_ic": True}
    _pipe_dump(_pipe_ablations(frame, runs, choice, market, cfg), "ablations.json", cfg)

    # step 11: leakage tests incl. shuffled-label run
    shuf, _, _ = run_schedule(frame, full, cfg, shuffle=True)
    leak = _pipe_leakage(frame, filings, full, event_cols, shuf, choice, market, cfg, infos)
    _pipe_dump(leak, "leakage_tests.json", cfg)
    assert leak["filings"]["flagged_without_filing"] == 0 and leak["filings"]["filing_after_eom"] == 0,         f"leakage: filing_date after eom / flagged without filing: {leak['filings']}"
    assert leak["window_violations"] == 0, "leakage: train < val < test window ordering violated"
    assert not leak["train_target_month_violations"], f"leakage: training target months reach test year: {leak['train_target_month_violations']}"
    assert not leak["target_like_features"], f"leakage: target-identical features: {leak['target_like_features']}"
    assert leak["duplicate_permno_eom"] == 0, "leakage: duplicate (permno, eom) keys"
    _pipe_log("pipeline complete")
