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
from src.risk import risk_model
from src.optimizer import run_book, run_book_v2, run_book_v3
from src.metrics import load_market, book_returns
from src.report import performance_pack, write_submission
from src.agent import run_agent

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


def _pipe_scored(preds, frame, cfg, rm=None):
    """Predictions + optimizer inputs (short_eligible, betas, sector, size_z = within-month z of log market cap, risk_factors columns);
    with rm also spec_var (specific variance from the risk model; null -> month median, then global median)."""
    lm = pl.when(pl.col("me_raw") > 0).then(pl.col("me_raw").log())
    size_z = ((lm - lm.mean().over("eom")) / lm.std().over("eom")).fill_nan(None).fill_null(0.0).alias("size_z")
    cols = _PIPE_KEYS + ["short_eligible", "beta_kf", "beta_var", "sector", size_z] + [c for c in cfg["risk_factors"] if c not in ("sector",)]
    out = preds.select(_PIPE_KEYS + ["score"]).join(frame.select(cols), on=_PIPE_KEYS, how="left")
    if rm is not None:
        sv = pl.col("spec_var").fill_nan(None)
        out = out.join(rm["spec"].select(_PIPE_KEYS + ["spec_var"]), on=_PIPE_KEYS, how="left").with_columns(
            sv.fill_null(sv.median().over("eom")).fill_null(sv.median()))
    return out


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


def _pipe_stats(rets, cfg):
    """IR, S&P beta and annualised Sharpe of the excess return of a book_returns frame."""
    x = rets["excess"].to_numpy()
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"ir": _pipe_ir(rets), "beta": _pipe_beta(rets, cfg)["beta"],
            "sharpe": float(np.sqrt(12) * x.mean() / sd) if sd and sd > 0 else float("nan")}


def _pipe_ic_map(infos, frame, cfg, val=False):
    """Pre-test IC for the v2 book: val=True -> the 2021 window's validation IC (float, for validation books);
    else dict test eom -> that window's validation IC (windows without one get the mean of the others)."""
    if val:
        return float(infos[0].get("val_ic") or 0.0)
    ics = {i["year"]: i.get("val_ic") for i in infos}
    fb = float(np.mean([v for v in ics.values() if v is not None] or [0.0]))
    tm = frame.select("eom", "target_month").unique()
    return {e: float(ics[t.year] if ics[t.year] is not None else fb) for e, t in tm.iter_rows() if t.year in ics}


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
def _pipe_book_job(scored, ltc, lb, lr, lf, rm, cfg):
    """Worker: one optimizer run over all months of `scored`."""
    return run_book(scored, ltc, lb, cfg, lam_risk=lr, lam_fac=lf, rm=rm)


def _pipe_tune(frames, feats, market, cfg, rm, vals=None):
    """Step 10: XGB on the 2021 window per d (skipped if `vals` given), then grid over
    (lam_tc, lam_beta, short_me_q, lam_risk, lam_fac) by validation IR. Validation months only. -> (choice, grid, rets, vals)"""
    win = windows(cfg)[0]
    fit, vals, tasks = vals is None, dict(vals or {}), []
    for d in cfg["gru_d_grid"]:
        fr = frames[d].filter(pl.col("target_month") <= win["test"][1])
        if fit:
            _, vals[d], info = fit_predict_year(fr, feats[d], win, cfg)
            _pipe_log(f"tune: d={d} xgb val IC {info.get('val_ic')}")
        for q in cfg["short_me_q_grid"]:
            sc = _pipe_scored(vals[d], _pipe_with_short(fr, q, cfg), cfg)
            tasks += [(d, ltc, lb, q, lr, lf, sc) for ltc in cfg["lambda_tc_grid"] for lb in cfg["lambda_beta_grid"]
                      for lr in cfg["lambda_risk_grid"] for lf in cfg["lambda_fac_grid"]]
    _pipe_log(f"tune: {len(tasks)} optimizer runs on {cfg['n_jobs']} workers")
    res = Parallel(n_jobs=cfg["n_jobs"], backend="loky")(
        delayed(_pipe_book_job)(sc, ltc, lb, lr, lf, rm, cfg) for d, ltc, lb, q, lr, lf, sc in tasks)
    grid, best = [], None
    for (d, ltc, lb, q, lr, lf, _), (w, log) in zip(tasks, res):
        rets, _ = book_returns(w, frames[d], market, cfg)
        ir = _pipe_ir(rets)
        grid.append({"d": d, "lam_tc": ltc, "lam_beta": lb, "short_me_q": q, "lam_risk": lr, "lam_fac": lf, "val_ir": ir,
                     "n_relaxed_months": sum(1 for r in log if (r.get("beta_tol") or 0) > cfg["beta_tol"] + 1e-12)})
        if np.isfinite(ir) and (best is None or ir > best[0]):
            best = (ir, grid[-1], rets)
    choice = {k: best[1][k] for k in ["d", "lam_tc", "lam_beta", "short_me_q", "lam_risk", "lam_fac"]} | {"val_ir": best[0]}
    return choice, grid, best[2], vals


def _pipe_beta_check(rets, cfg):
    """Chosen validation book vs S&P: beta is clearly nonzero (warning) when |t| > beta_check_t."""
    b = _pipe_beta(rets, cfg)
    b["warning"] = bool(abs(b["t"]) > cfg["beta_check_t"])
    if b["warning"]:
        print(f"WARNING: validation book beta to S&P = {b['beta']:.3f} (t={b['t']:.2f}) is clearly nonzero", flush=True)
    return b


def _pipe_beta_variant(name, df, panel, cfg, logs):
    """Beta estimates [permno, eom, beta_kf, beta_var] for one variant (kf_rfloor is cached)."""
    kf = df.select(_PIPE_KEYS + ["beta_kf", "beta_var"])
    if name == "kf":
        return kf
    if name == "kf_rfloor":
        def build():
            return kalman_betas(panel, cfg, r_floor=cfg["kalman_r_floor"])
        out, logs["kalman_rfloor"] = _pipe_cached_info(_pipe_path(cfg, "kalman_rfloor.parquet"), build)
        return df.select(_PIPE_KEYS).join(out, on=_PIPE_KEYS, how="left")
    k, b = cfg["beta_shrink"], pl.col("beta_60m")          # b60_shrunk; raw beta_60m clipped at its 1st/99th pct (raw reaches +/-1e4)
    fit = panel.filter(pl.col("eom") <= date.fromisoformat(cfg["kalman_fit_end_eom"]))["beta_60m"]   # pre-2019 only
    lo, hi = fit.quantile(0.01), fit.quantile(0.99)
    raw = panel.select(_PIPE_KEYS + [b.clip(lo, hi).alias("beta_60m")])
    return kf.join(raw, on=_PIPE_KEYS, how="left").with_columns(
        beta_kf=pl.when(b.is_not_null()).then((1 - k) * b + k).otherwise(pl.col("beta_kf"))).drop("beta_60m")


def _pipe_swap_beta(frames, bv):
    return {d: fr.drop(["beta_kf", "beta_var"]).join(bv, on=_PIPE_KEYS, how="left") for d, fr in frames.items()}


def _pipe_beta_fix(df, panel, frames, feats, vals, choice, chk, market, cfg, rm, logs):
    """Step 10 beta fix: validation beta of each variant at the current choice; if the best is not "kf", redo the grid.
    -> (frames, choice, grid, check, log); grid is None when "kf" is kept."""
    fix, swapped = {"variants": {}, "choice_kf": choice, "check_kf": chk}, {}
    for name in cfg["beta_variants"]:
        swapped[name] = _pipe_swap_beta(frames, _pipe_beta_variant(name, df, panel, cfg, logs))
        fr = _pipe_with_short(swapped[name][choice["d"]], choice["short_me_q"], cfg)
        v = fix["variants"][name] = _pipe_beta(_pipe_run_book_stats(vals[choice["d"]], fr, choice, market, cfg, rm, spec=True)[1], cfg)
        _pipe_log(f"beta fix: {name} val beta {v['beta']:.3f} (t={v['t']:.2f})")
    best = min(fix["variants"], key=lambda n: abs(fix["variants"][n]["beta"]) if np.isfinite(fix["variants"][n]["beta"]) else np.inf)
    fix["chosen"], grid = best, None
    if best != "kf":
        frames = swapped[best]
        choice, grid, rets, _ = _pipe_tune(frames, feats, market, cfg, rm, vals)
        chk = _pipe_beta_check(rets, cfg)
        fix |= {"choice_final": choice, "check_final": chk}
    _pipe_log(f"beta fix: variant {best}; val beta {chk['beta']:.3f} (t={chk['t']:.2f}); choice {choice}")
    return frames, choice, grid, chk, fix


# ----------------------------------------------------------------------------- full run + step 11 / 12 helpers
def _pipe_book(preds, frame, choice, cfg, rm, ic, spec=False):
    """Main book cfg["book"]: "spec" (chosen lambdas), "v2" (alpha = ic * xs_vol * z) or "v3_s<smooth>" (sector-neutral tail book);
    spec=True forces the spec book. Logs the number of months with relaxed constraints. -> (weights, log)"""
    sc = _pipe_scored(preds, frame, cfg, rm)
    book = "spec" if spec else cfg["book"]
    if book == "v2":
        w, log = run_book_v2(sc, cfg, rm, ic)
    elif book.startswith("v3_s"):
        w, log = run_book_v3(sc, cfg, float(book[4:]))
    else:
        book = "spec"
        w, log = run_book(sc, choice["lam_tc"], choice["lam_beta"], cfg, lam_risk=choice["lam_risk"], lam_fac=choice["lam_fac"], rm=rm)
    n_rel = sum(1 for r in log if isinstance(r, dict) and any(v for k, v in r.items() if "relax" in k))
    _pipe_log(f"book {book}: {len(log)} months, {n_rel} with relaxed constraints")
    return w, log


def _pipe_run_book_stats(preds, frame, choice, market, cfg, rm, ic=None, spec=False):
    """Main book (see _pipe_book) on `preds` -> (weights, rets, n_missing_returns)."""
    w, _ = _pipe_book(preds, frame, choice, cfg, rm, ic, spec)
    rets, n_miss = book_returns(w, frame, market, cfg)
    return w, rets, n_miss


def _pipe_candidates(cfg):
    return ["spec", "v2"] + [f"v3_s{x}" for x in cfg["v3_smooth_grid"]]


def _pipe_select_book(val_2021, preds, frame, choice, market, cfg, rm, infos):
    """cfg["book"] == "auto": pick the candidate with the highest 2019-2020 validation IR (test stats are computed afterwards, for the
    record only). Explicit cfg["book"] is kept. -> (cfg with the chosen book, book_selection dict)"""
    ic_val, ic_test = _pipe_ic_map(infos, frame, cfg, val=True), _pipe_ic_map(infos, frame, cfg)
    cands = _pipe_candidates(cfg) if cfg["book"] == "auto" else [cfg["book"]]
    val = {c: _pipe_stats(_pipe_run_book_stats(val_2021, frame, choice, market, {**cfg, "book": c}, rm, ic_val)[1], cfg) for c in cands}
    # market-neutral mandate: only books with |validation beta| <= 0.2 are eligible; else the smallest |beta|
    ok = [c for c in cands if np.isfinite(val[c]["beta"]) and abs(val[c]["beta"]) <= 0.2]
    best = (max(ok, key=lambda c: val[c]["ir"] if np.isfinite(val[c]["ir"]) else -np.inf) if ok
            else min(cands, key=lambda c: abs(val[c]["beta"]) if np.isfinite(val[c]["beta"]) else np.inf))
    cfg = {**cfg, "book": best}
    test = {c: _pipe_stats(_pipe_run_book_stats(preds, frame, choice, market, {**cfg, "book": c}, rm, ic_test)[1], cfg) for c in cands}
    _pipe_log("book selection (val IR/beta): " + ", ".join(f"{c} {val[c]['ir']:.3f}/{val[c]['beta']:.3f}" for c in cands) + f" -> {best}")
    return cfg, {"rule": "highest 2019-2020 validation IR among books with |validation beta| <= 0.2", "chosen": best, "validation": val, "test_for_transparency": test}


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


def _pipe_leakage(frame, filings, feats, event_cols, preds_shuf, choice, market, cfg, rm, infos, shuf_infos=None):
    """Step 11: structural checks plus shuffled-label run (test IC and IR should be ~0)."""
    ev = frame.filter(pl.col("has_filing") == 1) if "has_filing" in frame.columns else frame.head(0)
    out = {"filings": _pipe_leak_filings(filings, ev) if ev.height else {"flagged_without_filing": 0, "filing_after_eom": 0},
           "window_violations": _pipe_leak_windows(windows(cfg)),
           "target_like_features": _pipe_leak_target(frame, feats),
           "duplicate_permno_eom": _pipe_leak_dups(frame),
           "train_target_month_violations": _pipe_leak_target_months(infos),
           "shuffled_ic": _pipe_ic(preds_shuf, frame, cfg)}
    _, rets, _ = _pipe_run_book_stats(preds_shuf, frame, choice, market, cfg, rm, _pipe_ic_map(shuf_infos or infos, frame, cfg))
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


def _pipe_ablations(frame, runs, choice, market, cfg, rm):
    """Step 12: per ablation - validation/test rank IC, ICIR, IR, beta (chosen lambdas); IC on has_filing=1 rows for 4 and 5. Main book, each ablation's own val ICs."""
    jobs = [(r["preds"], _pipe_ic_map(r["infos"], frame, cfg)) if k == "preds" else (r["val"], _pipe_ic_map(r["infos"], frame, cfg, val=True))
            for r in runs.values() for k in ("preds", "val")]
    books = Parallel(n_jobs=min(cfg["n_jobs"], len(jobs)), backend="loky")(
        delayed(_pipe_run_book_stats)(p, frame, choice, market, cfg, rm, ic) for p, ic in jobs)
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
    t0 = time.time()
    rm = risk_model(frames[cfg["gru_d_grid"][0]], cfg, market)     # once per run: independent of d, betas, short_me_q
    _pipe_log(f"risk model: {len(rm['L'])} months with a covariance, {time.time() - t0:.1f}s")

    # step 10: tuning on 2019-2020 validation
    choice, grid, val_rets, vals = _pipe_tune(frames, feats, market, cfg, rm)
    beta_chk, beta_fix = _pipe_beta_check(val_rets, cfg), None
    if beta_chk["warning"]:
        frames, choice, grid2, beta_chk, beta_fix = _pipe_beta_fix(df, panel, frames, feats, vals, choice, beta_chk, market, cfg, rm, logs)
        if grid2:
            beta_fix["grid_kf"], grid = grid, grid2
    _pipe_log(f"step 10 choice {choice}; validation beta {beta_chk}")
    d = choice["d"]
    frame = _pipe_with_short(frames[d], choice["short_me_q"], cfg)
    full = feats[d]
    _pipe_dump({"choice": choice, "beta_check": beta_chk, "beta_fix": beta_fix, "grid": grid, "logs": logs}, "settings_log.json", cfg)

    # step 5 + 9: full schedule, predictions, book, performance, submission
    preds, val_2021, infos = run_schedule(frame, full, cfg)
    preds = preds.select(_PIPE_KEYS + ["score", "ret_hat"])
    preds.write_parquet(out_dir / "predictions.parquet")
    ic_test = _pipe_ic_map(infos, frame, cfg)           # per test year: that window's validation IC (a pre-test quantity)
    cfg, book_sel = _pipe_select_book(val_2021, preds, frame, choice, market, cfg, rm, infos)
    weights, rets, n_miss = _pipe_run_book_stats(preds, frame, choice, market, cfg, rm, ic_test)
    _pipe_log(f"full book ({cfg['book']}): IR {_pipe_ir(rets):.3f}, missing returns {n_miss}")
    # comparison: spec book on the same test predictions; main book on 2019-2020 validation (a check, nothing tuned on it)
    test_stats = _pipe_stats(rets, cfg)
    spec_book = _pipe_stats(_pipe_run_book_stats(preds, frame, choice, market, cfg, rm, spec=True)[1], cfg)
    v2_val = _pipe_stats(_pipe_run_book_stats(val_2021, frame, choice, market, cfg, rm, _pipe_ic_map(infos, frame, cfg, val=True))[1], cfg)
    _pipe_log(f"{cfg['book']} test IR {test_stats['ir']:.3f} beta {test_stats['beta']:.3f} | spec test IR {spec_book['ir']:.3f} "
              f"beta {spec_book['beta']:.3f} | {cfg['book']} val IR {v2_val['ir']:.3f} beta {v2_val['beta']:.3f}")
    performance_pack(rets, weights, frame, filings, _pipe_attach_target(preds, frame), cfg)
    write_submission(weights, frame, filings, rets, cfg)
    _pipe_dump({"choice": choice, "beta_check": beta_chk, "beta_fix": beta_fix, "grid": grid, "logs": logs, "xgb_infos": infos,
                "test_ic": _pipe_ic(preds, frame, cfg), "test_ir": _pipe_ir(rets), "test_beta": _pipe_beta(rets, cfg),
                "book": cfg["book"], "test_stats": test_stats, "spec_book": spec_book, "v2_validation": v2_val, "book_selection": book_sel},
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
    _pipe_dump(_pipe_ablations(frame, runs, choice, market, cfg, rm), "ablations.json", cfg)

    # step 11: leakage tests incl. shuffled-label run
    shuf, _, shuf_infos = run_schedule(frame, full, cfg, shuffle=True)
    leak = _pipe_leakage(frame, filings, full, event_cols, shuf, choice, market, cfg, rm, infos, shuf_infos)
    _pipe_dump(leak, "leakage_tests.json", cfg)
    assert leak["filings"]["flagged_without_filing"] == 0 and leak["filings"]["filing_after_eom"] == 0,         f"leakage: filing_date after eom / flagged without filing: {leak['filings']}"
    assert leak["window_violations"] == 0, "leakage: train < val < test window ordering violated"
    assert not leak["train_target_month_violations"], f"leakage: training target months reach test year: {leak['train_target_month_violations']}"
    assert not leak["target_like_features"], f"leakage: target-identical features: {leak['target_like_features']}"
    assert leak["duplicate_permno_eom"] == 0, "leakage: duplicate (permno, eom) keys"
    _pipe_log("pipeline complete")
    if cfg.get("agent_enabled"):
        try:
            run_agent(cfg)
        except Exception as e:
            print(f"[agent] warning: explanation agent failed: {e}")
