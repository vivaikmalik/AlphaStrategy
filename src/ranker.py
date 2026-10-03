"""
src/ranker.py - Step 5: XGBRanker (rank:pairwise, qid = eom) with walk-forward windows, plus the Ridge baseline.
Scores are cross-sectional only; ret_hat comes from a rank regression fitted on validation scores of the train-only model.
"""
import calendar
from datetime import date

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import Ridge


def _rk_me(s):
    """'YYYY-MM' -> month-end date."""
    y, m = int(s[:4]), int(s[5:7])
    return date(y, m, calendar.monthrange(y, m)[1])


def windows(cfg):
    """One dict per test year, by TARGET month: train 2015-02..Y-3, val Y-2..Y-1, test Y (2026 ends at last_test_month)."""
    first, last = _rk_me(cfg["first_train_month"]), _rk_me(cfg["last_test_month"])
    return [dict(year=y, train=(first, date(y - 3, 12, 31)), val=(date(y - 2, 1, 31), date(y - 1, 12, 31)),
                 test=(date(y, 1, 31), min(date(y, 12, 31), last))) for y in cfg["test_years"]]


def rank_ic(pred, cfg=None):
    """Monthly Spearman IC of score vs ret_exc_lead1m (Pearson on within-month average ranks); null targets dropped."""
    return (pred.drop_nulls("ret_exc_lead1m").group_by("eom")
            .agg(pl.corr(pl.col("score").rank(), pl.col("ret_exc_lead1m").rank()).alias("ic")).sort("eom"))


def _rk_mean_ic(eom, score, y):
    """Mean monthly rank IC from numpy arrays (months with undefined IC are skipped)."""
    ic = rank_ic(pl.DataFrame({"eom": eom, "score": score, "ret_exc_lead1m": y}))["ic"].drop_nans().drop_nulls()
    return float(ic.mean()) if len(ic) else float("nan")


def _rk_labels(eom, y):
    """Within-month decile 0-9 of y: floor(10*(rank_ordinal-1)/n)."""
    d = pl.DataFrame({"eom": eom, "y": y}).with_columns(
        (10 * (pl.col("y").rank("ordinal").over("eom") - 1) // pl.len().over("eom")).cast(pl.Int32).alias("l"))
    return d["l"].to_numpy()


def _rk_shuffle(eom, y, seed):
    """Permute y within each month (rows are sorted by eom, so blocks stay in place)."""
    order = np.lexsort((np.random.default_rng(seed).random(len(y)), eom.to_numpy()))
    return y[order]


def _rk_ranks(eom, score):
    """Within-month rank of score scaled to [-1, 1]."""
    d = pl.DataFrame({"eom": eom, "s": score}).with_columns(
        r=pl.col("s").rank().over("eom"), n=pl.len().over("eom"))
    return d.select(pl.when(pl.col("n") > 1).then((pl.col("r") - 1) / (pl.col("n") - 1) * 2 - 1).otherwise(0.0))["r"].to_numpy()


def _rk_mat(d, features):
    return d.select(features).cast(pl.Float32).to_numpy()  # nulls -> NaN (XGBoost handles them natively)


def fit_predict_year(df, features, win, cfg, model="xgb", shuffle=False):
    """Tune on val (train-only fits), refit on train+val, predict the test year. Returns (test_pred, val_pred, info)."""
    df = df.sort(["eom", "permno"])
    rng = lambda w: (pl.col("target_month") >= w[0]) & (pl.col("target_month") <= w[1])
    has_y = pl.col("ret_exc_lead1m").is_not_null()
    tr, va = df.filter(rng(win["train"]) & has_y), df.filter(rng(win["val"]) & has_y)
    te = df.filter(rng(win["test"]))
    va_all = df.filter(rng(win["val"]))  # incl. null-target rows: val_pred must cover every validation month-row
    ytr, yva = tr["ret_exc_lead1m"].to_numpy(), va["ret_exc_lead1m"].to_numpy()
    if shuffle:  # leakage test: destroy the label-feature link in train AND val
        ytr, yva = _rk_shuffle(tr["eom"], ytr, cfg["seed"]), _rk_shuffle(va["eom"], yva, cfg["seed"] + 1)
    Xtr, Xva, Xte = _rk_mat(tr, features), _rk_mat(va, features), _rk_mat(te, features)
    Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
    eall = pl.concat([tr["eom"], va["eom"]])
    kw = dict(device="cuda") if cfg["device"] == "cuda" and xgb.build_info().get("USE_CUDA") else {}

    def _xgb(depth, trees, X, y, e):
        m = xgb.XGBRanker(**cfg["xgb_fixed"], max_depth=depth, n_estimators=trees, random_state=cfg["seed"], **kw)
        m.fit(X, _rk_labels(e, y), qid=e.cast(pl.Int32).to_numpy())
        return m

    grid, best, val_score, best_m = {}, (-np.inf, None, None), None, None
    if model == "xgb":
        for depth in cfg["xgb_depth_grid"]:
            m = _xgb(depth, cfg["xgb_max_trees"], Xtr, ytr, tr["eom"])
            for k in range(cfg["xgb_eval_every"], cfg["xgb_max_trees"] + 1, cfg["xgb_eval_every"]):
                s = m.predict(Xva, iteration_range=(0, k))
                ic = _rk_mean_ic(va["eom"], s, yva)
                grid[f"{depth}_{k}"] = ic
                if ic > best[0]:
                    best, val_score, best_m = (ic, depth, k), s, m
        final = _xgb(best[1], best[2], Xall, yall, eall)
        te_score = final.predict(Xte)
        va_all_score = best_m.predict(_rk_mat(va_all, features), iteration_range=(0, best[2]))
    else:  # ridge baseline on the features (template: Ridge on the 147 characteristics)
        Xtr0, Xva0, Xte0, Xall0 = (np.nan_to_num(x) for x in (Xtr, Xva, Xte, Xall))
        for a in cfg["ridge_alpha_grid"]:
            rm = Ridge(alpha=a).fit(Xtr0, ytr)
            s = rm.predict(Xva0)
            ic = _rk_mean_ic(va["eom"], s, yva)
            grid[str(a)] = ic
            if ic > best[0]:
                best, val_score, best_m = (ic, a, None), s, rm
        te_score = Ridge(alpha=best[1]).fit(Xall0, yall).predict(Xte0)
        va_all_score = best_m.predict(np.nan_to_num(_rk_mat(va_all, features)))

    # ret_hat: linear regression of realised return on the within-month score rank, fitted on val scores of the train-only model
    slope, icpt = np.polyfit(_rk_ranks(va["eom"], val_score), yva, 1)
    out = lambda d, s: d.select("permno", "eom").with_columns(
        score=pl.Series(s, dtype=pl.Float64),
        ret_hat=pl.Series(icpt + slope * _rk_ranks(d["eom"], s), dtype=pl.Float64))
    info = dict(year=win["year"], model=model, best_depth=best[1] if model == "xgb" else None,
                best_trees=best[2], val_ic=best[0], grid=grid,
                train_max_target=max(tr["target_month"].max(), va["target_month"].max()))  # step 11 check
    if model == "ridge":
        info["best_alpha"] = best[1]
    return out(te, te_score), out(va_all, va_all_score), info


def run_schedule(df, features, cfg, model="xgb", shuffle=False):
    """Loop over test years. Returns (preds, val_2021, infos)."""
    tests, infos, val_2021 = [], [], None
    for w in windows(cfg):
        t, v, i = fit_predict_year(df, features, w, cfg, model, shuffle)
        tests.append(t); infos.append(i)
        if w["year"] == 2021:
            val_2021 = v
    return pl.concat(tests), val_2021, infos
