"""
src/cfi.py - Clustered Feature Importance helpers (port of teammate's experience-cfi-xgboost src/fiam/cfi.py).
Factors are clustered on TRAIN months (median monthly Spearman, d = 1-|rho|, complete linkage); groups are permuted jointly
within each VALIDATION month; top-ranked groups form candidate factor subsets. Model tuning lives in src/ranker.py.
"""
import math
import warnings

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata


def _cfi_month_corr(X):
    """Spearman matrix of one month (ranked columns used as is, nulls already 0) and its stock count."""
    X = np.nan_to_num(X.astype("float64"))
    if len(X) < 3:
        return np.full((X.shape[1],) * 2, np.nan), len(X)
    with np.errstate(all="ignore"):
        return np.corrcoef(rankdata(X, axis=0).T), len(X)


def cfi_clusters(train_df, factors, cfg):
    """{cut: [groups]} - groups are lists of factor names. Under-documented pairs -> their factors stay singletons (last)."""
    p = len(factors)
    rhos, ns = [], []
    for _, d in train_df.group_by("eom", maintain_order=True):
        r, n = _cfi_month_corr(d.select(factors).to_numpy())
        rhos.append(r)
        ns.append(n)
    if rhos:
        rho = np.stack(rhos)
        valid = (np.array(ns)[:, None, None] >= cfg["cfi_min_pairs"]) & np.isfinite(rho)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(valid, rho, np.nan), axis=0)
        ok = valid.sum(axis=0) >= cfg["cfi_min_months"]
    else:
        med, ok = np.full((p, p), np.nan), np.zeros((p, p), bool)
    np.fill_diagonal(ok, True)
    bad = ~ok | ~np.isfinite(med)
    np.fill_diagonal(bad, False)
    iso = [i for i in range(p) if bad[i].any()]
    keep = [i for i in range(p) if i not in iso]
    Z = None
    if len(keep) > 1:
        D = 1.0 - np.abs(med[np.ix_(keep, keep)])
        D = np.clip((D + D.T) / 2.0, 0.0, None)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="complete")
    out = {}
    for cut in cfg["cfi_cuts"]:
        labels = fcluster(Z, t=1.0 - cut, criterion="distance") if Z is not None else np.ones(len(keep))
        by = {}
        for i, lab in zip(keep, labels):
            by.setdefault(lab, []).append(i)
        groups = sorted(by.values(), key=lambda g: g[0]) + [[i] for i in iso]
        out[cut] = [[factors[i] for i in g] for g in groups]
    return out


def _cfi_mean_ic(eom, score, y):
    d = pl.DataFrame({"eom": eom, "s": score, "y": y}).drop_nulls("y").group_by("eom").agg(
        pl.corr(pl.col("s").rank(), pl.col("y").rank()).alias("ic"))["ic"].drop_nans().drop_nulls()
    return float(d.mean()) if len(d) else float("nan")


def _cfi_perm(eom, seed):
    """Row permutation that stays inside each month (months visited in sorted order, so (seed, month) is reproducible)."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(eom))
    order = np.argsort(eom, kind="stable")
    se = eom[order]
    for pos in np.split(order, np.flatnonzero(se[1:] != se[:-1]) + 1):
        perm[pos] = rng.permutation(pos)
    return perm


def cfi_importance(predict_fn, val_df, features, groups, cfg):
    """[(group_index, importance)]; importance = base mean monthly rank IC - mean over seeds of permuted IC."""
    X = val_df.select(features).cast(pl.Float32).to_numpy()
    eom, y = val_df["eom"], val_df["ret_exc_lead1m"]
    base = _cfi_mean_ic(eom, predict_fn(X), y)
    eom_np = eom.to_numpy()
    perms = [_cfi_perm(eom_np, cfg["seed"] + r) for r in range(cfg["cfi_repeats"])]
    idx = {f: i for i, f in enumerate(features)}
    out = []
    for gi, g in enumerate(groups):
        cols = [idx[f] for f in g]
        ics = []
        for perm in perms:
            Xp = X.copy()
            Xp[:, cols] = X[perm][:, cols]  # one permutation for every column of the group
            ics.append(_cfi_mean_ic(eom, predict_fn(Xp), y))
        out.append((gi, base - float(np.mean(ics))))
    return out


def cfi_candidates(groups, importances, shares):
    """Factor subsets: top ceil(share*n_groups) groups by importance (ties -> group order); duplicates dropped."""
    ranked = sorted(importances, key=lambda t: (-t[1], t[0]))
    seen, out = set(), []
    for s in shares:
        k = max(1, math.ceil(s * len(groups) - 1e-9))
        sub = [f for gi, _ in ranked[:k] for f in groups[gi]]
        if tuple(sub) not in seen:
            seen.add(tuple(sub))
            out.append(sub)
    return out
