"""Fast synthetic tests for src/cfi.py and the CFI path of src/ranker.py."""
import calendar
from datetime import date

import numpy as np
import polars as pl

from src.cfi import _cfi_perm, cfi_candidates, cfi_clusters, cfi_importance
from src.config import CONFIG
from src.ranker import fit_predict_year, windows

CFG = {**CONFIG, "ranker_model": "xgb", "device": "cpu", "xgb_max_trees": 60, "xgb_depth_grid": [3], "test_years": [2021],
       "cfi_min_pairs": 20, "cfi_min_months": 3, "cfi_cuts": [0.75], "cfi_shares": [0.34, 0.67], "cfi_repeats": 2,
       "xgb_fixed": {**CONFIG["xgb_fixed"], "min_child_weight": 1, "learning_rate": 0.1}}
FACS = ["f0", "f1", "f2", "f3", "f4", "f5"]  # f1 == f0 (+tiny noise); f0 carries the signal
FEATS = FACS + ["f0_miss"]


def _me(y, m):
    return date(y, m, calendar.monthrange(y, m)[1])


def _panel(n=60, seed=0, train_corr_break=False):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2015, 2022):
        for m in range(1, 13):
            if (y, m) < (2015, 2):
                continue
            f = rng.normal(size=(n, 6))
            f[:, 1] = f[:, 0] + 0.01 * rng.normal(size=n)
            if train_corr_break and y >= 2019:  # val/test months: f4 becomes a copy of f2
                f[:, 4] = f[:, 2]
            r = 0.1 * f[:, 0] + 0.05 * rng.normal(size=n)
            e = _me(y - 1, 12) if m == 1 else _me(y, m - 1)
            rows.append(pl.DataFrame({"permno": np.arange(n), "eom": [e] * n, "target_month": [_me(y, m)] * n,
                                      "ret_exc_lead1m": r, "f0_miss": np.zeros(n),
                                      **{k: f[:, i] for i, k in enumerate(FACS)}}))
    return pl.concat(rows)


def test_clusters_group_correlated_and_use_train_only():
    df, w = _panel(), windows(CFG)[0]
    tr = df.filter((pl.col("target_month") >= w["train"][0]) & (pl.col("target_month") <= w["train"][1]))
    g = cfi_clusters(tr, FACS, CFG)[0.75]
    assert ["f0", "f1"] in g and sum(len(x) for x in g) == 6 and len(g) == 5
    # later months (val/test) make f4 == f2, but clusters fit on train rows only must not see it
    g2 = cfi_clusters(_panel(train_corr_break=True).filter(
        (pl.col("target_month") <= w["train"][1])), FACS, CFG)[0.75]
    assert g2 == g


def test_underdocumented_pairs_stay_singletons():
    df = _panel().filter(pl.col("eom") < date(2015, 3, 1))  # 2 months < cfi_min_months
    g = cfi_clusters(df, FACS, CFG)[0.75]
    assert g == [[f] for f in FACS]


def test_permutation_within_months():
    eom = np.repeat(np.array(["2020-01", "2020-02", "2020-03"]), 10)
    p = _cfi_perm(eom, 1)
    assert (eom[p] == eom).all() and not (p == np.arange(30)).all()
    assert (p == _cfi_perm(eom, 1)).all() and sorted(p) == list(range(30))


def test_importance_signal_group_is_top():
    df, w = _panel(), windows(CFG)[0]
    rng = lambda x: (pl.col("target_month") >= x[0]) & (pl.col("target_month") <= x[1])
    tr, va = df.filter(rng(w["train"])), df.filter(rng(w["val"]))
    import xgboost as xgb
    from src.ranker import _rk_labels, _rk_mat
    m = xgb.XGBRanker(**CFG["xgb_fixed"], max_depth=3, n_estimators=60, random_state=0)
    m.fit(_rk_mat(tr, FEATS), _rk_labels(tr["eom"], tr["ret_exc_lead1m"].to_numpy()), qid=tr["eom"].cast(pl.Int32).to_numpy())
    groups = [["f0", "f1"], ["f2"], ["f3"], ["f4"], ["f5"]]
    imp = cfi_importance(m.predict, va, FEATS, groups, CFG)
    assert imp[0][1] > 0.05 and all(imp[0][1] > 5 * abs(v) for _, v in imp[1:])
    assert cfi_candidates(groups, imp, [0.2, 0.4])[0] == ["f0", "f1"]
    assert len(cfi_candidates(groups, imp, [0.2, 0.2])) == 1


def test_fit_predict_year_cfi_picks_signal_and_ignores_test():
    df, w = _panel(), windows(CFG)[0]
    kw = dict(win=w, cfg=CFG)
    t1, v1, i1 = fit_predict_year(df, FEATS, cfi_factors=FACS, **kw)
    c = i1["cfi"]
    assert c["candidates"] and "f0" in i1["features"] and "f0_miss" in i1["features"]
    assert len(i1["features"]) <= len(FEATS)
    # destroying test-year rows (features and targets) must not change the selection or val predictions
    junk = df.with_columns([pl.when(pl.col("target_month") >= w["test"][0]).then(pl.col(f).reverse()).otherwise(pl.col(f)).alias(f)
                            for f in FACS])
    t2, v2, i2 = fit_predict_year(junk, FEATS, cfi_factors=FACS, **kw)
    assert i2["cfi"] == c and i2["features"] == i1["features"] and v1["score"].to_list() == v2["score"].to_list()
    # default (no cfi_factors) leaves info without cfi
    assert "cfi" not in fit_predict_year(df, FEATS, **kw)[2]
