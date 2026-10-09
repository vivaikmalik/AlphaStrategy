"""Fast synthetic tests for src/ranker.py (no real data)."""
import calendar
from datetime import date

import numpy as np
import polars as pl
import pytest

from src.config import CONFIG
from src.ranker import _rk_labels, fit_predict_year, rank_ic, run_schedule, windows

CFG = {**CONFIG, "ranker_model": "xgb", "device": "cpu", "xgb_max_trees": 100, "xgb_depth_grid": [3, 4], "test_years": [2021],
       "xgb_fixed": {**CONFIG["xgb_fixed"], "min_child_weight": 1, "learning_rate": 0.1}}
FEATS = ["f0", "f1", "f2", "f3"]


def _me(y, m):
    return date(y, m, calendar.monthrange(y, m)[1])


def _panel(signal=1.0, n=60, seed=0, last=(2021, 12)):
    """target months 2015-02..last; eom = target month - 1; planted signal in f0; last test month has null target."""
    rng = np.random.default_rng(seed)
    tms = [(y, m) for y in range(2015, 2022) for m in range(1, 13) if (2015, 2) <= (y, m) <= last]
    rows = []
    for y, m in tms:
        f = rng.normal(size=(n, 4))
        r = signal * 0.05 * f[:, 0] + 0.05 * rng.normal(size=n)
        e = _me(y - 1, 12) if m == 1 else _me(y, m - 1)
        rows.append(pl.DataFrame({"permno": np.arange(n), "eom": [e] * n, "target_month": [_me(y, m)] * n,
                                  "ret_exc_lead1m": r, **{k: f[:, i] for i, k in enumerate(FEATS)}}))
    df = pl.concat(rows)
    return df.with_columns(pl.when(pl.col("target_month") == _me(*last)).then(None).otherwise(pl.col("ret_exc_lead1m"))
                           .alias("ret_exc_lead1m"))


def _join(pred, df):
    return pred.join(df.select("permno", "eom", "ret_exc_lead1m"), on=["permno", "eom"])


def test_windows_match_schedule():
    ws = {w["year"]: w for w in windows({**CONFIG, "test_years": [2021, 2026]})}
    assert ws[2021]["train"] == (date(2015, 2, 28), date(2018, 12, 31))
    assert ws[2021]["val"] == (date(2019, 1, 31), date(2020, 12, 31))
    assert ws[2021]["test"] == (date(2021, 1, 31), date(2021, 12, 31))
    assert ws[2026]["train"] == (date(2015, 2, 28), date(2023, 12, 31))
    assert ws[2026]["val"] == (date(2024, 1, 31), date(2025, 12, 31))
    assert ws[2026]["test"] == (date(2026, 1, 31), date(2026, 8, 31))


def test_labels_are_month_deciles():
    df = _panel().sort("eom")
    lab = _rk_labels(df["eom"], df["ret_exc_lead1m"].fill_null(0.0))
    d = df.with_columns(l=pl.Series(lab))
    assert set(lab) == set(range(10))
    assert d.group_by("eom").agg(pl.col("l").min().alias("a"), pl.col("l").max().alias("b")).filter(
        (pl.col("a") != 0) | (pl.col("b") != 9)).is_empty()
    one = d.filter(pl.col("eom") == d["eom"][0]).sort("ret_exc_lead1m")["l"].to_list()
    assert one == sorted(one)


@pytest.fixture(scope="module")
def planted():
    df = _panel()
    return df, fit_predict_year(df, FEATS, windows(CFG)[0], CFG)


def test_signal_gives_positive_ic_and_trees_multiple_of_50(planted):
    df, (te, va, info) = planted
    assert info["val_ic"] > 0.2 and info["best_trees"] % 50 == 0 and info["best_depth"] in (3, 4)
    assert rank_ic(_join(te, df))["ic"].mean() > 0.2
    assert set(te.columns) == {"permno", "eom", "score", "ret_hat"}


def test_ret_hat_from_val_scores_and_null_target_rows_predicted(planted):
    df, (te, va, info) = planted
    assert te["eom"].max() == _me(2021, 11)  # target month 2021-12 has null target but is still scored
    assert te["score"].null_count() == 0 and te["ret_hat"].null_count() == 0
    # ret_hat is an affine function of the within-month score rank, with the slope fitted on val -> positive here
    t = te.with_columns(r=pl.col("score").rank().over("eom"))
    assert t.group_by("eom").agg(pl.corr("r", "ret_hat").alias("c"))["c"].min() > 0.99
    assert _join(va, df).drop_nulls()["ret_hat"].std() > 0 and va["eom"].min() >= _me(2018, 12)


def test_shuffle_gives_zero_ic(planted):
    df, _ = planted
    te, _, _ = fit_predict_year(df, FEATS, windows(CFG)[0], CFG, shuffle=True)
    ic0 = rank_ic(_join(planted[1][0], df))["ic"].mean()
    assert abs(rank_ic(_join(te, df))["ic"].mean()) < min(0.2, ic0 / 2)  # noise level (11 test months, sd ~0.05-0.1)


def test_ridge_path_and_run_schedule():
    df = _panel()
    te, va, infos = run_schedule(df, FEATS, {**CFG, "ridge_alpha_grid": [1, 100]}, model="ridge")
    assert infos[0]["model"] == "ridge" and infos[0]["best_alpha"] in (1, 100) and infos[0]["val_ic"] > 0.2
    assert rank_ic(_join(te, df))["ic"].mean() > 0.2 and va["eom"].min() >= _me(2018, 12)


def test_val_pred_includes_null_target_rows():
    df = _panel()
    w = windows(CFG)[0]
    df = df.with_columns(pl.when((pl.col("target_month") == w["val"][1]) & (pl.col("permno") < 10)).then(None)
                         .otherwise(pl.col("ret_exc_lead1m")).alias("ret_exc_lead1m"))
    _, va, _ = fit_predict_year(df, FEATS, w, CFG)
    n = df.filter((pl.col("target_month") >= w["val"][0]) & (pl.col("target_month") <= w["val"][1])).height
    assert va.height == n and va["ret_hat"].null_count() == 0 and va["score"].null_count() == 0
