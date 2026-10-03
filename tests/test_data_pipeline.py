"""Fast synthetic tests for src/data.py."""
import datetime as dt
import numpy as np
import polars as pl
from src.config import CONFIG
from src.data import build_universe, preprocess

CFG = dict(CONFIG)


def _panel():
    rng = np.random.default_rng(0)
    eoms = [dt.date(2018, 10, 31), dt.date(2018, 11, 30), dt.date(2018, 12, 31), dt.date(2019, 1, 31)]
    rows = []
    for e in eoms:
        for k in range(50):
            rows.append(dict(permno=k, eom=e, prc=(-1 if k % 7 == 0 else 1) * (3.0 if k == 1 else 20.0 + k),
                             market_equity=float(k + 1), dolvol_126d=float(rng.random()),
                             zero_trades_126d=float(rng.random()),
                             x1=float(rng.random()) if k % 2 else None, x2=float(k % 3), x3=1.0,
                             gics="45103010", ret_exc_lead1m=0.01))
    df = pl.DataFrame(rows).with_columns(pl.col("eom").cast(pl.Date))
    return df.with_columns(target_month=pl.col("eom").dt.offset_by("1mo").dt.month_end())


def test_universe_and_raw_copies():
    u = build_universe(_panel(), CFG)
    e = u.filter(pl.col("eom") == dt.date(2018, 11, 30))
    assert 1 not in e["permno"].to_list()  # price below 5
    assert e["market_equity"].min() > 10.0  # above the 20th pct of all 50 stocks
    assert e["me_pct"].min() > 0.2 and (e["prc_raw"] == e["prc"]).all()
    assert e["short_eligible"].dtype == pl.Boolean
    assert (e.filter(pl.col("short_eligible"))["me_pct"] >= CFG["short_me_q"]).all()


def test_preprocess_rank_flags():
    u = build_universe(_panel(), CFG)
    df, flags = preprocess(u, ["x1", "x2", "x3"], CFG)
    assert flags == ["x1_miss"]  # x1 is ~50% null on training rows
    assert df["x1_miss"].sum() == u["x1"].is_null().sum()
    assert df["x1"].is_null().sum() == 0 and df["x1"].min() >= -1 and df["x1"].max() <= 1
    assert (df["x3"] == 0).all()  # constant -> max rank 0 -> 0
    assert df["x2"].max() == 1.0 and df["x2"].min() == -1.0  # dense rank: 3 levels -> -1, 0, 1
    assert (df["me_raw"] == u["me_raw"]).all() and df["sector"][0] == "45"
    assert (df.filter(pl.col("x1_miss") == 1)["x1"] == 0).all()
