"""Fast synthetic tests for src/gru.py."""
import datetime as dt
import numpy as np
import polars as pl
from src.config import CONFIG
from src.gru import gru_embeddings

CFG = dict(CONFIG, gru_max_epochs=3, gru_patience=2, gru_batch=256, device="cpu", gru_hidden=16)


def _months():
    out = []
    for y in range(2017, 2021):
        for mo in range(1, 13):
            nxt = dt.date(y + (mo == 12), mo % 12 + 1, 1)
            out.append(nxt - dt.timedelta(days=1))
    return out


def _df(n=40):
    rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        for i, e in enumerate(_months()):
            if k == 0 and i % 2 == 1:
                continue  # permno 0 has alternate months missing -> no calendar-consecutive changes
            rows.append((k, e, *rng.uniform(-1, 1, 3)))
    return pl.DataFrame(rows, schema=["permno", "eom", "a", "b", "c"], orient="row")


def test_gru_shapes_nan_determinism():
    df = _df()
    emb, info = gru_embeddings(df, ["a", "b", "c"], 4, CFG)
    assert emb.shape == (len(df), 6) and emb.columns[2] == "gru_1"
    assert emb.filter(pl.col("eom") == dt.date(2017, 3, 31))["gru_1"].is_nan().all()  # < 6 valid changes
    late = emb.filter((pl.col("eom") == dt.date(2020, 12, 31)) & (pl.col("permno") > 0))
    assert late["gru_1"].is_nan().sum() == 0
    gappy = emb.filter(pl.col("permno") == 0)
    assert gappy["gru_1"].is_nan().all()
    assert info["best_wd"] in CFG["gru_wd_grid"] and set(info["wd_losses"]) == set(CFG["gru_wd_grid"])
    emb2, _ = gru_embeddings(df, ["a", "b", "c"], 4, CFG)
    assert np.allclose(emb["gru_1"].fill_nan(0).to_numpy(), emb2["gru_1"].fill_nan(0).to_numpy())
