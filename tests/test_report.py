"""Tests for src/report.py on small synthetic data."""
import datetime as dt
import json
import numpy as np
import polars as pl
from src.metrics import book_returns
from src.report import performance_pack, write_submission

N = 14
def _me(y, m):
    return dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)


EOMS = [_me(2021 + (m // 12), m % 12 + 1) for m in range(N)]  # month-ends from 2021-01-31


def _data(tmp_path):
    rng = np.random.default_rng(1)
    pan = []
    for e in EOMS:
        for p in range(1, 9):
            pan.append((p, e, rng.normal(0, 0.05), f"T{p}" if p != 3 else None, f"Firm {p}" if p != 3 else None,
                        "micro" if p % 2 else "large", 1e9 * p, 1e6 * p))
    panel = pl.DataFrame(pan, schema=["permno", "eom", "ret_exc_lead1m", "ticker", "company_name", "size_grp", "me_raw",
                                      "dolvol_raw"], orient="row")
    w = pl.DataFrame([(p, e, 0.25 if p <= 4 else -0.25) for e in EOMS for p in range(1, 9)],
                     schema=["permno", "eom", "weight"], orient="row")
    filings = pl.DataFrame({"permno": [3, 3], "filing_date": [dt.date(2021, 3, 5), dt.date(2022, 1, 5)],
                            "ticker": ["OLD3", "NEW3"], "company_name": ["Old Three", "New Three"]})
    market = pl.DataFrame({"eom": [dt.date(2020, 12, 31)] + [_me(e.year, e.month + 1) if e.month < 12 else _me(e.year + 1, 1) for e in EOMS],
                           "tb3ms": 0.5, "sp500_ret": rng.normal(0.01, 0.04, N + 1)})
    cfg = {"output_dir": tmp_path, "premium_annual": 0.04, "nw_lags": 3}
    preds = panel.select("permno", "eom", "ret_exc_lead1m").with_columns(score=pl.col("ret_exc_lead1m"),
                                                                           ret_hat=pl.col("ret_exc_lead1m") * 0.1)
    rets, _ = book_returns(w, panel, market, cfg)
    return w, panel, filings, rets, preds, cfg


def test_submission_files(tmp_path):
    w, panel, filings, rets, _, cfg = _data(tmp_path)
    res = write_submission(w, panel, filings, rets, cfg)
    h = pl.read_csv(tmp_path / "holdings.csv")
    assert h.columns == ["Date", "PERMNO", "TICKER", "COMPANY NAME", "WEIGHT"]
    assert h["Date"][0] == "2021-02-01" and h.height == w.height
    assert set(h["WEIGHT"].unique().to_list()) == {25.0, -25.0}  # % of NAV, signed
    assert "%" not in (tmp_path / "holdings.csv").read_text()
    # permno 3 has no label in the panel: nearest 8-K filing wins (Jan-2021 eom -> Mar-2021 filing)
    r3 = h.filter(pl.col("PERMNO") == 3)
    assert r3["TICKER"][0] == "OLD3" and r3["TICKER"][-1] == "NEW3" and res["unlabeled_permnos"] == []
    r = pl.read_csv(tmp_path / "returns.csv")
    assert r.columns == ["Date", "total", "excess", "active", "long_leg", "short_leg", "benchmark", "sp500"]


def test_unlabeled_logged(tmp_path):
    w, panel, filings, rets, _, cfg = _data(tmp_path)
    assert write_submission(w, panel, None, rets, cfg)["unlabeled_permnos"] == [3]


def test_performance_pack(tmp_path):
    w, panel, filings, rets, preds, cfg = _data(tmp_path)
    s = performance_pack(rets, w, panel, filings, preds, cfg)
    for f in ["cumulative_returns", "underwater", "rolling_active_return", "rolling_ir", "rolling_beta",
              "return_histogram", "contributors"]:
        assert (tmp_path / "figures" / f"{f}.png").stat().st_size > 1000
    j = json.loads((tmp_path / "performance.json").read_text())
    assert j["book"]["largest_weight"] == 0.25 and abs(j["book"]["top10_share_of_gross"] - 1.0) < 1e-9
    assert abs(j["book"]["short_small_cap_share"] - 0.5) < 1e-9  # shorts are permnos 5..8: 5, 7 micro
    assert 0 < j["oos_r2"] < 1 and "gross of trading costs" in " ".join(j["notes"])
    assert j["top_contributors"][0]["label"].startswith(("T", "?", "OLD3", "NEW3")) and ", " in j["top_contributors"][0]["label"]
    assert s["n_missing_returns"] == 0
