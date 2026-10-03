"""Unit tests of the step-11 leakage-check helpers in src/pipeline.py on synthetic data."""
from datetime import date

import polars as pl

from src.pipeline import (_pipe_leak_filings, _pipe_leak_windows, _pipe_leak_target, _pipe_leak_dups,
                          _pipe_ir, _pipe_jsonable, _pipe_leak_target_months)

EOM = date(2020, 1, 31)


def test_filings_ok_and_bad():
    filings = pl.DataFrame({"permno": [1, 2], "filing_date": [date(2020, 1, 15), date(2020, 1, 31)]})
    ok = pl.DataFrame({"permno": [1, 2], "eom": [EOM, EOM], "has_filing": [1, 1]})
    assert _pipe_leak_filings(filings, ok) == {"flagged_without_filing": 0, "filing_after_eom": 0}
    bad = pl.DataFrame({"permno": [1, 3], "eom": [date(2020, 2, 29), EOM], "has_filing": [1, 1]})
    r = _pipe_leak_filings(filings, bad)
    assert r["flagged_without_filing"] == 2


def test_windows():
    d = lambda y, m: date(y, m, 28)
    good = [{"year": 2021, "train": (d(2015, 2), d(2018, 12)), "val": (d(2019, 1), d(2020, 12)), "test": (d(2021, 1), d(2021, 12))}]
    assert _pipe_leak_windows(good) == 0
    overlap = [{"year": 2021, "train": (d(2015, 2), d(2019, 6)), "val": (d(2019, 1), d(2020, 12)), "test": (d(2021, 1), d(2021, 12))}]
    assert _pipe_leak_windows(overlap) > 0
    late = [{"year": 2021, "train": (d(2015, 2), d(2018, 12)), "val": (d(2019, 1), d(2021, 3)), "test": (d(2021, 1), d(2021, 12))}]
    assert _pipe_leak_windows(late) > 0


def test_target_columns():
    df = pl.DataFrame({"ret_exc_lead1m": [0.1, -0.2, 0.3, 0.05], "a": [1.0, 2.0, 3.0, 4.0],
                       "b": [0.1, -0.2, 0.3, 0.05], "c": [2.0, 1.0, 4.0, 3.0], "s": ["x", "y", "z", "w"]})
    assert _pipe_leak_target(df, ["a", "c"]) == []
    assert _pipe_leak_target(df, ["a", "b", "c"]) == ["b"]
    assert "ret_exc_lead1m" in _pipe_leak_target(df, ["ret_exc_lead1m", "a"])


def test_duplicates_and_helpers():
    df = pl.DataFrame({"permno": [1, 1, 2], "eom": [EOM, EOM, EOM]})
    assert _pipe_leak_dups(df) == 1
    assert _pipe_leak_dups(df.unique()) == 0
    r = pl.DataFrame({"active": [0.01, 0.02, 0.0, 0.03]})
    assert _pipe_ir(r) > 0
    assert _pipe_jsonable(EOM) == "2020-01-31"


def test_leak_target_months():
    ok = [dict(year=2021, train_max_target=date(2020, 12, 31))]
    assert _pipe_leak_target_months(ok) == []
    assert len(_pipe_leak_target_months([dict(year=2021, train_max_target=date(2021, 1, 31))])) == 1
