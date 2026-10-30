import pandas as pd

from src.walkforward import GAP, make_splits


def test_train_always_before_test_with_gap():
    dates = pd.Series(pd.bdate_range("2016-01-01", "2026-09-30"))
    splits = make_splits(dates)
    assert len(splits) > 0
    for train, test in splits:
        assert train.max() < test.min()
        n_between = ((dates >= train.max()) & (dates < test.min())).sum()
        assert n_between > GAP


def test_test_windows_do_not_overlap():
    dates = pd.Series(pd.bdate_range("2016-01-01", "2026-09-30"))
    splits = make_splits(dates)
    for (_, a), (_, b) in zip(splits[:-1], splits[1:]):
        assert a.max() < b.min()
