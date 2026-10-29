import numpy as np
import pandas as pd

from src.clean_data import clean_prices
from src.features import HORIZON, add_features

FEATURE_COLS = ["mom_5", "mom_20", "mom_60", "vol_20", "volchg_20"]


def make_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1_000, 10_000, n).astype(float),
        "ticker": "TEST.T",
    })


def scramble_after(df, cut, seed=1):
    """把第cut行之后的价格和成交量全部随机打乱,用来模拟'未来发生了别的事'。"""
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = len(out) - cut - 1
    for c in ["Open", "High", "Low", "Close"]:
        out.loc[cut + 1:, c] = out.loc[cut + 1:, c].values * rng.uniform(0.5, 1.5, m)
    out.loc[cut + 1:, "Volume"] = rng.integers(1_000, 10_000, m).astype(float)
    return out


def test_features_do_not_use_future_data():
    base = make_prices()
    cut = 200
    a = add_features(base)
    b = add_features(scramble_after(base, cut))
    pd.testing.assert_frame_equal(a.loc[:cut, FEATURE_COLS], b.loc[:cut, FEATURE_COLS])


def test_test_is_sensitive_to_future_changes():
    """对照实验:目标本来就依赖未来,所以打乱未来后目标必须变化。
    这证明上面的测试确实有能力发现问题。"""
    base = make_prices()
    cut = 200
    a = add_features(base)
    b = add_features(scramble_after(base, cut))
    rows = slice(cut - HORIZON + 1, cut)
    assert not np.allclose(a.loc[rows, "target"], b.loc[rows, "target"])


def test_target_is_forward_return():
    base = make_prices()
    a = add_features(base)
    i = 50
    expected = base["Close"].iloc[i + HORIZON] / base["Close"].iloc[i] - 1
    assert np.isclose(a["target"].iloc[i], expected)


def test_clean_prices_removes_holiday_filler_rows():
    df = make_prices(5)
    df.loc[2, ["Open", "High", "Low", "Close"]] = 100.0
    df.loc[2, "Volume"] = 0.0
    cleaned, n_removed = clean_prices(df)
    assert n_removed == 1
    assert len(cleaned) == 4
