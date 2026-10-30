import pandas as pd

from src.clean_data import remove_known_bad


def test_remove_known_bad_only_touches_listed_rows():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-27", "2026-03-30", "2026-03-31", "2026-04-01"] * 2),
        "ticker": ["1306.T"] * 4 + ["7203.T"] * 4,
        "Close": [1.0] * 8,
    })
    rules = [("1306.T", "2026-03-30", "2026-03-31", "test")]
    out, n = remove_known_bad(df, rules)
    assert n == 2
    assert len(out[out["ticker"] == "1306.T"]) == 2
    assert len(out[out["ticker"] == "7203.T"]) == 4
