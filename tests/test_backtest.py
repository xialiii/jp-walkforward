import numpy as np
import pandas as pd

from src.backtest import metrics, run_backtest


def test_max_drawdown():
    m = metrics(pd.Series([0.1, -0.5, 0.2]))
    assert np.isclose(m["max_drawdown"], -0.5)


def test_cost_only_when_portfolio_changes():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    rows = []
    for d in dates:
        rows += [
            {"date": d, "ticker": "A", "pred": 1.0, "target": 0.02},
            {"date": d, "ticker": "B", "pred": 0.0, "target": -0.01},
        ]
    out = run_backtest(pd.DataFrame(rows), top_n=1, cost=0.001, hold_days=1)
    assert np.isclose(out["cost"].iloc[0], 0.001)
    assert out["cost"].iloc[1] == 0
    assert np.isclose(out["net"].iloc[0], 0.02 - 0.001)
