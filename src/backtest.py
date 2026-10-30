import numpy as np
import pandas as pd

HOLD_DAYS = 5
PERIODS_PER_YEAR = 252 / HOLD_DAYS
TOP_N = 5
COST_ONE_WAY = 0.0010  # 单边10个基点


def run_backtest(pred, top_n=TOP_N, cost=COST_ONE_WAY, hold_days=HOLD_DAYS):
    """pred需要有 date, ticker, pred, target 四列。
    每hold_days个交易日换仓一次:买入预测值最高的top_n只,等权持有。
    target是从当天收盘起未来hold_days天的收益。"""
    dates = np.sort(pred["date"].unique())
    prev_w = pd.Series(dtype=float)
    rows = []
    for d in dates[::hold_days]:
        day = pred[pred["date"] == d].dropna(subset=["pred", "target"])
        if len(day) < top_n:
            continue
        chosen = day.nlargest(top_n, "pred")
        w = pd.Series(1.0 / top_n, index=chosen["ticker"].values)
        names = prev_w.index.union(w.index)
        traded = (w.reindex(names, fill_value=0.0)
                  - prev_w.reindex(names, fill_value=0.0)).abs().sum()
        gross = chosen["target"].mean()
        c = traded * cost
        rows.append({
            "date": d, "gross": gross, "cost": c, "net": gross - c,
            "turnover": traded, "ew": day["target"].mean(),
        })
        prev_w = w
    return pd.DataFrame(rows).set_index("date")


def metrics(r):
    """r是每个持有期的收益序列。返回年化收益、年化波动、夏普比率、最大回撤。"""
    r = pd.Series(r).dropna().astype(float)
    n = len(r)
    cum = np.concatenate([[1.0], (1 + r).cumprod().values])
    peak = np.maximum.accumulate(cum)
    return {
        "ann_return": cum[-1] ** (PERIODS_PER_YEAR / n) - 1,
        "ann_vol": r.std() * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR),
        "max_drawdown": (cum / peak - 1).min(),
        "n_periods": n,
    }
