import numpy as np
import pandas as pd

FEATURES = ["mom_5", "mom_20", "mom_60", "vol_20", "volchg_20"]
GAP = 5  # 训练集和测试集之间隔开5天,因为目标是未来5天收益,避免训练标签和测试期重叠


def make_splits(dates, train_years=5, test_months=12, gap=GAP):
    """生成滚动窗口(扩展式):训练集永远在测试集之前。
    返回 (训练日期列表, 测试日期列表) 的列表。"""
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    first = dates[0]
    test_start = first + pd.DateOffset(years=train_years)
    splits = []
    while test_start < dates[-1]:
        test_end = test_start + pd.DateOffset(months=test_months)
        test = dates[(dates >= test_start) & (dates < test_end)]
        if len(test) == 0:
            break
        train_all = dates[dates < test_start]
        train = train_all[:-gap] if gap > 0 else train_all
        splits.append((train, test))
        test_start = test_end
    return splits


def run_walkforward(data, predict_fn, splits):
    """predict_fn(train_df, test_df) 返回测试集每行的预测值。
    返回带 pred 列的测试期数据(所有窗口拼接)。"""
    data = data.dropna(subset=FEATURES + ["target"])
    outs = []
    for train_dates, test_dates in splits:
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)].copy()
        test["pred"] = predict_fn(train, test)
        outs.append(test)
    return pd.concat(outs, ignore_index=True)
