import numpy as np
import pandas as pd

HORIZON = 5  # 预测未来5个交易日的收益


def add_features(g):
    """对单只股票按日期排序后的数据计算特征。
    每个特征在第t天只使用第t天及以前的数据。"""
    g = g.sort_values("date").copy()
    close = g["Close"]
    ret1 = close.pct_change()  # 当天相对前一天的收益

    # 动量:过去N天累计收益
    g["mom_5"] = close.pct_change(5)
    g["mom_20"] = close.pct_change(20)
    g["mom_60"] = close.pct_change(60)

    # 波动率:过去20天日收益的标准差
    g["vol_20"] = ret1.rolling(20).std()

    # 成交量变化:今天成交量相对过去20天平均的比值(取对数)
    g["volchg_20"] = np.log(g["Volume"] / g["Volume"].rolling(20).mean())

    # 目标:未来HORIZON天的收益。shift(-HORIZON)取的是未来的价格,
    # 它只能出现在目标里,绝不能进入特征。
    g["target"] = close.shift(-HORIZON) / close - 1
    return g


def build_dataset(prices):
    parts = [add_features(g) for _, g in prices.groupby("ticker")]
    return pd.concat(parts, ignore_index=True)


def main():
    prices = pd.read_csv("data/prices_clean.csv", parse_dates=["date"])
    # ETF只作为基准,不参与个股排名
    prices = prices[prices["ticker"] != "1306.T"]
    data = build_dataset(prices)
    data.to_csv("data/dataset.csv", index=False)
    print(data.shape)
    print(data.dropna().shape)
    print(data[data["ticker"] == "7203.T"].dropna().head(3).to_string())


if __name__ == "__main__":
    main()
