import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import HOLD_DAYS, metrics, run_backtest

MODELS = ["random", "momentum_20", "ridge", "lasso", "lightgbm"]


def etf_returns(rebal_dates):
    px = pd.read_csv("data/prices_clean.csv", parse_dates=["date"])
    etf = px[px["ticker"] == "1306.T"].sort_values("date").set_index("date")["Close"]
    fwd = etf.shift(-HOLD_DAYS) / etf - 1
    return fwd.reindex(rebal_dates).dropna()


def main():
    rows, curves = {}, {}
    for name in MODELS:
        pred = pd.read_csv(f"data/pred_{name}.csv", parse_dates=["date"])
        bt = run_backtest(pred)
        m = metrics(bt["net"])
        m["gross_sharpe"] = metrics(bt["gross"])["sharpe"]
        m["avg_turnover"] = bt["turnover"].mean()
        rows[name] = m
        curves[name] = (1 + bt["net"]).cumprod()

    rows["equal_weight_30"] = metrics(bt["ew"])
    curves["equal_weight_30"] = (1 + bt["ew"]).cumprod()
    etf = etf_returns(bt.index)
    rows["buy_hold_TOPIX_ETF"] = metrics(etf)
    curves["buy_hold_TOPIX_ETF"] = (1 + etf).cumprod()

    out = pd.DataFrame(rows).T
    out.to_csv("results/backtest_summary.csv")
    print(out.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, c in curves.items():
        ax.plot(c.index, c.values, label=name, linewidth=2 if "buy_hold" in name else 1)
    ax.set_title("Cumulative return, net of 10 bps one way costs (models), 2021 to 2026")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/equity_curves.png", dpi=150)
    print("Saved results/equity_curves.png")


if __name__ == "__main__":
    main()
