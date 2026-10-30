import pandas as pd

from src import models
from src.evaluate import daily_ic, summarize_ic
from src.walkforward import make_splits, run_walkforward

CANDIDATES = {
    "random": models.random_baseline,
    "momentum_20": models.momentum_baseline,
    "ridge": models.ridge(),
    "lasso": models.lasso(),
    "lightgbm": models.lightgbm(),
}


def main():
    data = pd.read_csv("data/dataset.csv", parse_dates=["date"])
    splits = make_splits(data["date"])
    rows = []
    for name, fn in CANDIDATES.items():
        res = run_walkforward(data, fn, splits)
        res[["date", "ticker", "pred", "target"]].to_csv(f"data/pred_{name}.csv", index=False)
        ic = daily_ic(res, "pred")
        s = summarize_ic(ic)
        s["model"] = name
        s["ic_std"] = ic.std()
        s["share_positive_days"] = (ic > 0).mean()
        rows.append(s)
        print(name, round(s["mean_ic"], 4), round(s["t_stat"], 2))
    out = pd.DataFrame(rows).set_index("model")
    out.to_csv("results/ic_summary.csv")
    print()
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
