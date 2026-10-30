import numpy as np
import pandas as pd

from src.evaluate import daily_ic, summarize_ic
from src.walkforward import make_splits, run_walkforward

SEED = 42


def random_baseline(train, test):
    rng = np.random.default_rng(SEED)
    return rng.normal(size=len(test))


def momentum_baseline(train, test):
    return test["mom_20"].values


def main():
    data = pd.read_csv("data/dataset.csv", parse_dates=["date"])
    splits = make_splits(data["date"])
    print("Number of walk forward windows:", len(splits))
    for tr, te in splits:
        print("  train", tr[0].date(), "to", tr[-1].date(), "| test", te[0].date(), "to", te[-1].date())

    for name, fn in [("random", random_baseline), ("momentum_20", momentum_baseline)]:
        res = run_walkforward(data, fn, splits)
        ic = daily_ic(res, "pred")
        print(name, summarize_ic(ic))


if __name__ == "__main__":
    main()
