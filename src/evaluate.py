import numpy as np
import pandas as pd


def daily_ic(df, pred_col, target_col="target"):
    """每天计算横截面的排名相关(Spearman),返回按日期的IC序列。"""
    def one_day(g):
        if len(g) < 5:
            return np.nan
        return g[pred_col].rank().corr(g[target_col].rank())
    return df.groupby("date").apply(one_day, include_groups=False).dropna()


def summarize_ic(ic):
    """平均IC,以及IC均值除以标准误差得到的t值(粗略)。"""
    mean = ic.mean()
    t = mean / (ic.std() / np.sqrt(len(ic)))
    return {"mean_ic": mean, "t_stat": t, "n_days": len(ic)}
