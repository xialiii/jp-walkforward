import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.walkforward import FEATURES

SEED = 42


def demeaned_target(train):
    """目标减去当天所有股票的平均值。我们关心的是股票之间的排名,
    不是大盘整体涨跌。只用训练集内部的数据计算,不涉及测试集。"""
    return train["target"] - train.groupby("date")["target"].transform("mean")


def random_baseline(train, test):
    rng = np.random.default_rng(SEED)
    return rng.normal(size=len(test))


def momentum_baseline(train, test):
    return test["mom_20"].values


def _fit_predict(model):
    def fn(train, test):
        model.fit(train[FEATURES], demeaned_target(train))
        return model.predict(test[FEATURES])
    return fn


def ridge(alpha=1.0):
    return _fit_predict(make_pipeline(StandardScaler(), Ridge(alpha=alpha)))


def lasso(alpha=1e-4):
    return _fit_predict(make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000)))


def lightgbm():
    model = LGBMRegressor(
        n_estimators=200, learning_rate=0.03, num_leaves=8,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=SEED, deterministic=True,
        force_row_wise=True, n_jobs=1, verbose=-1,
    )
    return _fit_predict(model)
