import pandas as pd

# 已核实的数据源错误。每一条都要写明原因,README 里也要列出。
KNOWN_BAD = [
    ("1306.T", "2026-03-30", "2026-03-31",
     "price drops to about one tenth with abnormal volume, then recovers"),
    ("8766.T", "2026-09-29", "2026-12-31",
     "artificial jump around stock split date, adjustment inconsistent"),
]


def clean_prices(df):
    """删除休市日的填充行(成交量为0且最高价等于最低价)。"""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    is_filler = (df["Volume"] == 0) & (df["High"] == df["Low"])
    return df[~is_filler].reset_index(drop=True), int(is_filler.sum())


def remove_known_bad(df, rules=KNOWN_BAD):
    """按清单删除已核实的错误行,返回(清洗后数据, 删除行数)。"""
    mask = pd.Series(False, index=df.index)
    for ticker, start, end, _reason in rules:
        mask |= (df["ticker"] == ticker) & (df["date"] >= start) & (df["date"] <= end)
    return df[~mask].reset_index(drop=True), int(mask.sum())


def main():
    df = pd.read_csv("data/prices.csv", parse_dates=["date"])
    df, n_filler = clean_prices(df)
    df, n_bad = remove_known_bad(df)
    df.to_csv("data/prices_clean.csv", index=False)

    print("Removed filler rows:", n_filler)
    print("Removed known bad rows:", n_bad)
    print("Remaining zero volume rows:", int((df["Volume"] == 0).sum()))
    print("Rows after cleaning:", len(df))

    r = df.sort_values(["ticker", "date"]).groupby("ticker")["Close"].pct_change()
    big = df.assign(ret=r)[r.abs() > 0.15][["date", "ticker", "Close", "ret"]]
    print()
    print("Daily moves above 15 percent (please review):")
    print(big.to_string())


if __name__ == "__main__":
    main()
