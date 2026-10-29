import pandas as pd


def clean_prices(df):
    """删除休市日的填充行(成交量为0且最高价等于最低价)。"""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    is_filler = (df["Volume"] == 0) & (df["High"] == df["Low"])
    return df[~is_filler].reset_index(drop=True), int(is_filler.sum())


def main():
    df = pd.read_csv("data/prices.csv", parse_dates=["date"])
    clean, n_removed = clean_prices(df)
    clean.to_csv("data/prices_clean.csv", index=False)

    print("Removed filler rows:", n_removed)
    print("Remaining zero volume rows:", int((clean["Volume"] == 0).sum()))
    print("Rows after cleaning:", len(clean))
    print()
    summary = clean.groupby("ticker")["date"].agg(["min", "max", "count"])
    print(summary.sort_values("min").tail(5))
    print()
    print(summary.sort_values("count").head(5))


if __name__ == "__main__":
    main()
