import time
from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = [
    "7203.T", "6758.T", "9984.T", "8306.T", "6861.T", "9432.T",
    "6501.T", "8035.T", "4063.T", "7974.T", "9983.T", "8316.T",
    "6098.T", "4502.T", "8058.T", "8001.T", "8031.T", "6367.T",
    "7267.T", "6902.T", "4519.T", "6954.T", "9433.T", "9434.T",
    "2914.T", "8766.T", "6594.T", "6981.T", "7741.T", "4568.T",
    "1306.T",
]

START = "2016-01-01"
END = "2026-10-01"
RAW_DIR = Path("data/raw")


def download_one(ticker, retries=4):
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=True)
            if df.empty:
                print(f"WARNING: no data for {ticker}")
                return None
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = df.index.tz_localize(None)
            df.index.name = "date"
            df["ticker"] = ticker
            return df.reset_index()
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"{ticker} failed ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
    return None


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        path = RAW_DIR / f"{t}.csv"
        if path.exists():
            print(f"skip {t} (already downloaded)")
            continue
        df = download_one(t)
        if df is not None:
            df.to_csv(path, index=False)
            print(f"saved {t}: {len(df)} rows")
        time.sleep(3)

    files = sorted(RAW_DIR.glob("*.csv"))
    prices = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    prices.to_csv("data/prices.csv", index=False)
    print("Combined:", prices["ticker"].nunique(), "tickers,", len(prices), "rows")


if __name__ == "__main__":
    main()
