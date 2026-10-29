# Walk Forward Backtesting of Return Prediction Models

Status: work in progress (started Oct 2026)

## Goal
Compare simple and complex models for predicting cross sectional stock returns on Japanese large caps, with strict walk forward validation and honest reporting. The focus is rigorous out of sample evaluation, not profit.

## Data
Daily prices for 30 Japanese large cap stocks plus the TOPIX ETF (1306.T) as a benchmark, January 2016 to September 2026, downloaded with yfinance (adjusted prices).
Holiday filler rows (zero volume, High equal to Low) are removed in src/clean_data.py.

## Features and target
Momentum (5, 20, 60 days), 20 day volatility, 20 day volume change. Target is the 5 trading day forward return. Features use only information available at the close of day t.

## Planned models
Naive baselines, ridge, Lasso, LightGBM. Evaluation by rank information coefficient, hit rate, and a cost adjusted backtest versus buy and hold.

## Reproduce
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/clean_data.py
python src/features.py
pytest -q
```

## Known limitations
Stocks were chosen by hand from today's large caps, so results are subject to survivorship bias. Yahoo Finance data may contain errors.
