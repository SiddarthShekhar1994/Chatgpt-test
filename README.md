# Chatgpt-test

This repository provides a simple command-line tool for summarizing historical stock-price behavior. It uses moving-average crossovers to flag moments when many trend-following traders *would have* bought or sold in the past, based purely on historical data.

## Getting started
1. Ensure Python 3.10+ is available.
2. Install optional helper packages if you want additional CSV tooling (not required).

## Usage
Run the analyzer against any CSV file containing `date` and `close` columns:

```bash
python stock_analysis.py data/sample_prices.csv --short-window 5 --long-window 10
```

You will receive a plain-text report highlighting the overall historical trend and crossover-based buy/sell moments.

A small sample dataset is provided in `data/sample_prices.csv` for quick experimentation.
