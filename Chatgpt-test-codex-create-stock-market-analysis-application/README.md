# Stock Market Analysis Application

This repository provides a comprehensive command-line tool for analyzing historical stock-price behavior. It uses moving-average crossovers to flag moments when many trend-following traders *would have* bought or sold in the past, based purely on historical data. The tool now includes **interactive visualizations** showing price trends, buy/sell signals, and optimal entry/exit points.

## Features

- 🔴 **Live Real-Time Analysis**: Continuously updates stock data and graphs with live price movements
- 📈 **Real-time Stock Data**: Fetches historical data directly from Yahoo Finance (default: Palantir/PLTR)
- 📊 **Interactive Graphs**: Visual representation of price trends with buy/sell signals
- 🎯 **Peak Detection**: Identifies price peaks (sell zones) and valleys (buy zones)
- 📉 **Moving Average Analysis**: Configurable short and long moving averages
- 💹 **Trading Signals**: Automatic detection of buy/sell opportunities based on MA crossovers

## Getting Started

1. Ensure Python 3.10+ is available.
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Analyze Palantir (PLTR) Stock (Default)

```bash
python stock_analysis.py
```

This will fetch the last year of PLTR data and generate both a text report and a visualization graph.

### Analyze Other Stocks

```bash
python stock_analysis.py --ticker AAPL --period 2y
```

### Customize Moving Averages

```bash
python stock_analysis.py --ticker PLTR --short-window 10 --long-window 30
```

### Real-Time Live Analysis

Monitor stock prices in real-time with auto-updating graphs:

```bash
python stock_analysis.py --realtime
```

Customize the refresh interval (in seconds):

```bash
python stock_analysis.py --realtime --refresh-interval 15
```

The real-time mode will:
- Continuously fetch the latest stock data
- Update the graph automatically at your specified interval
- Show current price and latest trading signals
- Display timestamp of last update

**Note**: For real-time analysis, shorter periods (1d, 5d) are recommended for better performance.

### Available Options

- `--ticker`: Stock ticker symbol (default: PLTR)
- `--period`: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
- `--short-window`: Days for short moving average (default: 5)
- `--long-window`: Days for long moving average (default: 10)
- `--csv-path`: Optional path to CSV file (overrides ticker)
- `--no-graph`: Skip generating the visualization
- `--output`: Custom path to save the graph image
- `--realtime`: Enable real-time analysis with live updating graph
- `--refresh-interval`: Refresh interval in seconds for real-time mode (default: 30)

### Using CSV Files

You can still use CSV files if preferred:

```bash
python stock_analysis.py --csv-path data/sample_prices.csv
```

## Output

The tool generates:
1. **Text Report**: Summary of price trends and all buy/sell signals
2. **Visualization Graph**: Shows:
   - Price history line
   - Short and long moving averages
   - Buy signals (green triangles) - where MA crossover suggests buying
   - Sell signals (red triangles) - where MA crossover suggests selling
   - Price peaks (red markers) - optimal sell zones
   - Price valleys (green markers) - optimal buy zones

The graph is automatically saved as `{TICKER}_analysis.png` in high resolution (300 DPI).

### Real-Time Mode Output

In real-time mode, the graph updates automatically showing:
- Current stock price in the title
- Last update timestamp
- Live buy/sell signals as they occur
- Continuously updating peaks and valleys
- Status information with refresh interval and data point count

Press `Ctrl+C` to stop the real-time analysis.

A small sample dataset is provided in `data/sample_prices.csv` for quick experimentation.
