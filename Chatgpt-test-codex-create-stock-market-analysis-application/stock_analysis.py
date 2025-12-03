from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import yfinance as yf


@dataclass
class PricePoint:
    """Represents a historical closing price for a single trading day."""

    date: datetime
    close: float


@dataclass
class Signal:
    """Represents a historical buy/sell moment based on moving-average crossovers."""

    date: datetime
    action: str
    short_ma: float
    long_ma: float


def fetch_stock_data(ticker: str, period: str = "1y", use_cache: bool = True, interval: Optional[str] = None) -> List[PricePoint]:
    """Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'PLTR')
        period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        use_cache: Whether to use cached data (set False for real-time updates)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        List of PricePoint objects sorted by date
    """
    try:
        # Create new Ticker instance each time to avoid caching
        stock = yf.Ticker(ticker)
        
        # For real-time, force fresh data
        if not use_cache:
            # Clear any cached data
            try:
                if hasattr(stock, '_history'):
                    stock._history = None
                if hasattr(stock, '_info'):
                    stock._info = None
            except:
                pass
        
        # Fetch data with interval for intraday data
        if interval:
            hist = stock.history(period=period, interval=interval, prepost=False, repair=True)
        else:
            hist = stock.history(period=period, prepost=False, repair=True)
        
        if hist.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        
        rows: List[PricePoint] = []
        for date, row in hist.iterrows():
            # Convert pandas Timestamp to datetime
            if isinstance(date, pd.Timestamp):
                date_dt = date.to_pydatetime()
            else:
                date_dt = datetime.fromisoformat(str(date))
            
            rows.append(
                PricePoint(
                    date=date_dt,
                    close=float(row["Close"]),
                )
            )
        
        # Sort by date to ensure chronological order
        rows.sort(key=lambda x: x.date)
        return rows
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {e}")


def load_price_history(path: Path) -> List[PricePoint]:
    """Load historical price data from a CSV file.

    The file is expected to have `date` and `close` columns where `date` is in
    ISO-8601 format (YYYY-MM-DD).
    """

    rows: List[PricePoint] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PricePoint(
                    date=datetime.fromisoformat(row["date"]),
                    close=float(row["close"]),
                )
            )
    return rows


def moving_averages(prices: Sequence[PricePoint], window: int) -> List[Optional[float]]:
    """Calculate simple moving averages for a list of prices."""

    closing_prices = [p.close for p in prices]
    result: List[Optional[float]] = []
    for idx in range(len(prices)):
        if idx + 1 < window:
            result.append(None)
            continue
        window_prices = closing_prices[idx + 1 - window : idx + 1]
        result.append(sum(window_prices) / window)
    return result


def detect_crossovers(
    prices: Sequence[PricePoint],
    short_ma: Sequence[Optional[float]],
    long_ma: Sequence[Optional[float]],
) -> List[Signal]:
    """Identify historical buy/sell moments using moving-average crossovers."""

    signals: List[Signal] = []
    previous_diff: Optional[float] = None
    for idx, price in enumerate(prices):
        short_val = short_ma[idx]
        long_val = long_ma[idx]
        if short_val is None or long_val is None:
            continue

        diff = short_val - long_val
        if previous_diff is not None:
            if previous_diff <= 0 < diff:
                signals.append(
                    Signal(
                        date=price.date,
                        action="Buy",
                        short_ma=short_val,
                        long_ma=long_val,
                    )
                )
            elif previous_diff >= 0 > diff:
                signals.append(
                    Signal(
                        date=price.date,
                        action="Sell",
                        short_ma=short_val,
                        long_ma=long_val,
                    )
                )
        previous_diff = diff
    return signals


def summarize_trend(prices: Sequence[PricePoint]) -> str:
    """Provide a simple description of the overall trend for the dataset."""

    if not prices:
        return "No data available."

    first, last = prices[0], prices[-1]
    change = last.close - first.close
    percent = (change / first.close) * 100 if first.close else 0
    direction = "upward" if change > 0 else "downward" if change < 0 else "flat"
    return (
        f"Prices moved {direction} by {change:.2f} points "
        f"({percent:.2f}% change) from {first.date.date()} to {last.date.date()}."
    )


def format_signals(signals: Iterable[Signal]) -> str:
    lines = []
    for signal in signals:
        lines.append(
            f"- {signal.date.date()}: {signal.action} signal when short MA "
            f"({signal.short_ma:.2f}) crossed {'above' if signal.action == 'Buy' else 'below'} "
            f"long MA ({signal.long_ma:.2f})."
        )
    return "\n".join(lines) if lines else "No crossover signals were detected."


def detect_peaks_and_valleys(prices: Sequence[PricePoint], window: int = 5) -> Tuple[List[PricePoint], List[PricePoint]]:
    """Detect local peaks (highs) and valleys (lows) in price data.
    
    Args:
        prices: Sequence of price points
        window: Number of days on each side to consider for peak/valley detection
    
    Returns:
        Tuple of (peaks, valleys) - lists of PricePoint objects
    """
    if len(prices) < window * 2 + 1:
        return [], []
    
    peaks: List[PricePoint] = []
    valleys: List[PricePoint] = []
    
    for i in range(window, len(prices) - window):
        current = prices[i]
        # Check if it's a peak (higher than neighbors)
        is_peak = all(current.close >= prices[i - j].close for j in range(1, window + 1)) and \
                  all(current.close >= prices[i + j].close for j in range(1, window + 1))
        
        # Check if it's a valley (lower than neighbors)
        is_valley = all(current.close <= prices[i - j].close for j in range(1, window + 1)) and \
                    all(current.close <= prices[i + j].close for j in range(1, window + 1))
        
        if is_peak:
            peaks.append(current)
        elif is_valley:
            valleys.append(current)
    
    return peaks, valleys


def build_report(prices: Sequence[PricePoint], signals: Sequence[Signal], short: int, long: int) -> str:
    """Compose a human-readable report summarizing historical behavior."""

    latest_price = prices[-1].close if prices else float("nan")
    report_lines = [
        "Historical Market Behavior Report",
        "---------------------------------",
        summarize_trend(prices),
        f"Latest closing price: {latest_price:.2f}",
        "",
        (
            "Below are the moments when many trend-following participants would "
            "have considered buying or selling based on the classic short/long "
            f"moving-average crossover strategy ({short}-day vs {long}-day)."
        ),
        format_signals(signals),
    ]
    return "\n".join(report_lines)


def create_visualization(
    prices: Sequence[PricePoint],
    short_ma: Sequence[Optional[float]],
    long_ma: Sequence[Optional[float]],
    signals: Sequence[Signal],
    ticker: str = "Stock",
    short_window: int = 5,
    long_window: int = 10,
    output_path: Optional[str] = None,
) -> None:
    """Create a comprehensive visualization of stock price trends with buy/sell signals.
    
    Args:
        prices: Historical price data
        short_ma: Short moving average values
        long_ma: Long moving average values
        signals: Buy/sell signals
        ticker: Stock ticker symbol for title
        short_window: Short MA window size
        long_window: Long MA window size
        output_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Extract dates and prices
    dates = [p.date for p in prices]
    closes = [p.close for p in prices]
    
    # Plot price line
    ax.plot(dates, closes, label="Closing Price", color="black", linewidth=1.5, alpha=0.7)
    
    # Plot moving averages
    short_ma_values = [ma if ma is not None else np.nan for ma in short_ma]
    long_ma_values = [ma if ma is not None else np.nan for ma in long_ma]
    ax.plot(dates, short_ma_values, label=f"{short_window}-day MA", color="blue", linewidth=1.5, alpha=0.8)
    ax.plot(dates, long_ma_values, label=f"{long_window}-day MA", color="orange", linewidth=1.5, alpha=0.8)
    
    # Detect and plot peaks and valleys
    peaks, valleys = detect_peaks_and_valleys(prices, window=5)
    
    if peaks:
        peak_dates = [p.date for p in peaks]
        peak_prices = [p.close for p in peaks]
        ax.scatter(peak_dates, peak_prices, color="red", marker="v", s=150, 
                  label="Price Peaks (Sell Zones)", zorder=5, alpha=0.8, edgecolors="darkred", linewidths=1.5)
        # Annotate peaks with price
        for peak in peaks[-10:]:  # Show last 10 peaks to avoid clutter
            ax.annotate(f"${peak.close:.2f}\nSell", 
                       xy=(peak.date, peak.close),
                       xytext=(10, 20), textcoords="offset points",
                       fontsize=8, ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    
    if valleys:
        valley_dates = [p.date for p in valleys]
        valley_prices = [p.close for p in valleys]
        ax.scatter(valley_dates, valley_prices, color="green", marker="^", s=150,
                  label="Price Valleys (Buy Zones)", zorder=5, alpha=0.8, edgecolors="darkgreen", linewidths=1.5)
        # Annotate valleys with price
        for valley in valleys[-10:]:  # Show last 10 valleys to avoid clutter
            ax.annotate(f"${valley.close:.2f}\nBuy", 
                       xy=(valley.date, valley.close),
                       xytext=(10, -30), textcoords="offset points",
                       fontsize=8, ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    
    # Plot buy/sell signals
    buy_signals = [s for s in signals if s.action == "Buy"]
    sell_signals = [s for s in signals if s.action == "Sell"]
    
    # Get prices at signal dates
    price_dict = {p.date.date(): p.close for p in prices}
    
    if buy_signals:
        buy_dates = [s.date for s in buy_signals]
        buy_prices = [price_dict.get(s.date.date(), 0) for s in buy_signals]
        ax.scatter(buy_dates, buy_prices, color="lime", marker="^", s=200,
                  label="Buy Signals (MA Crossover)", zorder=6, alpha=0.9, 
                  edgecolors="darkgreen", linewidths=2)
    
    if sell_signals:
        sell_dates = [s.date for s in sell_signals]
        sell_prices = [price_dict.get(s.date.date(), 0) for s in sell_signals]
        ax.scatter(sell_dates, sell_prices, color="crimson", marker="v", s=200,
                  label="Sell Signals (MA Crossover)", zorder=6, alpha=0.9,
                  edgecolors="darkred", linewidths=2)
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
    ax.set_title(f"{ticker} Stock Analysis - Price Trends & Trading Signals", 
                fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nGraph saved to: {output_path}")
    else:
        plt.savefig(f"{ticker}_analysis.png", dpi=300, bbox_inches="tight")
        print(f"\nGraph saved to: {ticker}_analysis.png")
    
    plt.show()


def create_realtime_visualization(
    ticker: str,
    short_window: int = 5,
    long_window: int = 10,
    period: str = "1d",
    refresh_interval: int = 30,
) -> None:
    """Create a real-time updating visualization of stock price trends.
    
    Args:
        ticker: Stock ticker symbol
        short_window: Short MA window size
        long_window: Long MA window size
        period: Time period for historical data (use shorter periods for real-time)
        refresh_interval: Refresh interval in seconds
    """
    # Enable interactive mode for real-time updates
    plt.ion()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.canvas.manager.set_window_title(f"{ticker} Real-Time Analysis")
    
    print(f"\n{'='*60}")
    print(f"Real-time analysis started for {ticker}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print(f"Using intraday data (5-minute intervals) for live updates")
    print(f"Press Ctrl+C or close the window to stop")
    print(f"{'='*60}\n")
    
    update_count = 0
    last_data_hash = None
    
    def plot_data(prices: List[PricePoint], update_num: int):
        """Helper function to plot the data"""
        nonlocal last_data_hash
        
        # Calculate moving averages and signals
        short_ma = moving_averages(prices, short_window)
        long_ma = moving_averages(prices, long_window)
        signals = detect_crossovers(prices, short_ma, long_ma)
        
        # Extract data
        dates = [p.date for p in prices]
        closes = [p.close for p in prices]
        short_ma_values = [ma if ma is not None else np.nan for ma in short_ma]
        long_ma_values = [ma if ma is not None else np.nan for ma in long_ma]
        
        # Check if data actually changed
        current_hash = hash((len(prices), prices[-1].date.isoformat() if prices else "", prices[-1].close if prices else 0))
        if current_hash == last_data_hash and update_num > 1:
            print(f"[Update #{update_num}] No new data available (data unchanged)")
            return
        last_data_hash = current_hash
        
        # Clear previous plots
        ax.clear()
        
        # Plot price line
        ax.plot(dates, closes, label="Closing Price", 
               color="black", linewidth=1.5, alpha=0.7)
        
        # Plot moving averages
        ax.plot(dates, short_ma_values, label=f"{short_window}-period MA", 
               color="blue", linewidth=1.5, alpha=0.8)
        ax.plot(dates, long_ma_values, label=f"{long_window}-period MA", 
               color="orange", linewidth=1.5, alpha=0.8)
        
        # Detect and plot peaks and valleys
        peaks, valleys = detect_peaks_and_valleys(prices, window=min(5, len(prices)//10))
        
        if peaks:
            peak_dates = [p.date for p in peaks]
            peak_prices = [p.close for p in peaks]
            ax.scatter(peak_dates, peak_prices, color="red", marker="v", s=150,
                      label="Price Peaks (Sell Zones)", zorder=5, alpha=0.8,
                      edgecolors="darkred", linewidths=1.5)
            # Annotate recent peaks
            for peak in peaks[-5:]:
                ax.annotate(f"${peak.close:.2f}", 
                           xy=(peak.date, peak.close),
                           xytext=(5, 15), textcoords="offset points",
                           fontsize=8, ha="left",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        
        if valleys:
            valley_dates = [p.date for p in valleys]
            valley_prices = [p.close for p in valleys]
            ax.scatter(valley_dates, valley_prices, color="green", marker="^", s=150,
                      label="Price Valleys (Buy Zones)", zorder=5, alpha=0.8,
                      edgecolors="darkgreen", linewidths=1.5)
            # Annotate recent valleys
            for valley in valleys[-5:]:
                ax.annotate(f"${valley.close:.2f}", 
                           xy=(valley.date, valley.close),
                           xytext=(5, -25), textcoords="offset points",
                           fontsize=8, ha="left",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
        
        # Plot buy/sell signals
        buy_signals = [s for s in signals if s.action == "Buy"]
        sell_signals = [s for s in signals if s.action == "Sell"]
        price_dict = {p.date: p.close for p in prices}
        
        if buy_signals:
            buy_dates = [s.date for s in buy_signals]
            buy_prices = [price_dict.get(s.date, 0) for s in buy_signals]
            ax.scatter(buy_dates, buy_prices, color="lime", marker="^", s=200,
                      label="Buy Signals (MA Crossover)", zorder=6, alpha=0.9,
                      edgecolors="darkgreen", linewidths=2)
        
        if sell_signals:
            sell_dates = [s.date for s in sell_signals]
            sell_prices = [price_dict.get(s.date, 0) for s in sell_signals]
            ax.scatter(sell_dates, sell_prices, color="crimson", marker="v", s=200,
                      label="Sell Signals (MA Crossover)", zorder=6, alpha=0.9,
                      edgecolors="darkred", linewidths=2)
        
        # Formatting
        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
        current_price = prices[-1].close if prices else 0
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(f"{ticker} Real-Time Stock Analysis | Current: ${current_price:.2f} | Updated: {current_time}", 
                    fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Update status
        latest_signal = signals[-1] if signals else None
        signal_info = ""
        if latest_signal:
            signal_info = f" | Latest: {latest_signal.action} @ {latest_signal.date.strftime('%H:%M')}"
        ax.text(0.02, 0.98, 
               f"Refresh: {refresh_interval}s | Updates: {update_num} | Data points: {len(prices)}{signal_info}",
               transform=ax.transAxes, fontsize=9, verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        # Force redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print(f"[Update #{update_num}] ✓ Graph updated - Price: ${current_price:.2f} | Data points: {len(prices)} | Time: {current_time}")
    
    # Main update loop
    try:
        while True:
            update_count += 1
            
            try:
                # Fetch latest intraday data (5-minute intervals for real-time)
                # Use 1d period with 5m interval for intraday data
                prices = fetch_stock_data(ticker, period="1d", use_cache=False, interval="5m")
                
                if not prices:
                    ax.clear()
                    ax.text(0.5, 0.5, f"Error: No data available for {ticker}", 
                           transform=ax.transAxes, fontsize=14, ha="center",
                           bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    print(f"[Update #{update_count}] ⚠ No data available")
                else:
                    plot_data(prices, update_count)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                ax.clear()
                error_msg = f"Error: {str(e)}"
                ax.text(0.5, 0.5, error_msg, 
                       transform=ax.transAxes, fontsize=12, ha="center",
                       bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))
                fig.canvas.draw()
                fig.canvas.flush_events()
                print(f"[Update #{update_count}] ✗ Error: {e}")
            
            # Wait for next update
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nReal-time analysis stopped by user.")
    finally:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze historical price movements to see when traders would have "
            "most commonly bought or sold based on moving-average crossovers. "
            "Fetches Palantir (PLTR) stock data by default."
        )
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="PLTR",
        help="Stock ticker symbol to analyze (default: PLTR)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        help="Time period for historical data (default: 1y)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional: Path to a CSV file with date and close columns (overrides ticker)",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=5,
        help="Window (in days) for the short moving average (default: 5)",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=10,
        help="Window (in days) for the long moving average (default: 10)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip generating the visualization graph",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the graph image (default: {ticker}_analysis.png)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable real-time analysis with live updating graph",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=30,
        help="Refresh interval in seconds for real-time mode (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        # Real-time mode
        if args.realtime:
            if args.csv_path:
                print("Error: Real-time mode is not available for CSV files.")
                print("Please use --ticker instead of --csv-path for real-time analysis.")
                return
            
            # Use shorter period for real-time (1d or 5d recommended)
            realtime_period = "1d" if args.period in ["1y", "2y", "5y", "10y", "max"] else args.period
            if realtime_period not in ["1d", "5d", "1mo"]:
                print(f"Warning: Using '1d' period for real-time mode (recommended for live updates)")
                realtime_period = "1d"
            
            create_realtime_visualization(
                ticker=args.ticker,
                short_window=args.short_window,
                long_window=args.long_window,
                period=realtime_period,
                refresh_interval=args.refresh_interval,
            )
            return
        
        # Standard mode (one-time analysis)
        # Load data from CSV or fetch from yfinance
        if args.csv_path:
            print(f"Loading data from CSV: {args.csv_path}")
            prices = load_price_history(args.csv_path)
            ticker = "CSV Data"
        else:
            print(f"Fetching {args.ticker} stock data for period: {args.period}...")
            prices = fetch_stock_data(args.ticker, args.period)
            ticker = args.ticker
            print(f"Successfully fetched {len(prices)} data points")
        
        if not prices:
            print("Error: No price data available")
            return
        
        # Calculate moving averages and detect signals
        short_ma = moving_averages(prices, args.short_window)
        long_ma = moving_averages(prices, args.long_window)
        signals = detect_crossovers(prices, short_ma, long_ma)

        # Generate and print report
        report = build_report(prices, signals, args.short_window, args.long_window)
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        # Create visualization unless disabled
        if not args.no_graph:
            print("\nGenerating visualization...")
            create_visualization(
                prices,
                short_ma,
                long_ma,
                signals,
                ticker=ticker,
                short_window=args.short_window,
                long_window=args.long_window,
                output_path=args.output,
            )
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{args.csv_path}'")
        print("Please provide a valid path to a CSV file with 'date' and 'close' columns.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
    except KeyboardInterrupt:
        print("\n\nReal-time analysis stopped by user.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

