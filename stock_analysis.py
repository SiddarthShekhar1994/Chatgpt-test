from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze historical price movements to see when traders would have "
            "most commonly bought or sold based on moving-average crossovers."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to a CSV file with date and close columns")
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
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    prices = load_price_history(args.csv_path)
    short_ma = moving_averages(prices, args.short_window)
    long_ma = moving_averages(prices, args.long_window)
    signals = detect_crossovers(prices, short_ma, long_ma)

    report = build_report(prices, signals, args.short_window, args.long_window)
    print(report)


if __name__ == "__main__":
    main()
