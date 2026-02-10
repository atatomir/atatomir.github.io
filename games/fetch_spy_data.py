#!/usr/bin/env python3
"""Fetch real SPY 5-min intraday data and save as JSON files.

Only uses 5-minute interval data filtered to regular market hours (9:30-16:00 ET)
so charts match TradingView exactly. Yahoo Finance limits 5m data to ~60 trading days.
"""

import json
import os
from datetime import time

import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Regular market hours: 9:30 AM - 4:00 PM Eastern
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# 6.5 hours * 12 bars/hour = 78 bars expected; require at least 70
MIN_BARS = 70


def fetch_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)

    ticker = yf.Ticker("SPY")
    all_days = {}

    print("Fetching 5-min data (last 60 days)...")
    df = ticker.history(period="60d", interval="5m")
    if df.empty:
        print("ERROR: No data returned from yfinance")
        return

    # Convert to Eastern time
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    # Filter to regular market hours only
    df = df.between_time(MARKET_OPEN, MARKET_CLOSE, inclusive="left")

    # Remove timezone info after filtering
    df.index = df.index.tz_localize(None)

    for date, group in df.groupby(df.index.date):
        closes = group["Close"].dropna().tolist()
        if len(closes) >= MIN_BARS:
            all_days[str(date)] = closes

    print(f"  Got {len(all_days)} complete trading days")

    sorted_dates = sorted(all_days.keys())

    # Clean out old files
    for f in os.listdir(DATA_DIR):
        if f.startswith("SPY_") and f.endswith(".json"):
            os.remove(os.path.join(DATA_DIR, f))

    # Save individual day files
    saved = []
    for date_str in sorted_dates:
        prices = [round(p, 2) for p in all_days[date_str]]
        filename = f"SPY_{date_str}.json"
        with open(os.path.join(DATA_DIR, filename), "w") as f:
            json.dump({"date": date_str, "prices": prices}, f)
        saved.append(filename)

    # Save index
    with open(os.path.join(DATA_DIR, "index.json"), "w") as f:
        json.dump({"files": saved}, f, indent=2)

    # Generate bundled JS file (works with file:// protocol, no CORS issues)
    entries = []
    for date_str in sorted_dates:
        prices = [round(p, 2) for p in all_days[date_str]]
        entries.append(json.dumps([date_str, prices]))
    with open(os.path.join(DATA_DIR, "spy_data.js"), "w") as f:
        f.write("window.SPY_DATA=[\n" + ",\n".join(entries) + "\n];\n")

    print(f"\nDone! Saved {len(saved)} days + spy_data.js to {DATA_DIR}/")
    if sorted_dates:
        print(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        # Show bar counts for first and last day
        for d in [sorted_dates[0], sorted_dates[-1]]:
            print(f"  {d}: {len(all_days[d])} bars")


if __name__ == "__main__":
    fetch_and_save()
