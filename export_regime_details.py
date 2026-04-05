#!/usr/bin/env python3
"""
Export detailed regime conditions for each 1H bar.
Useful for analyzing which conditions passed/failed at each signal point.
"""

import asyncio
import csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import httpx
import config
from indicators import compute_indicators

async def fetch_hist(pair: str, interval: str, days: int, extra_candles: int = 0) -> pd.DataFrame:
    """Fetch historical klines (same logic as backtest)."""
    bars_per_day = {"1h": 24, "4h": 6, "15m": 96, "1d": 1}.get(interval, 24)
    total_needed = days * bars_per_day + extra_candles
    PAGE = 1000

    async with httpx.AsyncClient(timeout=20) as c:
        if total_needed <= PAGE:
            r = await c.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": pair, "interval": interval, "limit": total_needed},
            )
            data = r.json()
            if not data or isinstance(data, dict):
                return pd.DataFrame()
            all_data = data
        else:
            all_data = []
            end_time = None
            remaining = total_needed
            while remaining > 0:
                page_limit = min(remaining, PAGE)
                params = {"symbol": pair, "interval": interval, "limit": page_limit}
                if end_time:
                    params["endTime"] = end_time
                r = await c.get("https://api.binance.com/api/v3/klines", params=params)
                data = r.json()
                if not data or isinstance(data, dict):
                    break
                all_data = data + all_data
                end_time = data[0][0] - 1
                remaining -= len(data)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_vol", "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time").sort_index()
    return df


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate ADX."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Choppiness Index."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    range_n = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10(atr_sum / range_n.replace(0, np.nan)) / np.log10(n)


async def main():
    print("Fetching data...")
    LOOKBACK_DAYS = 30

    hist_1h = {}
    hist_4h = {}
    hist_1d = {}

    for pair in config.PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        hist_1h[pair] = await fetch_hist(pair, "1h", LOOKBACK_DAYS, extra_candles=200)
        hist_4h[pair] = await fetch_hist(pair, "4h", LOOKBACK_DAYS, extra_candles=50)
        hist_1d[pair] = await fetch_hist(pair, "1d", LOOKBACK_DAYS, extra_candles=50)
        print("OK")

    print("\nProcessing indicators...")
    for pair in hist_1h:
        hist_1h[pair] = compute_indicators(hist_1h[pair])
        hist_4h[pair] = compute_indicators(hist_4h[pair])
        hist_1d[pair] = compute_indicators(hist_1d[pair])

    # Generate regime details CSV
    print(f"\nExporting regime details to regime_export.csv...")
    with open("regime_export.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "pair",
            "close", "ema50", "ema200", "rsi", "adx14", "chop14", "bb_pctb", "atr", "atr_sma",
            "daily_close", "daily_ema21", "daily_ema50",
            "price_above_ema200", "price_near_ema50", "rsi_pullback_zone", "bb_lower",
            "adx_ranging", "chop_consolidating",
            "daily_filter_ok", "macro_uptrend_4h",
            "is_pullback", "is_trending", "detected_regime", "score"
        ])

        # Sample: Jan 1 - Jan 31
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)

        for pair in config.PAIRS:
            if pair not in hist_1h:
                continue
            df1h = hist_1h[pair][(hist_1h[pair].index >= start_date) & (hist_1h[pair].index <= end_date)]

            for ts, row in df1h.iterrows():
                price = row["close"]
                ema50 = row["ema50"]
                ema200 = row["ema200"]
                rsi = row["rsi"]
                adx_val = row.get("adx14", np.nan)
                chop_val = row.get("chop14", np.nan)
                bb_b = row.get("bb_pctb", np.nan)
                atr = row["atr"]
                atr_sma = row["atr_sma"]

                # Daily data
                if pair in hist_1d:
                    daily_slice = hist_1d[pair][hist_1d[pair].index <= ts].tail(21)
                    if len(daily_slice) > 0:
                        last_daily = daily_slice.iloc[-1]
                        daily_close = last_daily["close"]
                        daily_ema21 = last_daily.get("ema20", np.nan)
                        daily_ema50 = last_daily.get("ema50", np.nan)
                    else:
                        daily_close = daily_ema21 = daily_ema50 = np.nan
                else:
                    daily_close = daily_ema21 = daily_ema50 = np.nan

                # 4H data
                if pair in hist_4h:
                    df4h_slice = hist_4h[pair][hist_4h[pair].index <= ts].tail(50)
                else:
                    df4h_slice = pd.DataFrame()

                # Conditions
                price_above_ema200 = 1 if (not np.isnan(ema200) and price > ema200) else 0
                price_near_ema50 = 1 if (not np.isnan(ema50) and abs(price - ema50) / ema50 < 0.025) else 0
                rsi_pullback = 1 if (28 <= rsi <= 55) else 0
                rsi_trending = 1 if (50 <= rsi <= 75) else 0
                bb_lower = 1 if (not np.isnan(bb_b) and bb_b < 0.40) else 0
                adx_range = 1 if (not np.isnan(adx_val) and adx_val < 25) else 0
                adx_trend = 1 if (not np.isnan(adx_val) and adx_val > 25) else 0
                chop_consol = 1 if (not np.isnan(chop_val) and chop_val > 55) else 0
                chop_trend = 1 if (not np.isnan(chop_val) and chop_val < 38.2) else 0

                daily_ok = 1 if (not np.isnan(daily_close) and not np.isnan(daily_ema21) and
                                  not np.isnan(daily_ema50) and
                                  daily_close > daily_ema21 and daily_ema21 > daily_ema50) else 0

                macro_up = 0
                if len(df4h_slice) >= 1:
                    e50_4h = df4h_slice.iloc[-1].get("ema50", np.nan)
                    e200_4h = df4h_slice.iloc[-1].get("ema200", np.nan)
                    macro_up = 1 if (not np.isnan(e50_4h) and not np.isnan(e200_4h) and e50_4h > e200_4h) else 0

                # Pullback
                is_pb = (price_above_ema200 and price_near_ema50 and rsi_pullback and bb_lower and
                         adx_range and chop_consol and daily_ok and macro_up)

                # Trending
                is_tr = (price > ema50 and price > ema200 and rsi_trending and
                         atr >= atr_sma and adx_trend and chop_trend and daily_ok)

                regime = "PULLBACK" if is_pb else "TRENDING" if is_tr else "RANGING"

                writer.writerow([
                    ts.strftime("%Y-%m-%d %H:%M UTC"),
                    pair,
                    f"{price:.2f}",
                    f"{ema50:.2f}" if not np.isnan(ema50) else "N/A",
                    f"{ema200:.2f}" if not np.isnan(ema200) else "N/A",
                    f"{rsi:.1f}" if not np.isnan(rsi) else "N/A",
                    f"{adx_val:.1f}" if not np.isnan(adx_val) else "N/A",
                    f"{chop_val:.1f}" if not np.isnan(chop_val) else "N/A",
                    f"{bb_b:.2f}" if not np.isnan(bb_b) else "N/A",
                    f"{atr:.2f}" if not np.isnan(atr) else "N/A",
                    f"{atr_sma:.2f}" if not np.isnan(atr_sma) else "N/A",
                    f"{daily_close:.2f}" if not np.isnan(daily_close) else "N/A",
                    f"{daily_ema21:.2f}" if not np.isnan(daily_ema21) else "N/A",
                    f"{daily_ema50:.2f}" if not np.isnan(daily_ema50) else "N/A",
                    price_above_ema200, price_near_ema50, rsi_pullback, bb_lower,
                    adx_range, chop_consol,
                    daily_ok, macro_up,
                    1 if is_pb else 0, 1 if is_tr else 0,
                    regime, "N/A"
                ])

    print("✓ regime_export.csv created (open in Excel or import to TradingView alerts)")


if __name__ == "__main__":
    asyncio.run(main())
