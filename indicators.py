"""
ATLAS Lean v2.0 — Indicator Engine
Computes all technical indicators from raw OHLCV data using pandas-ta.
Zero external cost. Computed once per cycle and cached in memory.
Every indicator value is also interpreted into human-readable context for Claude.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Computation
# ---------------------------------------------------------------------------

def compute_indicators(klines: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Add all indicator columns to each timeframe DataFrame.
    Input:  {"15m": df, "1h": df, "4h": df, "1d": df, "1w": df}
    Output: Same dict with indicator columns appended.
    """
    result = {}
    for tf, df in klines.items():
        if df is None or df.empty:
            continue
        try:
            result[tf] = _add_indicators(df.copy(), tf)
        except Exception as e:
            logger.warning("compute_indicators(%s) failed: %s", tf, e)
            result[tf] = df  # Return raw data if indicators fail
    return result


def _add_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Add all technical indicator columns to a single DataFrame."""

    # --- Trend: EMAs ---
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)
    if len(df) >= 200:
        df["ema200"] = ta.ema(df["close"], length=200)
    else:
        df["ema200"] = np.nan

    # --- Momentum: RSI ---
    df["rsi14"] = ta.rsi(df["close"], length=14)

    # --- Momentum: MACD ---
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_line"] = macd.iloc[:, 0]
        df["macd_hist"] = macd.iloc[:, 1]
        df["macd_signal"] = macd.iloc[:, 2]
    else:
        df["macd_line"] = df["macd_hist"] = df["macd_signal"] = np.nan

    # --- Momentum: Stochastic ---
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    else:
        df["stoch_k"] = df["stoch_d"] = np.nan

    # --- Momentum: ROC ---
    df["roc10"] = ta.roc(df["close"], length=10)

    # --- Volatility: ATR ---
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_pct"] = (df["atr14"] / df["close"]) * 100

    # ATR SMA for expansion/contraction detection
    df["atr14_sma20"] = df["atr14"].rolling(20).mean()

    # --- Volatility: Bollinger Bands ---
    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None:
        df["bb_lower"] = bbands.iloc[:, 0]
        df["bb_mid"] = bbands.iloc[:, 1]
        df["bb_upper"] = bbands.iloc[:, 2]
        df["bb_width"] = bbands.iloc[:, 3]
        df["bb_pctb"] = bbands.iloc[:, 4]
    else:
        df["bb_lower"] = df["bb_mid"] = df["bb_upper"] = np.nan
        df["bb_width"] = df["bb_pctb"] = np.nan

    # BB Width average for compression detection
    df["bb_width_sma20"] = df["bb_width"].rolling(20).mean()

    # --- Volume: SMA ---
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]

    # Volume trend (last 3 candles)
    df["vol_trend"] = df["volume"].rolling(3).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
    )

    # --- Taker buy ratio (from raw kline data) ---
    if "taker_buy_volume" in df.columns:
        df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)

    # --- Trade count ratio ---
    if "trades" in df.columns:
        df["trades_sma20"] = df["trades"].rolling(20).mean()
        df["trades_ratio"] = df["trades"] / df["trades_sma20"].replace(0, np.nan)

    # --- Compression ratio (for breakout detection) ---
    if not df["atr14"].isna().all():
        recent_range = df["high"].rolling(6).max() - df["low"].rolling(6).min()
        df["compression_ratio"] = recent_range / df["atr14"].replace(0, np.nan)

    # --- ATR expansion rate ---
    df["atr_expansion"] = df["atr14"] / df["atr14"].shift(5).replace(0, np.nan)

    # --- Regime Detection: ADX (Average Directional Index) ---
    # ADX < 20 = ranging/non-trending; ADX > 25 = strong trend confirmed
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_result is not None and not adx_result.empty:
        df["adx14"]   = adx_result.iloc[:, 0]  # ADX_14 — trend strength
        df["adx_pos"] = adx_result.iloc[:, 1]  # DMP_14 — +DI (bullish directional move)
        df["adx_neg"] = adx_result.iloc[:, 2]  # DMN_14 — -DI (bearish directional move)
    else:
        df["adx14"] = df["adx_pos"] = df["adx_neg"] = np.nan

    # --- Regime Detection: Choppiness Index ---
    # CHOP > 61.8 (Fibonacci) = choppy/ranging; CHOP < 38.2 = trending strongly
    chop = ta.chop(df["high"], df["low"], df["close"], length=14)
    if chop is not None:
        df["chop14"] = chop
    else:
        df["chop14"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Volume Metrics (from supplementary endpoints)
# ---------------------------------------------------------------------------

def compute_volume_metrics(
    klines_1h: Optional[pd.DataFrame],
    klines_4h: Optional[pd.DataFrame],
    depth: Optional[dict],
    book_ticker: Optional[dict],
    agg_trades: Optional[list[dict]],
    ticker24h: Optional[dict],
) -> dict:
    """
    Compute the 7+ volume metrics from all data sources.
    Returns a dict of computed metrics.
    """
    metrics = {}

    # --- Volume ratio from 1H klines ---
    if klines_1h is not None and "vol_ratio" in klines_1h.columns:
        latest = klines_1h["vol_ratio"].iloc[-1]
        metrics["volume_ratio_1h"] = round(float(latest), 3) if not pd.isna(latest) else 1.0
    else:
        metrics["volume_ratio_1h"] = 1.0

    if klines_4h is not None and "vol_ratio" in klines_4h.columns:
        latest = klines_4h["vol_ratio"].iloc[-1]
        metrics["volume_ratio_4h"] = round(float(latest), 3) if not pd.isna(latest) else 1.0
    else:
        metrics["volume_ratio_4h"] = 1.0

    # --- Volume trend from 1H ---
    if klines_1h is not None and "vol_trend" in klines_1h.columns:
        vt = klines_1h["vol_trend"].iloc[-1]
        metrics["volume_trend_1h"] = "expanding" if vt > 0 else "contracting"
    else:
        metrics["volume_trend_1h"] = "unknown"

    # --- Taker buy ratio from klines ---
    if klines_1h is not None and "taker_buy_ratio" in klines_1h.columns:
        tbr = klines_1h["taker_buy_ratio"].iloc[-1]
        metrics["taker_buy_ratio_1h"] = round(float(tbr), 3) if not pd.isna(tbr) else 0.5
    else:
        metrics["taker_buy_ratio_1h"] = 0.5

    if klines_4h is not None and "taker_buy_ratio" in klines_4h.columns:
        tbr = klines_4h["taker_buy_ratio"].iloc[-1]
        metrics["taker_buy_ratio_4h"] = round(float(tbr), 3) if not pd.isna(tbr) else 0.5
    else:
        metrics["taker_buy_ratio_4h"] = 0.5

    # --- Taker buy ratio from aggTrades (more granular, last 200 trades) ---
    if agg_trades:
        buyer_vol = sum(t["qty"] for t in agg_trades if not t["is_buyer_maker"])
        total_vol = sum(t["qty"] for t in agg_trades)
        metrics["taker_buy_ratio_agg"] = round(buyer_vol / total_vol, 3) if total_vol > 0 else 0.5
    else:
        metrics["taker_buy_ratio_agg"] = 0.5

    # --- Order book imbalance ---
    if depth:
        bid_vol = sum(q for _, q in depth["bids"])
        ask_vol = sum(q for _, q in depth["asks"])
        total = bid_vol + ask_vol
        if total > 0:
            metrics["ob_bid_pct"] = round(bid_vol / total, 3)
            metrics["ob_ask_pct"] = round(ask_vol / total, 3)
        else:
            metrics["ob_bid_pct"] = 0.5
            metrics["ob_ask_pct"] = 0.5
    else:
        metrics["ob_bid_pct"] = 0.5
        metrics["ob_ask_pct"] = 0.5

    # --- Spread ---
    if book_ticker:
        mid = (book_ticker["bid_price"] + book_ticker["ask_price"]) / 2
        if mid > 0:
            metrics["spread_pct"] = round(
                (book_ticker["ask_price"] - book_ticker["bid_price"]) / mid * 100, 4
            )
        else:
            metrics["spread_pct"] = 0.0
    else:
        metrics["spread_pct"] = 0.0

    # --- 24h metrics ---
    if ticker24h:
        metrics["price_change_24h_pct"] = ticker24h["price_change_pct"]
        metrics["volume_24h"] = ticker24h["volume_24h"]
        metrics["quote_volume_24h"] = ticker24h["quote_volume_24h"]
        metrics["trade_count_24h"] = ticker24h["trade_count_24h"]
        metrics["vwap_24h"] = ticker24h["weighted_avg_price"]
    else:
        metrics["price_change_24h_pct"] = 0.0
        metrics["volume_24h"] = 0.0
        metrics["quote_volume_24h"] = 0.0
        metrics["trade_count_24h"] = 0
        metrics["vwap_24h"] = 0.0

    # --- Large trade detection from aggTrades ---
    if agg_trades:
        large_threshold = 100000  # $100k in quote volume
        large_trades = [
            t for t in agg_trades
            if t["price"] * t["qty"] >= large_threshold
        ]
        metrics["large_trades_count"] = len(large_trades)
        metrics["large_trades_buy"] = sum(
            1 for t in large_trades if not t["is_buyer_maker"]
        )
    else:
        metrics["large_trades_count"] = 0
        metrics["large_trades_buy"] = 0

    # --- Trade count ratio ---
    if klines_1h is not None and "trades_ratio" in klines_1h.columns:
        tr = klines_1h["trades_ratio"].iloc[-1]
        metrics["trade_count_ratio_1h"] = round(float(tr), 3) if not pd.isna(tr) else 1.0
    else:
        metrics["trade_count_ratio_1h"] = 1.0

    return metrics


# ---------------------------------------------------------------------------
# Summarize for Claude Payload
# ---------------------------------------------------------------------------

def summarize_indicators(
    klines: dict[str, pd.DataFrame],
    volume_metrics: dict,
    pair: str,
) -> dict:
    """
    Extract the latest indicator values into a compact, JSON-serializable dict.
    This becomes the Claude payload — compact but complete.
    """
    summary = {"pair": pair}

    def _safe_latest(df: pd.DataFrame, col: str, decimals: int = 4) -> Optional[float]:
        """Safely get the latest non-NaN value from a column."""
        if df is None or col not in df.columns:
            return None
        val = df[col].iloc[-1]
        if pd.isna(val):
            return None
        return round(float(val), decimals)

    def _safe_prev(df: pd.DataFrame, col: str, offset: int = 1, decimals: int = 4) -> Optional[float]:
        """Get a previous value from a column."""
        if df is None or col not in df.columns or len(df) <= offset:
            return None
        val = df[col].iloc[-1 - offset]
        if pd.isna(val):
            return None
        return round(float(val), decimals)

    # Current price
    if "1h" in klines:
        summary["price"] = _safe_latest(klines["1h"], "close", 2)

    # --- Trend indicators ---
    for tf in ["1h", "4h"]:
        df = klines.get(tf)
        if df is None:
            continue
        prefix = tf
        summary[f"ema21_{prefix}"] = _safe_latest(df, "ema21", 2)
        summary[f"ema50_{prefix}"] = _safe_latest(df, "ema50", 2)
        summary[f"ema200_{prefix}"] = _safe_latest(df, "ema200", 2)

    # EMA stack interpretation
    df_1h = klines.get("1h")
    if df_1h is not None:
        e21 = _safe_latest(df_1h, "ema21")
        e50 = _safe_latest(df_1h, "ema50")
        e200 = _safe_latest(df_1h, "ema200")
        if all(v is not None for v in [e21, e50, e200]):
            if e21 > e50 > e200:
                summary["ema_stack_1h"] = "bullish (21>50>200)"
            elif e21 < e50 < e200:
                summary["ema_stack_1h"] = "bearish (21<50<200)"
            else:
                summary["ema_stack_1h"] = "mixed"

    # Price vs EMA50 distance on 4H
    df_4h = klines.get("4h")
    if df_4h is not None:
        price = _safe_latest(df_4h, "close")
        ema50 = _safe_latest(df_4h, "ema50")
        if price and ema50 and ema50 > 0:
            dist = (price - ema50) / ema50 * 100
            summary["price_vs_ema50_4h_pct"] = round(dist, 2)
            summary["price_above_ema50_4h"] = price > ema50

    # Price vs EMA200 on 4H
    if df_4h is not None:
        price = _safe_latest(df_4h, "close")
        ema200 = _safe_latest(df_4h, "ema200")
        if price and ema200:
            summary["price_above_ema200_4h"] = price > ema200

    # --- HH/HL structure on 4H ---
    if df_4h is not None and len(df_4h) >= 20:
        summary["structure_4h"] = _detect_structure(df_4h)

    # --- Daily 200 MA ---
    df_1d = klines.get("1d")
    if df_1d is not None:
        ema200_d = _safe_latest(df_1d, "ema200")
        price_d = _safe_latest(df_1d, "close")
        if ema200_d and price_d:
            summary["above_200d_ma"] = price_d > ema200_d

    # --- Momentum indicators ---
    for tf in ["15m", "1h", "4h"]:
        df = klines.get(tf)
        if df is None:
            continue
        summary[f"rsi_{tf}"] = _safe_latest(df, "rsi14", 1)

    for tf in ["1h", "4h"]:
        df = klines.get(tf)
        if df is None:
            continue
        summary[f"macd_hist_{tf}"] = _safe_latest(df, "macd_hist", 6)
        summary[f"macd_hist_prev_{tf}"] = _safe_prev(df, "macd_hist", 1, 6)

    # MACD direction interpretation
    if df_1h is not None:
        curr = _safe_latest(df_1h, "macd_hist")
        prev = _safe_prev(df_1h, "macd_hist")
        if curr is not None and prev is not None:
            if curr > 0 and curr > prev:
                summary["macd_direction_1h"] = "bullish expanding"
            elif curr > 0 and curr < prev:
                summary["macd_direction_1h"] = "bullish contracting"
            elif curr < 0 and curr > prev:
                summary["macd_direction_1h"] = "bearish contracting (potential cross)"
            else:
                summary["macd_direction_1h"] = "bearish expanding"

    # Stochastic on 4H
    if df_4h is not None:
        summary["stoch_k_4h"] = _safe_latest(df_4h, "stoch_k", 1)
        summary["stoch_d_4h"] = _safe_latest(df_4h, "stoch_d", 1)

    # ROC on 1H
    if df_1h is not None:
        summary["roc10_1h"] = _safe_latest(df_1h, "roc10", 2)

    # --- Volatility indicators ---
    for tf in ["1h", "4h"]:
        df = klines.get(tf)
        if df is None:
            continue
        # ATR uses 6 decimal places — cheap coins (XRP $1.31, ADA $0.25) have
        # ATR values like $0.003 which round to 0.00 at 2 decimal places,
        # causing division-by-zero in the position sizer.
        summary[f"atr14_{tf}"] = _safe_latest(df, "atr14", 6)
        summary[f"atr_pct_{tf}"] = _safe_latest(df, "atr_pct", 4)
        summary[f"bb_pctb_{tf}"] = _safe_latest(df, "bb_pctb", 3)
        summary[f"bb_width_{tf}"] = _safe_latest(df, "bb_width", 6)

    # ATR expansion
    if df_1h is not None:
        summary["atr_expanding_1h"] = bool(
            _safe_latest(df_1h, "atr_expansion") is not None
            and _safe_latest(df_1h, "atr_expansion") > 1.0
        )
    if df_4h is not None:
        atr_curr = _safe_latest(df_4h, "atr14")
        atr_avg = _safe_latest(df_4h, "atr14_sma20")
        if atr_curr and atr_avg:
            summary["atr_above_avg_4h"] = atr_curr > atr_avg

    # Compression ratio
    if df_1h is not None:
        summary["compression_ratio_1h"] = _safe_latest(df_1h, "compression_ratio", 3)

    # BB width vs average (compression detection)
    if df_4h is not None:
        bbw = _safe_latest(df_4h, "bb_width")
        bbw_avg = _safe_latest(df_4h, "bb_width_sma20")
        if bbw and bbw_avg and bbw_avg > 0:
            summary["bb_width_vs_avg_4h"] = round(bbw / bbw_avg, 3)

    # --- Regime detection indicators: ADX + Choppiness Index ---
    # These are the primary regime gates used by regime_detector.py
    # ADX < 20 = ranging  |  ADX > 25 = trending
    # CHOP > 61.8 = consolidating  |  CHOP < 38.2 = trending strongly
    for tf in ["1h", "4h"]:
        df = klines.get(tf)
        if df is None:
            continue
        summary[f"adx14_{tf}"]  = _safe_latest(df, "adx14", 1)
        summary[f"adx_pos_{tf}"] = _safe_latest(df, "adx_pos", 1)
        summary[f"adx_neg_{tf}"] = _safe_latest(df, "adx_neg", 1)
        summary[f"chop14_{tf}"] = _safe_latest(df, "chop14", 1)

    # --- Volume metrics (already computed) ---
    summary.update(volume_metrics)

    # --- Interpreted signals for Claude ---
    summary["interpretations"] = _build_interpretations(summary)

    return summary


def _detect_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """Detect HH/HL or LH/LL structure on a timeframe."""
    if len(df) < lookback:
        return "insufficient data"

    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values

    # Find swing points (simplified: local max/min over 3-bar windows)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        ll = swing_lows[-1] < swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]

        if hh and hl:
            return "HH/HL — uptrend intact"
        elif ll and lh:
            return "LH/LL — downtrend"
        else:
            return "mixed — no clear structure"

    return "insufficient swing points"


def _build_interpretations(summary: dict) -> dict:
    """Build human-readable interpretations of key indicators for Claude."""
    interp = {}

    # RSI interpretation
    rsi = summary.get("rsi_1h")
    if rsi is not None:
        if rsi < 30:
            interp["rsi_1h"] = f"RSI {rsi:.0f} — oversold"
        elif rsi < 50:
            interp["rsi_1h"] = f"RSI {rsi:.0f} — below neutral (pullback territory)"
        elif rsi < 65:
            interp["rsi_1h"] = f"RSI {rsi:.0f} — momentum mode (trending sweet spot)"
        elif rsi < 70:
            interp["rsi_1h"] = f"RSI {rsi:.0f} — approaching overbought"
        else:
            interp["rsi_1h"] = f"RSI {rsi:.0f} — overbought"

    # Volume ratio interpretation
    vr = summary.get("volume_ratio_1h")
    if vr is not None:
        pct = int(vr * 100)
        if vr > 1.5:
            interp["volume"] = f"{pct}% of avg — significant volume spike"
        elif vr > 1.3:
            interp["volume"] = f"{pct}% of avg — above average"
        elif vr > 0.8:
            interp["volume"] = f"{pct}% of avg — normal"
        else:
            interp["volume"] = f"{pct}% of avg — low volume"

    # Taker buy ratio
    tbr = summary.get("taker_buy_ratio_1h")
    if tbr is not None:
        if tbr > 0.60:
            interp["taker_bias"] = f"taker buy {tbr:.2f} — strong buyer aggression"
        elif tbr > 0.55:
            interp["taker_bias"] = f"taker buy {tbr:.2f} — moderate buyer bias"
        elif tbr > 0.45:
            interp["taker_bias"] = f"taker buy {tbr:.2f} — neutral"
        else:
            interp["taker_bias"] = f"taker buy {tbr:.2f} — seller dominated"

    # ADX interpretation (1H)
    adx = summary.get("adx14_1h")
    if adx is not None:
        if adx < 20:
            interp["adx_1h"] = f"ADX {adx:.0f} — non-trending, ranging market (use mean-reversion)"
        elif adx < 25:
            interp["adx_1h"] = f"ADX {adx:.0f} — weak trend developing"
        elif adx < 50:
            interp["adx_1h"] = f"ADX {adx:.0f} — strong trend confirmed (use trend-following)"
        else:
            interp["adx_1h"] = f"ADX {adx:.0f} — very strong trend"

    # ADX interpretation (4H)
    adx4 = summary.get("adx14_4h")
    if adx4 is not None:
        if adx4 < 20:
            interp["adx_4h"] = f"ADX {adx4:.0f} — ranging 4H"
        elif adx4 >= 25:
            interp["adx_4h"] = f"ADX {adx4:.0f} — trending 4H"

    # Choppiness Index interpretation (1H)
    chop = summary.get("chop14_1h")
    if chop is not None:
        if chop > 61.8:
            interp["chop_1h"] = f"CHOP {chop:.1f} — consolidating / ranging (above 61.8 Fibonacci)"
        elif chop < 38.2:
            interp["chop_1h"] = f"CHOP {chop:.1f} — trending strongly (below 38.2 Fibonacci)"
        else:
            interp["chop_1h"] = f"CHOP {chop:.1f} — transitional / indeterminate"

    # Order book
    ob = summary.get("ob_bid_pct")
    if ob is not None:
        bid_pct = int(ob * 100)
        ask_pct = 100 - bid_pct
        if ob > 0.60:
            interp["order_book"] = f"bids {bid_pct}% / asks {ask_pct}% — strong buy pressure"
        elif ob > 0.55:
            interp["order_book"] = f"bids {bid_pct}% / asks {ask_pct}% — moderate buy pressure"
        elif ob > 0.45:
            interp["order_book"] = f"bids {bid_pct}% / asks {ask_pct}% — balanced"
        else:
            interp["order_book"] = f"bids {bid_pct}% / asks {ask_pct}% — sell pressure"

    return interp


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from scanner import BinanceScanner

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def test():
        scanner = BinanceScanner()
        try:
            data = await scanner.scan_all()
            pair = "BTCUSDT"
            pair_data = data[pair]

            # Compute indicators
            klines = compute_indicators(pair_data["klines"])

            # Compute volume metrics
            vol_metrics = compute_volume_metrics(
                klines.get("1h"),
                klines.get("4h"),
                pair_data.get("depth"),
                pair_data.get("book_ticker"),
                pair_data.get("agg_trades"),
                pair_data.get("ticker24h"),
            )

            # Summarize
            summary = summarize_indicators(klines, vol_metrics, pair)

            print(f"\n{'='*60}")
            print(f"INDICATOR SUMMARY: {pair}")
            print(f"{'='*60}")
            for k, v in summary.items():
                if k == "interpretations":
                    print(f"\nInterpretations:")
                    for ik, iv in v.items():
                        print(f"  {ik}: {iv}")
                else:
                    print(f"  {k}: {v}")

            # Verify key indicators exist
            print(f"\n{'='*60}")
            print("VERIFICATION:")
            checks = [
                "price", "ema21_1h", "ema50_4h", "rsi_1h", "rsi_4h",
                "macd_hist_1h", "atr14_1h", "bb_pctb_1h",
                "volume_ratio_1h", "taker_buy_ratio_1h", "ob_bid_pct",
            ]
            for check in checks:
                val = summary.get(check)
                status = "OK" if val is not None else "MISSING"
                print(f"  [{status}] {check} = {val}")

        finally:
            await scanner.close()

    asyncio.run(test())
