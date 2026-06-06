"""
ATLAS Lean v2.0 — Market Regime Detector
Classifies each pair into one of 4 regimes: TRENDING, PULLBACK, RANGING, BREAKOUT.
The regime determines which scoring rules apply, which RSI range is valid,
and what Claude should look for when making its decision.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    regime: str           # "TRENDING" | "PULLBACK" | "RANGING" | "BREAKOUT"
    confidence: float     # 0.0 - 1.0
    details: dict = field(default_factory=dict)


def detect_regime(
    klines: dict[str, pd.DataFrame],
    volume_metrics: dict,
) -> RegimeResult:
    """
    Classify the current market regime for a pair.
    Priority: BREAKOUT > PULLBACK > TRENDING > RANGING (default).
    """
    df_1h = klines.get("1h")
    df_4h = klines.get("4h")

    if df_1h is None or df_4h is None:
        return RegimeResult("RANGING", 0.0, {"reason": "missing critical timeframes"})

    # Check TRENDING conditions (used for PULLBACK macro context only — entries disabled)
    # TRENDING disabled: 0% win rate across all backtests (peak-chasing at exhaustion tops).
    # Still called so PULLBACK can use trending_pass as macro uptrend confirmation.
    trending_pass, trending_score, trending_details = _check_trending(df_1h, df_4h, volume_metrics)

    # Check BREAKOUT (highest priority specific regime)
    breakout_pass, breakout_score, breakout_details = _check_breakout(df_1h, df_4h, volume_metrics)
    if breakout_pass:
        return RegimeResult("BREAKOUT", breakout_score, breakout_details)

    # Check PULLBACK (requires trending macro context)
    pullback_pass, pullback_score, pullback_details = _check_pullback(
        df_1h, df_4h, volume_metrics, trending_pass
    )
    if pullback_pass:
        return RegimeResult("PULLBACK", pullback_score, pullback_details)

    # TRENDING entries intentionally disabled — returns RANGING instead.
    # Backtest shows 0% win rate: entries fire at momentum exhaustion tops.
    # Re-enable only after identifying a structural fix (e.g. volume-confirmed breakout).
    # if trending_pass:
    #     return RegimeResult("TRENDING", trending_score, trending_details)

    # Default: RANGING
    ranging_score, ranging_details = _check_ranging(df_1h, df_4h)
    return RegimeResult("RANGING", ranging_score, ranging_details)


# ---------------------------------------------------------------------------
# TRENDING Detection
# ---------------------------------------------------------------------------

def _check_trending(
    df_1h: pd.DataFrame, df_4h: pd.DataFrame, volume_metrics: dict
) -> tuple[bool, float, dict]:
    """
    TRENDING requires ALL of:
    1. Price above EMA50 AND EMA200 on 4H
    2. EMA21 > EMA50 on 1H
    3. ATR(14) on 4H > 20-period ATR average
    4. Volume on last 4H candle > 20-period volume average
    5. At least 2 of last 3 4H candles are bullish
    6. Higher high on 4H within last 10 candles
    """
    conditions = {}
    checks_passed = 0
    total_checks = 6

    # 1. Price above EMA50 AND EMA200 on 4H
    price_4h = _latest(df_4h, "close")
    ema50_4h = _latest(df_4h, "ema50")
    ema200_4h = _latest(df_4h, "ema200")

    if price_4h and ema50_4h:
        above_ema50 = price_4h > ema50_4h
    else:
        above_ema50 = False

    if price_4h and ema200_4h:
        above_ema200 = price_4h > ema200_4h
    else:
        # If EMA200 not available (insufficient data), relax this condition
        above_ema200 = True if ema200_4h is None else price_4h > ema200_4h

    cond1 = above_ema50 and above_ema200
    conditions["price_above_emas_4h"] = cond1
    if cond1:
        checks_passed += 1

    # 2. EMA21 > EMA50 on 1H
    ema21_1h = _latest(df_1h, "ema21")
    ema50_1h = _latest(df_1h, "ema50")
    cond2 = ema21_1h is not None and ema50_1h is not None and ema21_1h > ema50_1h
    conditions["ema21_above_ema50_1h"] = cond2
    if cond2:
        checks_passed += 1

    # ADX + CHOP regime gates (must both pass for TRENDING to qualify)
    # ADX > 25 confirms a strong directional move; CHOP < 38.2 confirms non-choppy
    adx_4h = _latest(df_4h, "adx14")
    chop_4h = _latest(df_4h, "chop14")
    adx_ok  = adx_4h is not None and adx_4h > 25
    chop_ok = chop_4h is not None and chop_4h < 38.2
    conditions["adx14_4h"] = round(adx_4h, 1) if adx_4h else None
    conditions["chop14_4h"] = round(chop_4h, 1) if chop_4h else None
    if not (adx_ok and chop_ok):
        # Market is not genuinely trending — skip remaining checks
        return False, 0.0, {
            "reason": f"ADX/CHOP regime gate failed: ADX={adx_4h:.1f if adx_4h else 'N/A'} (need >25), CHOP={chop_4h:.1f if chop_4h else 'N/A'} (need <38.2)",
            **conditions,
        }

    # 3. ATR expanding on 4H
    atr_4h = _latest(df_4h, "atr14")
    atr_avg_4h = _latest(df_4h, "atr14_sma20")
    cond3 = atr_4h is not None and atr_avg_4h is not None and atr_4h > atr_avg_4h
    conditions["atr_expanding_4h"] = cond3
    if cond3:
        checks_passed += 1

    # 4. Volume above average on last 4H candle
    vol_ratio_4h = _latest(df_4h, "vol_ratio")
    cond4 = vol_ratio_4h is not None and vol_ratio_4h > 1.0
    conditions["volume_above_avg_4h"] = cond4
    if cond4:
        checks_passed += 1

    # 5. At least 2 of last 3 4H candles are bullish
    if len(df_4h) >= 3:
        last3 = df_4h.tail(3)
        bullish_count = int((last3["close"] > last3["open"]).sum())
        cond5 = bullish_count >= 2
        conditions["bullish_candles_4h"] = f"{bullish_count}/3"
    else:
        cond5 = False
        conditions["bullish_candles_4h"] = "insufficient data"
    if cond5:
        checks_passed += 1

    # 6. Higher high on 4H within last 10 candles
    cond6 = _has_higher_high(df_4h, lookback=10)
    conditions["higher_high_4h"] = cond6
    if cond6:
        checks_passed += 1

    is_trending = checks_passed >= total_checks  # ALL must pass
    confidence = checks_passed / total_checks

    return is_trending, confidence, conditions


# ---------------------------------------------------------------------------
# PULLBACK Detection
# ---------------------------------------------------------------------------

def _check_pullback(
    df_1h: pd.DataFrame, df_4h: pd.DataFrame,
    volume_metrics: dict, is_trending: bool
) -> tuple[bool, float, dict]:
    """
    PULLBACK requires TRENDING macro + ALL of:
    1. TRENDING conditions met on 4H (macro trend intact)
    2. Price within 1.5% of EMA50 on 4H OR at swing low support ±0.8%
    3. RSI(14) on 1H between 28 and 50
    4. Bollinger %B on 1H below 0.35
    5. Volume contracting on pullback candles
    """
    conditions = {}
    checks_passed = 0
    total_checks = 4  # Excluding the trending prerequisite

    # Prerequisite: macro trend must be intact
    # We relax this slightly — require at least 4/6 trending conditions
    # (price can be pulling back below EMA50 temporarily)
    price_4h = _latest(df_4h, "close")
    ema50_4h = _latest(df_4h, "ema50")
    ema200_4h = _latest(df_4h, "ema200")

    # For pullback, trend is intact if price is still above EMA200 and
    # had been trending recently (even if now below EMA50)
    macro_intact = False
    if price_4h and ema200_4h:
        macro_intact = price_4h > ema200_4h
    elif price_4h and ema50_4h:
        # If no EMA200 data, use proximity to EMA50 as proxy
        dist = abs(price_4h - ema50_4h) / ema50_4h * 100
        macro_intact = dist < 3.0  # Within 3% of EMA50

    if not macro_intact:
        return False, 0.0, {"reason": "macro trend not intact"}

    conditions["macro_trend_intact"] = True

    # ADX + CHOP regime gates for PULLBACK
    # ADX < 20 = price is in a non-directional phase (pullback, not free-fall)
    # CHOP > 61.8 = consolidating, not a new trend leg starting
    adx_1h = _latest(df_1h, "adx14")
    chop_1h = _latest(df_1h, "chop14")
    # ADX < 25: not in a strong directional move (pullback = slowing, not free-falling)
    # CHOP > 55: some consolidation (strict 61.8 blocks pullbacks in transition phases)
    adx_ranging  = adx_1h is not None and adx_1h < 25
    chop_ranging = chop_1h is not None and chop_1h > 55
    conditions["adx14_1h"] = round(adx_1h, 1) if adx_1h else None
    conditions["chop14_1h"] = round(chop_1h, 1) if chop_1h else None
    if not (adx_ranging and chop_ranging):
        return False, 0.0, {
            "reason": f"ADX/CHOP gate failed: ADX={adx_1h:.1f if adx_1h else 'N/A'} (need <25), CHOP={chop_1h:.1f if chop_1h else 'N/A'} (need >55)",
            **conditions,
        }

    # 1. Price within 1.5% of EMA50 on 4H OR at swing low support
    if price_4h and ema50_4h and ema50_4h > 0:
        dist_to_ema50 = abs(price_4h - ema50_4h) / ema50_4h * 100
        near_ema50 = dist_to_ema50 <= 1.5

        # Also check for swing low support
        at_support = _near_swing_low(df_4h, price_4h, tolerance_pct=0.8)

        cond1 = near_ema50 or at_support
        conditions["near_ema50_or_support"] = cond1
        conditions["dist_to_ema50_pct"] = round(dist_to_ema50, 2)
    else:
        cond1 = False
        conditions["near_ema50_or_support"] = False
    if cond1:
        checks_passed += 1

    # 2. RSI between 28 and 50 on 1H
    rsi_1h = _latest(df_1h, "rsi14")
    cond2 = rsi_1h is not None and 28 <= rsi_1h <= 50
    conditions["rsi_pullback_range"] = cond2
    conditions["rsi_1h_value"] = round(rsi_1h, 1) if rsi_1h else None
    if cond2:
        checks_passed += 1

    # 3. Bollinger %B below 0.35 on 1H
    bb_pctb = _latest(df_1h, "bb_pctb")
    cond3 = bb_pctb is not None and bb_pctb < 0.35
    conditions["bb_pctb_low"] = cond3
    conditions["bb_pctb_value"] = round(bb_pctb, 3) if bb_pctb else None
    if cond3:
        checks_passed += 1

    # 4. Volume contracting on pullback
    vol_trend = volume_metrics.get("volume_trend_1h", "unknown")
    vol_ratio = volume_metrics.get("volume_ratio_1h", 1.0)
    cond4 = vol_trend == "contracting" or vol_ratio < 1.0
    conditions["volume_contracting"] = cond4
    if cond4:
        checks_passed += 1

    is_pullback = checks_passed >= 3  # Need at least 3/4 specific conditions
    confidence = checks_passed / total_checks

    return is_pullback, confidence, conditions


# ---------------------------------------------------------------------------
# BREAKOUT Detection
# ---------------------------------------------------------------------------

def _check_breakout(
    df_1h: pd.DataFrame, df_4h: pd.DataFrame, volume_metrics: dict
) -> tuple[bool, float, dict]:
    """
    BREAKOUT requires ALL of:
    1. Preceded by compression: range < 0.6x ATR on 1H for >= 6 candles
    2. Current 1H candle closes >0.3% outside compression range
    3. Volume on breakout candle > 150% of 20-period average
    4. Taker buy ratio > 0.60 (long) or < 0.40 (short)
    5. Direction aligns with higher timeframe trend
    """
    conditions = {}
    checks_passed = 0
    total_checks = 5

    # 1. Compression detection
    compression_ratio = _latest(df_1h, "compression_ratio")
    if compression_ratio is not None:
        # Check if compression existed in recent candles (6+ candles with ratio < 0.6)
        if "compression_ratio" in df_1h.columns and len(df_1h) >= 7:
            recent = df_1h["compression_ratio"].iloc[-7:-1]  # Last 6 candles before current
            compressed_count = int((recent < 0.6).sum())
            cond1 = compressed_count >= 4  # At least 4 of 6 compressed
            conditions["compression_detected"] = cond1
            conditions["compressed_candles"] = f"{compressed_count}/6"
        else:
            cond1 = False
            conditions["compression_detected"] = False
    else:
        cond1 = False
        conditions["compression_detected"] = False
    if cond1:
        checks_passed += 1

    # 2. Price closes outside compression range
    if cond1 and len(df_1h) >= 7:
        compression_zone = df_1h.iloc[-7:-1]
        range_high = compression_zone["high"].max()
        range_low = compression_zone["low"].min()
        current_close = _latest(df_1h, "close")

        if current_close and range_high and range_low:
            range_size = range_high - range_low
            mid = (range_high + range_low) / 2
            breakout_threshold = 0.003 * mid  # 0.3%

            broke_above = current_close > range_high + breakout_threshold
            broke_below = current_close < range_low - breakout_threshold
            cond2 = broke_above or broke_below
            conditions["broke_range"] = cond2
            if broke_above:
                conditions["breakout_direction"] = "LONG"
            elif broke_below:
                conditions["breakout_direction"] = "SHORT"
        else:
            cond2 = False
            conditions["broke_range"] = False
    else:
        cond2 = False
        conditions["broke_range"] = False
    if cond2:
        checks_passed += 1

    # 3. Volume spike > 150%
    vol_ratio = _latest(df_1h, "vol_ratio")
    cond3 = vol_ratio is not None and vol_ratio > 1.5
    conditions["volume_spike"] = cond3
    conditions["vol_ratio"] = round(vol_ratio, 3) if vol_ratio else None
    if cond3:
        checks_passed += 1

    # 4. Taker buy ratio confirmation
    tbr = volume_metrics.get("taker_buy_ratio_1h", 0.5)
    direction = conditions.get("breakout_direction", "LONG")
    if direction == "LONG":
        cond4 = tbr > 0.60
    else:
        cond4 = tbr < 0.40
    conditions["taker_confirmation"] = cond4
    conditions["taker_buy_ratio"] = tbr
    if cond4:
        checks_passed += 1

    # 5. Aligns with higher timeframe trend
    price_4h = _latest(df_4h, "close")
    ema50_4h = _latest(df_4h, "ema50")
    if price_4h and ema50_4h:
        if direction == "LONG":
            cond5 = price_4h > ema50_4h
        else:
            cond5 = price_4h < ema50_4h
    else:
        cond5 = False
    conditions["trend_alignment"] = cond5
    if cond5:
        checks_passed += 1

    is_breakout = checks_passed >= 4  # Need 4/5 (compression + 3 confirmations)
    confidence = checks_passed / total_checks

    return is_breakout, confidence, conditions


# ---------------------------------------------------------------------------
# RANGING Detection (explicit checks + default)
# ---------------------------------------------------------------------------

def _check_ranging(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> tuple[float, dict]:
    """
    RANGING characteristics:
    - BB width < 50% of its 20-period average on 4H
    - ATR contracting
    - EMAs within 1% of each other on 1H
    - No clear HH/HL or LL/LH structure
    """
    conditions = {}
    ranging_signals = 0

    # BB width compression
    bb_width = _latest(df_4h, "bb_width")
    bb_width_avg = _latest(df_4h, "bb_width_sma20")
    if bb_width and bb_width_avg and bb_width_avg > 0:
        ratio = bb_width / bb_width_avg
        if ratio < 0.5:
            ranging_signals += 1
        conditions["bb_compression"] = round(ratio, 3)

    # ATR contracting
    atr = _latest(df_4h, "atr14")
    atr_avg = _latest(df_4h, "atr14_sma20")
    if atr and atr_avg:
        conditions["atr_contracting"] = atr < atr_avg
        if atr < atr_avg:
            ranging_signals += 1

    # EMAs close together on 1H
    ema21 = _latest(df_1h, "ema21")
    ema50 = _latest(df_1h, "ema50")
    if ema21 and ema50 and ema50 > 0:
        ema_diff = abs(ema21 - ema50) / ema50 * 100
        conditions["ema_convergence_pct"] = round(ema_diff, 2)
        if ema_diff < 1.0:
            ranging_signals += 1

    # No clear structure
    structure = "mixed"
    if len(df_4h) >= 20:
        structure = _detect_structure_type(df_4h)
    conditions["structure"] = structure
    if structure == "mixed":
        ranging_signals += 1

    confidence = ranging_signals / 4
    conditions["ranging_signals"] = f"{ranging_signals}/4"

    return confidence, conditions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(df: pd.DataFrame, col: str) -> Optional[float]:
    """Safely get the latest value from a column."""
    if df is None or col not in df.columns or df.empty:
        return None
    val = df[col].iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def _has_higher_high(df: pd.DataFrame, lookback: int = 10) -> bool:
    """Check if there's a higher high in the last N candles."""
    if len(df) < lookback + 5:
        return False
    recent = df.tail(lookback)
    prior = df.iloc[-(lookback + 5):-lookback]
    if prior.empty:
        return False
    return float(recent["high"].max()) > float(prior["high"].max())


def _near_swing_low(df: pd.DataFrame, current_price: float, tolerance_pct: float = 0.8) -> bool:
    """Check if price is near a recent swing low (support)."""
    if len(df) < 10:
        return False

    lows = df["low"].values[-20:]
    swing_lows = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])

    if not swing_lows:
        return False

    # Check if current price is within tolerance of any swing low
    for sl in swing_lows:
        if sl > 0:
            dist = abs(current_price - sl) / sl * 100
            if dist <= tolerance_pct:
                return True
    return False


def _detect_structure_type(df: pd.DataFrame) -> str:
    """Detect HH/HL or LH/LL structure."""
    if len(df) < 20:
        return "insufficient data"

    recent = df.tail(20)
    highs = recent["high"].values
    lows = recent["low"].values

    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append(lows[i])

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        if hh and hl:
            return "uptrend"
        ll = swing_lows[-1] < swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]
        if ll and lh:
            return "downtrend"

    return "mixed"


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from scanner import BinanceScanner
    from indicators import compute_indicators, compute_volume_metrics

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def test():
        scanner = BinanceScanner()
        try:
            data = await scanner.scan_all()

            print(f"\n{'='*60}")
            print("REGIME DETECTION — ALL PAIRS")
            print(f"{'='*60}\n")

            for pair in sorted(data.keys()):
                pair_data = data[pair]
                klines = compute_indicators(pair_data["klines"])
                vol_metrics = compute_volume_metrics(
                    klines.get("1h"), klines.get("4h"),
                    pair_data.get("depth"), pair_data.get("book_ticker"),
                    pair_data.get("agg_trades"), pair_data.get("ticker24h"),
                )

                result = detect_regime(klines, vol_metrics)
                price = klines["1h"]["close"].iloc[-1] if "1h" in klines else "N/A"

                print(f"{pair} @ {price}")
                print(f"  Regime: {result.regime} (confidence: {result.confidence:.1%})")
                for k, v in result.details.items():
                    print(f"    {k}: {v}")
                print()

        finally:
            await scanner.close()

    asyncio.run(test())
