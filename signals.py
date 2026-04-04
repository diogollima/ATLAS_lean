"""
ATLAS Lean v2.0 — Signal Scoring Engine
Evaluates 9 binary signals per pair per cycle. Each signal is 0 or 1.
The regime classifier determines which signal rules apply.
Score range: 0-9. Claude call threshold is regime-dependent.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    pair: str
    regime: str
    score: int
    signals: dict[str, bool]
    details: dict[str, str] = field(default_factory=dict)


def score_pair(
    pair: str,
    klines: dict[str, pd.DataFrame],
    volume_metrics: dict,
    regime: str,
) -> SignalResult:
    """
    Score a pair on the 9-signal model.
    Returns SignalResult with score 0-9 and individual signal states.
    """
    signals = {}
    details = {}

    # Signal 1: Trend alignment
    s1, d1 = _sig_trend_alignment(klines, regime)
    signals["trend_alignment"] = s1
    details["trend_alignment"] = d1

    # Signal 2: EMA stack
    s2, d2 = _sig_ema_stack(klines, regime)
    signals["ema_stack"] = s2
    details["ema_stack"] = d2

    # Signal 3: RSI mode
    s3, d3 = _sig_rsi_mode(klines, regime)
    signals["rsi_mode"] = s3
    details["rsi_mode"] = d3

    # Signal 4: MACD
    s4, d4 = _sig_macd(klines, regime)
    signals["macd"] = s4
    details["macd"] = d4

    # Signal 5: Structure level
    s5, d5 = _sig_structure_level(klines, regime)
    signals["structure_level"] = s5
    details["structure_level"] = d5

    # Signal 6: Volume ratio
    s6, d6 = _sig_volume_ratio(klines, volume_metrics, regime)
    signals["volume_ratio"] = s6
    details["volume_ratio"] = d6

    # Signal 7: Taker buy bias
    s7, d7 = _sig_taker_buy_bias(volume_metrics, regime)
    signals["taker_buy_bias"] = s7
    details["taker_buy_bias"] = d7

    # Signal 8: Order book imbalance
    s8, d8 = _sig_orderbook_imbalance(volume_metrics, regime)
    signals["order_book"] = s8
    details["order_book"] = d8

    # Signal 9: Regime fit
    s9, d9 = _sig_regime_fit(regime)
    signals["regime_fit"] = s9
    details["regime_fit"] = d9

    score = sum(1 for v in signals.values() if v)

    return SignalResult(
        pair=pair,
        regime=regime,
        score=score,
        signals=signals,
        details=details,
    )


# ---------------------------------------------------------------------------
# Individual Signal Functions
# ---------------------------------------------------------------------------

def _sig_trend_alignment(klines: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 1: Trend alignment
    TRENDING/PULLBACK: Price above EMA50 AND EMA200 on 4H
    BREAKOUT: Price above EMA50 on 4H (direction of trend)
    """
    df_4h = klines.get("4h")
    if df_4h is None:
        return False, "no 4H data"

    price = _latest(df_4h, "close")
    ema50 = _latest(df_4h, "ema50")
    ema200 = _latest(df_4h, "ema200")

    if price is None or ema50 is None:
        return False, "missing price/EMA data"

    above_ema50 = price > ema50

    if regime == "BREAKOUT":
        if above_ema50:
            return True, f"price {price:.2f} above EMA50 {ema50:.2f}"
        return False, f"price {price:.2f} below EMA50 {ema50:.2f}"

    # TRENDING/PULLBACK: need both EMA50 and EMA200
    if ema200 is None:
        # If EMA200 unavailable, only check EMA50
        if above_ema50:
            return True, f"price above EMA50 (EMA200 N/A)"
        return False, f"price below EMA50"

    above_ema200 = price > ema200
    if above_ema50 and above_ema200:
        return True, f"price above both EMA50 and EMA200 on 4H"
    return False, f"price not above both EMAs on 4H"


def _sig_ema_stack(klines: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 2: EMA stack
    TRENDING/PULLBACK: EMA21 > EMA50 > EMA200 on 1H (fully aligned)
    BREAKOUT: N/A — use price vs EMA50 only (always pass if price > EMA50)
    """
    if regime == "BREAKOUT":
        df_4h = klines.get("4h")
        if df_4h is not None:
            price = _latest(df_4h, "close")
            ema50 = _latest(df_4h, "ema50")
            if price and ema50 and price > ema50:
                return True, "breakout: price above EMA50"
        return False, "breakout: price below EMA50"

    df_1h = klines.get("1h")
    if df_1h is None:
        return False, "no 1H data"

    ema21 = _latest(df_1h, "ema21")
    ema50 = _latest(df_1h, "ema50")
    ema200 = _latest(df_1h, "ema200")

    if ema21 is None or ema50 is None:
        return False, "missing EMA data on 1H"

    if ema200 is None:
        # If EMA200 not available, check 21>50 only
        if ema21 > ema50:
            return True, f"EMA21>EMA50 on 1H (EMA200 N/A)"
        return False, f"EMA21<EMA50 on 1H"

    if ema21 > ema50 > ema200:
        return True, "full bullish stack: 21>50>200 on 1H"
    return False, f"stack not aligned: 21={ema21:.0f} 50={ema50:.0f} 200={ema200:.0f}"


def _sig_rsi_mode(klines: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 3: RSI mode
    TRENDING: RSI 50-65 on 1H
    PULLBACK: RSI 30-50 on 1H
    BREAKOUT: any RSI (always True)
    RANGING: always False
    """
    if regime == "RANGING":
        return False, "RANGING — no entries"

    if regime == "BREAKOUT":
        return True, "BREAKOUT — any RSI accepted"

    df_1h = klines.get("1h")
    if df_1h is None:
        return False, "no 1H data"

    rsi = _latest(df_1h, "rsi14")
    if rsi is None:
        return False, "RSI unavailable"

    if regime == "TRENDING":
        if 50 <= rsi <= 65:
            return True, f"RSI {rsi:.1f} in TRENDING range (50-65)"
        return False, f"RSI {rsi:.1f} outside TRENDING range (50-65)"

    if regime == "PULLBACK":
        if 30 <= rsi <= 50:
            return True, f"RSI {rsi:.1f} in PULLBACK range (30-50)"
        return False, f"RSI {rsi:.1f} outside PULLBACK range (30-50)"

    return False, f"unknown regime: {regime}"


def _sig_macd(klines: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 4: MACD
    TRENDING/PULLBACK: Histogram positive OR crossing up on 1H
    BREAKOUT: MACD positive on breakout candle
    """
    df_1h = klines.get("1h")
    if df_1h is None:
        return False, "no 1H data"

    hist = _latest(df_1h, "macd_hist")
    prev_hist = _prev(df_1h, "macd_hist")

    if hist is None:
        return False, "MACD unavailable"

    if regime == "BREAKOUT":
        if hist > 0:
            return True, f"MACD histogram positive ({hist:.4f})"
        return False, f"MACD histogram negative ({hist:.4f})"

    # TRENDING/PULLBACK: positive OR crossing up
    if hist > 0:
        return True, f"MACD histogram positive ({hist:.6f})"

    if prev_hist is not None and prev_hist < 0 and hist > prev_hist:
        return True, f"MACD crossing up ({prev_hist:.6f} -> {hist:.6f})"

    return False, f"MACD negative and not crossing ({hist:.6f})"


def _sig_structure_level(klines: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 5: Structure level
    TRENDING/PULLBACK: Price within 1.0% of identified S/R level or EMA50
    BREAKOUT: Price just closed outside compression range
    """
    if regime == "BREAKOUT":
        df_1h = klines.get("1h")
        if df_1h is not None and "compression_ratio" in df_1h.columns:
            cr = _latest(df_1h, "compression_ratio")
            if cr is not None and cr > 1.0:
                return True, f"price outside compression range (ratio: {cr:.2f})"
        return False, "no breakout from compression detected"

    # TRENDING/PULLBACK: near S/R or EMA50
    df_1h = klines.get("1h")
    df_4h = klines.get("4h")
    if df_1h is None:
        return False, "no 1H data"

    price = _latest(df_1h, "close")
    if price is None:
        return False, "no price data"

    # Check proximity to EMA50 on 4H
    ema50_4h = _latest(df_4h, "ema50") if df_4h is not None else None
    if ema50_4h and ema50_4h > 0:
        dist = abs(price - ema50_4h) / ema50_4h * 100
        if dist <= 1.0:
            return True, f"within {dist:.2f}% of EMA50 on 4H"

    # Check proximity to EMA50 on 1H
    ema50_1h = _latest(df_1h, "ema50")
    if ema50_1h and ema50_1h > 0:
        dist = abs(price - ema50_1h) / ema50_1h * 100
        if dist <= 1.0:
            return True, f"within {dist:.2f}% of EMA50 on 1H"

    # Check proximity to recent swing highs/lows (S/R)
    if len(df_1h) >= 20:
        recent = df_1h.tail(20)
        highs = recent["high"].values
        lows = recent["low"].values

        for i in range(2, len(highs) - 2):
            # Swing high as resistance
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                dist = abs(price - highs[i]) / highs[i] * 100
                if dist <= 1.0:
                    return True, f"within {dist:.2f}% of swing high {highs[i]:.2f}"
            # Swing low as support
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                dist = abs(price - lows[i]) / lows[i] * 100
                if dist <= 1.0:
                    return True, f"within {dist:.2f}% of swing low {lows[i]:.2f}"

    return False, "not near any structure level"


def _sig_volume_ratio(klines: dict, volume_metrics: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 6: Volume ratio
    Standard: current 1H volume > 130% of 20-period average
    Breakout: breakout candle volume > 150% of average
    """
    vol_ratio = volume_metrics.get("volume_ratio_1h", 0)

    if regime == "BREAKOUT":
        threshold = config.VOLUME_RATIO_BREAKOUT
        label = "150%"
    else:
        threshold = config.VOLUME_RATIO_THRESHOLD
        label = "130%"

    pct = int(vol_ratio * 100)
    if vol_ratio >= threshold:
        return True, f"volume {pct}% of avg (above {label} threshold)"
    return False, f"volume {pct}% of avg (below {label} threshold)"


def _sig_taker_buy_bias(volume_metrics: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 7: Taker buy bias
    Standard: taker buy ratio > 0.55 (long) or < 0.45 (short)
    Breakout: > 0.60 (long) or < 0.40 (short)
    """
    tbr = volume_metrics.get("taker_buy_ratio_1h", 0.5)

    if regime == "BREAKOUT":
        long_thresh = config.TAKER_BIAS_BREAKOUT_LONG
        short_thresh = config.TAKER_BIAS_BREAKOUT_SHORT
    else:
        long_thresh = config.TAKER_BIAS_LONG
        short_thresh = config.TAKER_BIAS_SHORT

    if tbr > long_thresh:
        return True, f"taker buy {tbr:.3f} > {long_thresh} (bullish)"
    if tbr < short_thresh:
        return True, f"taker buy {tbr:.3f} < {short_thresh} (bearish)"
    return False, f"taker buy {tbr:.3f} — neutral"


def _sig_orderbook_imbalance(volume_metrics: dict, regime: str) -> tuple[bool, str]:
    """
    Signal 8: Order book imbalance
    Standard: bid/ask imbalance > 58/42 in trade direction
    Breakout: > 60/40
    """
    ob_bid = volume_metrics.get("ob_bid_pct", 0.5)

    if regime == "BREAKOUT":
        threshold = config.OB_IMBALANCE_BREAKOUT
    else:
        threshold = config.OB_IMBALANCE_THRESHOLD

    bid_pct = int(ob_bid * 100)
    ask_pct = 100 - bid_pct

    # Check for either direction
    if ob_bid >= threshold:
        return True, f"bids {bid_pct}%/asks {ask_pct}% — buy pressure"
    if (1 - ob_bid) >= threshold:
        return True, f"bids {bid_pct}%/asks {ask_pct}% — sell pressure"
    return False, f"bids {bid_pct}%/asks {ask_pct}% — balanced"


def _sig_regime_fit(regime: str) -> tuple[bool, str]:
    """
    Signal 9: Regime fit
    True if regime is not RANGING.
    For BREAKOUT: must be preceded by compression (handled by regime detector).
    """
    if regime == "RANGING":
        return False, "RANGING — no directional trades"
    return True, f"{regime} — regime supports entry"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(df: Optional[pd.DataFrame], col: str) -> Optional[float]:
    if df is None or col not in df.columns or df.empty:
        return None
    val = df[col].iloc[-1]
    return None if pd.isna(val) else float(val)


def _prev(df: Optional[pd.DataFrame], col: str, offset: int = 1) -> Optional[float]:
    if df is None or col not in df.columns or len(df) <= offset:
        return None
    val = df[col].iloc[-1 - offset]
    return None if pd.isna(val) else float(val)


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from scanner import BinanceScanner
    from indicators import compute_indicators, compute_volume_metrics
    from regime_detector import detect_regime

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def test():
        scanner = BinanceScanner()
        try:
            data = await scanner.scan_all()

            print(f"\n{'='*60}")
            print("SIGNAL SCORING — ALL PAIRS")
            print(f"{'='*60}\n")

            for pair in sorted(data.keys()):
                pair_data = data[pair]
                klines = compute_indicators(pair_data["klines"])
                vol_metrics = compute_volume_metrics(
                    klines.get("1h"), klines.get("4h"),
                    pair_data.get("depth"), pair_data.get("book_ticker"),
                    pair_data.get("agg_trades"), pair_data.get("ticker24h"),
                )

                regime = detect_regime(klines, vol_metrics)
                result = score_pair(pair, klines, vol_metrics, regime.regime)

                price = klines["1h"]["close"].iloc[-1] if "1h" in klines else "N/A"
                threshold = config.get_score_threshold(regime.regime)

                trigger = "TRIGGER CLAUDE" if result.score >= threshold else "below threshold"

                print(f"{pair} @ {price}")
                print(f"  Regime: {regime.regime} | Score: {result.score}/9 | "
                      f"Threshold: {threshold} | {trigger}")

                for sig_name, sig_pass in result.signals.items():
                    status = "PASS (+1)" if sig_pass else "FAIL ( 0)"
                    detail = result.details.get(sig_name, "")
                    print(f"    {status} {sig_name}: {detail}")
                print()

        finally:
            await scanner.close()

    asyncio.run(test())
