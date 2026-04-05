# ATLAS Lean — Regime Definitions & Thresholds

## Overview
The ATLAS Lean strategy classifies market conditions into **3 regimes**: PULLBACK, TRENDING, and RANGING.
Each regime has specific entry conditions verified across 1H and 4H timeframes.

---

## PULLBACK Regime
**Definition**: Buying dips in a confirmed uptrend

### Conditions (ALL must be true):

#### Price Structure (1H)
- `close > ema200` — Price above 200-period EMA
- `abs(close - ema50) / ema50 < 0.025` — Price within 2.5% of 50-period EMA (pullback zone)
- `rsi >= 28 AND rsi <= 55` — RSI in pullback mode (not overbought, not panic)
- `bb_pctb < 0.40` — Bollinger Band %B below 40% (closer to lower band)

#### Macro Context (4H)
- `price_4h > ema50_4h` — 4H price above 50-period EMA
- `ema50_4h > ema200_4h` — 4H EMA50 > EMA200 (macro uptrend confirmation)

#### Regime Strength (1H & 4H)
- `adx14_1h < 25` — ADX below 25 (not in a strong directional move; price is slowing/consolidating)
- `chop14_1h > 55` — Choppiness Index above 55 (market consolidating, not trending)

### Summary
```
PULLBACK = price_near_ema50 AND macro_uptrend AND rsi_pullback AND bb_lower AND adx_ranging AND chop_consolidating
```

---

## TRENDING Regime
**Definition**: Riding strong directional moves with momentum confirmation

### Conditions (ALL must be true):

#### Price Structure (1H)
- `close > ema50` — Price above 50-period EMA
- `close > ema200` — Price above 200-period EMA (both uptrends aligned)
- `rsi >= 50 AND rsi <= 75` — RSI in momentum mode (bullish, not yet exhausted)

#### Volatility Expansion (1H)
- `atr_14 >= atr_sma_20` — Current ATR >= 20-period SMA of ATR (volatility expanding, not contracting)

#### Regime Strength (4H)
- `adx14_4h > 25` — ADX above 25 (confirmed directional move, strong trend)
- `chop14_4h < 38.2` — Choppiness Index below 38.2 (strongly trending, not choppy)

### Summary
```
TRENDING = price_above_both_emas AND rsi_momentum AND atr_expanding AND adx_strong AND chop_trending
```

---

## RANGING Regime
**Definition**: Default state; no valid setup

### Condition:
```
RANGING = NOT(PULLBACK) AND NOT(TRENDING)
```

This catches:
- Consolidation without pullback setup
- Downtrends
- Mixed/choppy conditions
- Low momentum

---

## Safety Filters (Applied Before Entry)

### Daily Macro Filter
**Purpose**: Prevent entries during multi-day downtrends

- `daily_close > daily_ema21` — Daily price above 21-period EMA
- `daily_ema21 > daily_ema50` — Daily EMA21 above EMA50 (macro uptrend)

**Effect**: Blocks ~96% of entries during Mar 5–Apr 4 downtrend periods

### Score Threshold Filter
**Purpose**: Ensure sufficient signal quality

| Regime | Minimum Score | Signals Evaluated |
|---|---|---|
| PULLBACK | 4/8 | trend_align + ema_stack + rsi_mode + macd + structure + volume + taker_buy + regime_fit |
| TRENDING | 5/8 | (same 8 signals) |

Note: Order book signal (9th) excluded (historical data unavailable)

### Re-entry Cooldown
- **12 hours**: After any loss, skip re-entry for same pair for 12 hours

### Time Exit
- **48 hours**: Any open trade closed at market price after 48 hours (TTL)

---

## Indicator Reference

### Moving Averages
- **EMA20** (Daily): ~= EMA21; trend baseline
- **EMA50** (1H & 4H): Mid-term trend; pullback support
- **EMA200** (1H & 4H): Long-term trend; bounce confirmation

### Momentum
- **RSI(14)**:
  - 28–55: Pullback zone (dip buying safe)
  - 50–75: Trending zone (momentum riding)
  - <28: Oversold (panic)
  - >75: Overbought (exhaustion)

### Volatility
- **ATR(14)**: Average True Range; position sizing & stop distance
- **ATR SMA(20)**: 20-bar average of ATR; volatility baseline
- **BB%B(20,2)**: Bollinger Band position
  - <0.40: Near lower band (cheap)
  - >0.60: Near upper band (extended)

### Trend Strength
- **ADX(14)**: Average Directional Index
  - <20: Weak / ranging
  - 20–25: Transition
  - >25: Strong trend / directional move
  - >40: Very strong

- **CHOP(14)**: Choppiness Index (log scale, Fibonacci reference)
  - <38.2: Strongly trending (buy pullbacks, ride trends)
  - 38.2–61.8: Transition
  - >61.8: Choppy / consolidating (avoid, or grid trade)
  - Reference: Fibonacci levels at 38.2 and 61.8

---

## Entry Targets & Risk Management

### Position Sizing (SPOT mode)
- **Risk per trade**: 2% of account
- **Max allocation**: 30% of account per trade
- **Min stop distance**: 3% (floor for low-volatility pairs)

### Take Profit Structure
- **TP1**: Entry + (SL distance × 1.5R) — close 100% here
- **TP2**: Entry + (SL distance × 3.0R) — rarely hit (48h time exit first)

### Stop Loss
- **Initial SL**: Entry - (ATR × 1.5)
- Adjusted upward as trade progresses (break-even, then profit)

---

## Visual Summary

```
                    TRENDING (Strong Momentum)
                    ├─ Price > EMA50 & EMA200
                    ├─ RSI 50–75 (momentum)
                    ├─ ADX > 25 (strong trend)
                    ├─ CHOP < 38.2 (not choppy)
                    └─ ATR >= ATR_SMA (expanding)

        ╔═══════════════════════════════════════╗
        ║  EMA200  EMA50  Price  TA Indicators  ║
        ╚═══════════════════════════════════════╝

                    PULLBACK (Dip Buying)
                    ├─ Price near EMA50 (±2.5%)
                    ├─ 4H EMA50 > EMA200 (macro up)
                    ├─ RSI 28–55 (pullback)
                    ├─ BB%B < 0.40 (lower band)
                    ├─ ADX < 25 (consolidating)
                    └─ CHOP > 55 (choppy)

                    RANGING (No Setup)
                    └─ All other conditions
```

---

## TradingView Implementation

See `regime_detector.pine` for the full Pine Script that plots:
- PULLBACK zones (light green background)
- TRENDING zones (light blue background)
- RANGING zones (light gray background)
- Key indicator levels (ADX, CHOP, RSI bands, BB%B)
- Entry signals (↑ PULLBACK, ↗ TRENDING)

---

## Backtest Results (90-day, Jan–Apr 2026)

| Metric | PULLBACK | TRENDING | Combined |
|---|---|---|---|
| Trades | 4 | 2 | 6 |
| Win Rate | 50% | 0% | 33% |
| Avg Winner | +0.56R | -0.27R | +0.39R |
| Avg Loser | -1.0R | -0.27R | -0.64R |
| Expectancy | -0.22R | -0.27R | -0.293R |

**Note**: Jan 14–19 window only (all other periods blocked by daily filter). Market was in downtrend Mar 5–Apr 4.

---

## Next Steps

1. **Visualize on TradingView**: Load `regime_detector.pine` script
2. **Validate thresholds**: Compare backtest entries with TV chart visually
3. **Test on different pairs**: SOL, BNB, ETH use same logic
4. **Consider add-ons**: Grid trading for RANGING periods, trailing stops for TRENDING
