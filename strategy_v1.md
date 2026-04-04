# ATLAS Strategy v1.0
# Created: 2026-03-26  |  Status: ACTIVE
# -----------------------------------------

## REGIME DETECTION RULES

### TRENDING (require ALL of):
- Price above EMA50 AND EMA200 on 4H
- EMA21 > EMA50 on 1H
- ATR(14) on 4H > 20-period ATR average
- Volume on last 4H candle > 20-period volume average
- At least 2 of last 3 4H candles are bullish
- Price made a higher high on 4H within last 10 candles

### PULLBACK (require TRENDING + ALL of):
- Price within 1.5% of EMA50 on 4H OR at swing low +/-0.8%
- RSI(14) on 1H between 28 and 50
- Bollinger %B on 1H below 0.35
- Volume contracting on retracement candles

### BREAKOUT (require ALL of):
- Preceded by compression: range < 0.6x ATR on 1H for >=6 candles
- Price closes >0.3% outside compression range
- Breakout candle volume > 150% of 20-period average
- Taker buy ratio on breakout candle > 0.60 (long) or < 0.40 (short)
- Direction aligns with 4H/Daily trend

### RANGING: default if none of above match

## SIGNAL SCORING RULES

Signal 1 -- Trend: price above EMA50 AND EMA200 on 4H (+1)
Signal 2 -- EMA stack: 21>50>200 on 1H (+1)
Signal 3 -- RSI: TRENDING=50-65, PULLBACK=30-50, BREAKOUT=any (+1)
Signal 4 -- MACD: histogram positive or crossing up on 1H (+1)
Signal 5 -- Structure: within 1.0% of support/resistance/EMA50 (+1)
Signal 6 -- Volume ratio: current > 130% of 20-period avg (+1)
Signal 7 -- Taker buy ratio: >0.55 (long) or <0.45 (short) (+1)
Signal 8 -- Order book: imbalance >58/42 in direction (+1)
Signal 9 -- Regime fit: not RANGING (+1)

## THRESHOLDS

TRENDING threshold:  6 of 9
PULLBACK threshold:  5 of 9
BREAKOUT threshold:  6 of 9
RANGING threshold:   never trigger Claude

## RISK RULES (ENFORCED IN PYTHON -- NOT OVERRIDABLE)

Max risk per trade:      2.0% of account
Max simultaneous trades: 3
Daily drawdown halt:     4% of account
Min R:R ratio:           1.5
Stop loss:               structural (no fixed pips)

## CLAUDE GUIDANCE

When evaluating a TRENDING setup:
- Volume expansion on the entry candle is the strongest confirmation.
- Taker buy ratio >0.60 is a high-conviction signal.
- Prefer entries where MACD is positive on both 1H and 4H.

When evaluating a PULLBACK setup:
- Volume contracting on the pullback is POSITIVE (healthy dip).
- Stochastic oversold on 4H + RSI 30-45 on 1H = high probability.
- MACD may be negative on 1H -- this is acceptable for PULLBACK mode.
- Key question: is the 4H trend still intact above EMA50?

When evaluating a BREAKOUT setup:
- Volume is the most important signal -- no volume = fake breakout.
- Taker buy ratio must be strong (>0.60) -- aggressive buyers driving it.
- Confirm the breakout level has significance (previous S/R, range high).
- Be cautious of breakouts into overhead resistance within 1.5%.
