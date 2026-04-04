"""
ATLAS Lean v2.0 — 7-Day Walk-Forward Backtest
Fetches historical klines for all 6 pairs, replays every 1H candle
as if it were "now", runs the full signal pipeline, and simulates
all qualifying entries forward to SL/TP resolution.

Assumptions:
- Entry at OPEN of the candle following the signal candle
- SL and TP calculated from ATR at signal time (× 1.5 stop, 1.5R/3.0R targets)
- Max 3 simultaneous trades, 1 per pair at a time
- Long-only; pair skipped if price < EMA200 × 0.97
- Taker buy ratio and order book: proxied from volume data (noted as limitation)
- Risk: 2% of account per trade (SPOT mode: capped at 30% allocation)
- Account: $1,000 USDT

Output:
- Per-trade table with entry/SL/TP/outcome/PnL
- Equity curve
- Win rate, expectancy, max drawdown
- Full report sent to Telegram
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd

import config

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="ATLAS Lean Walk-Forward Backtest")
_parser.add_argument("--days", type=int, default=7,
                     help="Number of days to backtest (default: 7)")
_args = _parser.parse_args()

# ── Settings ─────────────────────────────────────────────────────────────────
ACCOUNT_USDT   = 1000.0
RISK_PCT       = 0.02          # 2% per trade
SL_ATR_MULT    = 1.5
TP1_R          = 1.5
TP2_R          = 3.0
MAX_TRADES     = 3
SPOT_ALLOC_PCT = 0.30          # max 30% of account per trade (SPOT)
SPOT_MIN_STOP  = 0.03          # min 3% stop distance (SPOT floor)
LOOKBACK_DAYS  = _args.days
EVAL_INTERVAL  = "1h"          # evaluate at each 1H candle close
FWD_CANDLES    = 96            # 96 × 15m = 24h forward simulation per trade

PAIRS = config.PAIRS

# ── Telegram ──────────────────────────────────────────────────────────────────
async def tg(text: str) -> None:
    token = config.TELEGRAM_BOT_TOKEN
    chat  = config.TELEGRAM_CHAT_ID
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    async with httpx.AsyncClient(timeout=15) as c:
        for chunk in chunks:
            try:
                await c.post(url, json={"chat_id": chat, "text": chunk, "parse_mode": "HTML"})
            except Exception:
                pass


# ── Binance historical fetch ──────────────────────────────────────────────────
async def fetch_hist(pair: str, interval: str, days: int,
                     extra_candles: int = 0) -> pd.DataFrame:
    """Fetch N days of historical klines + extra leading candles for warmup.
    Paginates automatically when total bars exceed Binance's 1000-bar limit."""
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
            # Paginate: fetch in 1000-bar chunks going backwards from now
            all_data = []
            end_time = None
            remaining = total_needed
            while remaining > 0:
                page_limit = min(remaining, PAGE)
                params: dict = {"symbol": pair, "interval": interval, "limit": page_limit}
                if end_time is not None:
                    params["endTime"] = end_time
                r = await c.get("https://api.binance.com/api/v3/klines", params=params)
                page = r.json()
                if not page or isinstance(page, dict):
                    break
                all_data = page + all_data  # prepend older data
                end_time = page[0][0] - 1   # go further back
                remaining -= len(page)
                if len(page) < page_limit:
                    break  # no more history

    data = all_data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbv","tbqv","ignore"
    ])
    for col in ["open","high","low","close","volume","tbv","qav"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df


# ── Indicator computation (standalone, no pandas-ta import issues) ────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd_hist(s: pd.Series) -> pd.Series:
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return m - sig

def bollinger(s: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid  = s.rolling(n).mean()
    std  = s.rolling(n).std()
    return mid - 2*std, mid, mid + 2*std

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index. < 20 = ranging; > 25 = strong trend."""
    up   = df["high"].diff()
    down = -df["low"].diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_w    = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Choppiness Index. > 61.8 = consolidating; < 38.2 = trending strongly."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    range_n = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10(atr_sum / range_n.replace(0, np.nan)) / np.log10(n)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema20"]  = ema(c, 20)
    df["ema50"]  = ema(c, 50)
    df["ema100"] = ema(c, 100)
    df["ema200"] = ema(c, 200)
    df["rsi"]    = rsi(c, 14)
    df["atr"]    = atr(df, 14)
    df["atr_sma"]= df["atr"].rolling(20).mean()
    df["macd_h"] = macd_hist(c)
    bb_lo, bb_mid, bb_hi = bollinger(c, 20)
    df["bb_lo"]  = bb_lo
    df["bb_mid"] = bb_mid
    df["bb_hi"]  = bb_hi
    df["bb_pctb"]= (c - bb_lo) / (bb_hi - bb_lo + 1e-9)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"].replace(0, np.nan)
    # Taker buy proxy: tbv / volume (historical aggTrades unavailable)
    df["taker_proxy"] = df["tbv"] / df["volume"].replace(0, np.nan)
    # Regime detection
    df["adx14"]  = adx(df, 14)
    df["chop14"] = choppiness(df, 14)
    return df


# ── Signal scoring at a given bar ─────────────────────────────────────────────
def score_bar(row_1h: pd.Series, df_4h_slice: pd.DataFrame,
              regime: str) -> tuple[int, dict]:
    """
    Score 9 signals for a single 1H bar. Returns (score, details_dict).
    Signal 8 (order book) not available historically → forced 0.
    Signal 7 (taker buy) uses tbv proxy.
    """
    sigs = {}

    price  = row_1h["close"]
    ema20  = row_1h["ema20"]
    ema50  = row_1h["ema50"]
    ema100 = row_1h["ema100"]
    ema200 = row_1h["ema200"]
    rsi_v  = row_1h["rsi"]
    macd_h = row_1h["macd_h"]
    vol_r  = row_1h["vol_ratio"]
    bb_b   = row_1h["bb_pctb"]
    taker  = row_1h["taker_proxy"]

    # 4H reference
    if len(df_4h_slice) >= 2:
        r4 = df_4h_slice.iloc[-1]
        p4_ema50  = r4.get("ema50", np.nan)
        p4_ema200 = r4.get("ema200", np.nan)
        p4_close  = r4["close"]
        struct_4h = _structure(df_4h_slice)
    else:
        p4_ema50 = p4_ema200 = p4_close = np.nan
        struct_4h = "mixed"

    # 1. Trend alignment: 4H price above 4H EMA50 AND EMA200 (spec-correct timeframe)
    # Original used 1H EMA20>EMA50 which fires on every tiny bounce — too noisy.
    sigs["trend_alignment"] = int(
        not np.isnan(p4_ema50) and not np.isnan(p4_ema200) and
        p4_close > p4_ema50 and p4_close > p4_ema200
    )

    # 2. EMA stack: EMA20 > EMA50 > EMA100
    sigs["ema_stack"] = int(
        not any(np.isnan(x) for x in [ema20, ema50, ema100]) and
        ema20 > ema50 > ema100
    )

    # 3. RSI in regime-appropriate zone
    if regime == "PULLBACK":
        sigs["rsi_mode"] = int(not np.isnan(rsi_v) and 30 <= rsi_v <= 55)
    elif regime == "TRENDING":
        sigs["rsi_mode"] = int(not np.isnan(rsi_v) and 50 <= rsi_v <= 75)
    elif regime == "BREAKOUT":
        sigs["rsi_mode"] = int(not np.isnan(rsi_v) and rsi_v >= 55)
    else:
        sigs["rsi_mode"] = 0

    # 4. MACD histogram positive (bullish momentum)
    sigs["macd"] = int(not np.isnan(macd_h) and macd_h > 0)

    # 5. Structure: uptrend structure on 4H
    sigs["structure_level"] = int(struct_4h == "uptrend")

    # 6. Volume ratio >= threshold
    vol_thr = config.VOLUME_RATIO_BREAKOUT if regime == "BREAKOUT" else config.VOLUME_RATIO_THRESHOLD
    sigs["volume_ratio"] = int(not np.isnan(vol_r) and vol_r >= vol_thr)

    # 7. Taker buy proxy >= threshold (tbv/volume)
    thr = config.TAKER_BIAS_BREAKOUT_LONG if regime == "BREAKOUT" else config.TAKER_BIAS_LONG
    sigs["taker_buy_bias"] = int(not np.isnan(taker) and taker >= thr)

    # 8. Order book — NOT AVAILABLE historically → always 0
    sigs["order_book"] = 0

    # 9. Regime fit: LONG-only, require PULLBACK or TRENDING or BREAKOUT
    sigs["regime_fit"] = int(regime in ("PULLBACK", "TRENDING", "BREAKOUT"))

    score = sum(sigs.values())
    return score, sigs


def _structure(df: pd.DataFrame) -> str:
    if len(df) < 10:
        return "mixed"
    highs = df["high"].values[-10:]
    lows  = df["low"].values[-10:]
    swH, swL = [], []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]: swH.append(highs[i])
        if lows[i]  < lows[i-1]  and lows[i]  < lows[i+1]:  swL.append(lows[i])
    if len(swH) >= 2 and len(swL) >= 2:
        if swH[-1] > swH[-2] and swL[-1] > swL[-2]: return "uptrend"
        if swH[-1] < swH[-2] and swL[-1] < swL[-2]: return "downtrend"
    return "mixed"


# ── Regime detection (simplified for backtesting) ─────────────────────────────
def detect_regime_bar(row_1h: pd.Series, df_4h_slice: pd.DataFrame) -> str:
    price  = row_1h["close"]
    ema50  = row_1h["ema50"]
    ema200 = row_1h["ema200"]
    rsi_v  = row_1h["rsi"]
    bb_b   = row_1h["bb_pctb"]
    vol_r  = row_1h["vol_ratio"]
    atr_v  = row_1h["atr"]
    atr_s  = row_1h["atr_sma"]

    # LONG-only: must be above EMA200
    if not np.isnan(ema200) and price < ema200 * 0.97:
        return "RANGING"  # skip — downtrend

    # 4H context
    if len(df_4h_slice) >= 1:
        r4      = df_4h_slice.iloc[-1]
        p4_e50  = r4.get("ema50", np.nan)
        p4_e200 = r4.get("ema200", np.nan)
        p4_c    = r4["close"]
        above4  = (not np.isnan(p4_e50) and p4_c > p4_e50)
    else:
        above4 = False

    adx_v  = row_1h["adx14"]
    chop_v = row_1h["chop14"]

    # PULLBACK: macro uptrend + price near EMA50 + RSI in pullback zone +
    #           ADX < 20 (not in free-fall) + CHOP > 61.8 (consolidating)
    near_ema50 = (not np.isnan(ema50) and
                  abs(price - ema50) / ema50 < 0.025)
    macro_up_4h = (len(df_4h_slice) >= 1 and
                   not np.isnan(df_4h_slice.iloc[-1].get("ema50", np.nan)) and
                   not np.isnan(df_4h_slice.iloc[-1].get("ema200", np.nan)) and
                   df_4h_slice.iloc[-1]["ema50"] > df_4h_slice.iloc[-1]["ema200"])
    # ADX < 25 (not in a strong directional move — price slowing down, not free-falling)
    # CHOP > 55 (some consolidation; strict 61.8 blocks pullbacks in transition phases)
    adx_ranging  = not np.isnan(adx_v)  and adx_v  < 25
    chop_ranging = not np.isnan(chop_v) and chop_v > 55
    is_pullback = (above4 and near_ema50 and macro_up_4h and adx_ranging and chop_ranging and
                   not np.isnan(rsi_v) and 28 <= rsi_v <= 55 and
                   not np.isnan(bb_b) and bb_b < 0.40)

    # TRENDING: above both EMAs, RSI 50-75, ATR expanding +
    #           ADX > 25 (confirmed directional move) + CHOP < 38.2 (not choppy)
    # Re-enabled with proper regime guards (previously 14% win rate without them)
    adx_4h_v  = df_4h_slice.iloc[-1].get("adx14", np.nan)  if len(df_4h_slice) >= 1 else np.nan
    chop_4h_v = df_4h_slice.iloc[-1].get("chop14", np.nan) if len(df_4h_slice) >= 1 else np.nan
    adx_trending  = not np.isnan(adx_4h_v)  and adx_4h_v  > 25
    chop_trending = not np.isnan(chop_4h_v) and chop_4h_v < 38.2
    is_trending = (not np.isnan(ema50) and price > ema50 and
                   not np.isnan(ema200) and price > ema200 and
                   not np.isnan(rsi_v) and 50 <= rsi_v <= 75 and
                   not np.isnan(atr_v) and not np.isnan(atr_s) and atr_v >= atr_s and
                   adx_trending and chop_trending)

    if is_pullback:
        return "PULLBACK"
    if is_trending:
        return "TRENDING"
    return "RANGING"


# ── Trade simulation ───────────────────────────────────────────────────────────
def simulate_fwd(direction: str, entry: float, sl: float,
                 tp1: float, tp2: float,
                 fwd_bars: list[dict]) -> dict:
    tp1_hit    = False
    running_sl = sl
    for i, b in enumerate(fwd_bars):
        hi, lo = b["high"], b["low"]
        if direction == "LONG":
            if lo <= running_sl:
                px = running_sl
                pnl = TP1_R*0.5 + (px-entry)/abs(entry-sl)*0.5 if tp1_hit else (px-entry)/abs(entry-sl)
                return {"outcome": "TP1_SL" if tp1_hit else "SL", "px": px,  # legacy simulate_fwd
                        "r": round(pnl, 3), "bars": i+1, "tp1": tp1_hit, "tp2": False}
            if hi >= tp2:
                pnl = TP1_R*0.5 + TP2_R*0.5 if tp1_hit else TP2_R
                return {"outcome": "TP2", "px": tp2,
                        "r": round(pnl, 3), "bars": i+1, "tp1": True, "tp2": True}
            if not tp1_hit and hi >= tp1:
                tp1_hit    = True
                running_sl = entry  # breakeven
    last = fwd_bars[-1]["close"] if fwd_bars else entry
    raw  = (last - entry) / abs(entry - sl)
    pnl  = TP1_R*0.5 + raw*0.5 if tp1_hit else raw
    return {"outcome": "OPEN", "px": last, "r": round(pnl, 3),
            "bars": len(fwd_bars), "tp1": tp1_hit, "tp2": False}


# ── Position sizing ────────────────────────────────────────────────────────────
def size_spot(entry: float, sl: float, atr_val: float,
              account: float) -> dict:
    """SPOT sizing: cap notional at 30%, floor stop at 3%."""
    risk_dist = entry - sl
    min_dist  = entry * SPOT_MIN_STOP
    if risk_dist < min_dist:
        sl        = entry - min_dist
        risk_dist = min_dist
    tp1 = entry + risk_dist * TP1_R
    tp2 = entry + risk_dist * TP2_R

    risk_usdt  = account * RISK_PCT
    qty        = risk_usdt / risk_dist
    notional   = qty * entry
    cap        = account * SPOT_ALLOC_PCT
    if notional > cap:
        qty      = cap / entry
        notional = cap
    actual_risk = qty * risk_dist
    return {
        "sl": round(sl, 6), "tp1": round(tp1, 6), "tp2": round(tp2, 6),
        "qty": qty, "notional": notional,
        "risk_usdt": actual_risk, "risk_pct": actual_risk / account * 100,
        "widened": risk_dist > (entry - sl + 1e-9),
    }


# ── Main backtest ─────────────────────────────────────────────────────────────
async def run_backtest():
    print("=" * 65)
    print(f"  ATLAS LEAN — {LOOKBACK_DAYS}-DAY WALK-FORWARD BACKTEST")
    print("=" * 65)
    print(f"  Pairs:   {PAIRS}")
    print(f"  Period:  last {LOOKBACK_DAYS} days, evaluated at each 1H close")
    print(f"  Account: ${ACCOUNT_USDT:,.0f}  |  Risk: {RISK_PCT*100:.0f}%/trade  |  Max trades: {MAX_TRADES}")
    print(f"  Mode:    SPOT (30% allocation cap, 3% min stop)")
    print(f"  Note:    Order book signal disabled (historical data unavailable)")
    print(f"           Max score = 8/9; thresholds reduced by 1 for each regime")
    print()

    # Adjusted thresholds (−1 because order book always = 0)
    thresholds = {
        "PULLBACK": config.SCORE_THRESHOLD_PULLBACK - 1,   # 4
        "TRENDING": config.SCORE_THRESHOLD_TRENDING - 1,   # 5
        "BREAKOUT": config.SCORE_THRESHOLD_BREAKOUT - 1,   # 5
    }

    # Fetch historical data
    print("Fetching historical klines...")
    hist_1h, hist_4h, hist_15m, hist_1d = {}, {}, {}, {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        h1  = await fetch_hist(pair, "1h",  LOOKBACK_DAYS, extra_candles=210)
        h4  = await fetch_hist(pair, "4h",  LOOKBACK_DAYS, extra_candles=210)
        h15 = await fetch_hist(pair, "15m", LOOKBACK_DAYS, extra_candles=0)
        # Daily: extra 60 bars so EMA50 converges (needs ~50 bars of warmup)
        h1d = await fetch_hist(pair, "1d",  LOOKBACK_DAYS, extra_candles=60)
        if h1.empty or h4.empty:
            print("FAILED — skipped")
            continue
        hist_1h[pair]  = add_indicators(h1)
        hist_4h[pair]  = add_indicators(h4)
        hist_15m[pair] = h15
        hist_1d[pair]  = add_indicators(h1d)
        print(f"OK ({len(h1)} 1H bars, {len(h4)} 4H bars, {len(h15)} 15m bars, {len(h1d)} 1D bars)")

    # Walk-forward loop
    # Use the 1H index of BTC as the time spine
    btc_1h = hist_1h.get("BTCUSDT")
    if btc_1h is None:
        print("ERROR: No BTC data"); return

    # Evaluation window: skip warmup (first 210 bars for EMA200), evaluate rest
    eval_idx = btc_1h.index[210:]
    print(f"\nEvaluation window: {eval_idx[0].strftime('%Y-%m-%d %H:%M')} to {eval_idx[-1].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Total evaluation points: {len(eval_idx)} bars × {len(PAIRS)} pairs\n")

    # Track open positions {pair: {entry, sl, tp1, tp2, ...}}
    open_trades: dict[str, dict] = {}
    all_results: list[dict]      = []
    # Cooldown: track last-loss close time per pair (no re-entry for 12h)
    last_loss_time: dict[str, pd.Timestamp] = {}
    equity = ACCOUNT_USDT
    equity_curve: list[tuple]    = [(eval_idx[0], equity)]

    for ts in eval_idx:
        # ── Manage existing open positions ──────────────────────────────────
        to_close = []
        for pair, tr in open_trades.items():
            # ── 48-hour time stop ─────────────────────────────────────────────
            hours_open = (ts - tr["entry_time"]).total_seconds() / 3600
            if hours_open >= 48:
                if pair in hist_1h and ts in hist_1h[pair].index:
                    close_px = float(hist_1h[pair].loc[ts]["close"])
                else:
                    close_px = tr["entry"]
                raw_r = (close_px - tr["entry"]) / tr["risk_dist"]
                result_r = TP1_R * 0.5 + raw_r * 0.5 if tr["tp1_done"] else raw_r
                pnl_usdt = tr["risk_usdt"] * result_r
                to_close.append({
                    **tr,
                    "outcome": "TIME_STOP",
                    "close_px": close_px,
                    "close_time": ts,
                    "r": round(result_r, 3),
                    "pnl_usdt": round(pnl_usdt, 2),
                    "tp1": tr["tp1_done"], "tp2": False,
                })
                continue

            if pair not in hist_15m:
                continue
            # Get 15m bars after entry time
            df15 = hist_15m[pair]
            fwd = df15[df15.index > tr["entry_time"]]
            fwd = fwd[fwd.index <= ts]  # up to current 1H bar
            if fwd.empty:
                continue
            # Check each 15m bar
            for _, bar in fwd.iterrows():
                hi, lo = bar["high"], bar["low"]
                # TP1: close 100% (fix — previously only 50%, giving +0.75R avg)
                if not tr["tp1_done"] and hi >= tr["tp1"]:
                    result_r = TP1_R
                    pnl_usdt = tr["risk_usdt"] * result_r
                    to_close.append({
                        **tr,
                        "outcome": "TP1",
                        "close_px": tr["tp1"],
                        "close_time": bar.name,
                        "r": round(result_r, 3),
                        "pnl_usdt": round(pnl_usdt, 2),
                        "tp1": True, "tp2": False,
                    })
                    break
                if lo <= tr["running_sl"]:
                    result_r = -1.0
                    pnl_usdt = tr["risk_usdt"] * result_r
                    to_close.append({
                        **tr,
                        "outcome": "SL",
                        "close_px": tr["running_sl"],
                        "close_time": bar.name,
                        "r": round(result_r, 3),
                        "pnl_usdt": round(pnl_usdt, 2),
                        "tp1": False, "tp2": False,
                    })
                    break
                if hi >= tr["tp2"]:
                    result_r = TP2_R
                    pnl_usdt = tr["risk_usdt"] * result_r
                    to_close.append({
                        **tr,
                        "outcome": "TP2",
                        "close_px": tr["tp2"],
                        "close_time": bar.name,
                        "r": round(result_r, 3),
                        "pnl_usdt": round(pnl_usdt, 2),
                        "tp1": True, "tp2": True,
                    })
                    break

        for t in to_close:
            equity += t["pnl_usdt"]
            equity_curve.append((t["close_time"], equity))
            all_results.append(t)
            if t["pair"] in open_trades:
                del open_trades[t["pair"]]
            # Record cooldown for losses
            if t["r"] < 0:
                last_loss_time[t["pair"]] = t["close_time"]

        # ── Scan for new entries ─────────────────────────────────────────────
        if len(open_trades) >= MAX_TRADES:
            continue

        for pair in PAIRS:
            if pair in open_trades or len(open_trades) >= MAX_TRADES:
                continue
            if pair not in hist_1h or pair not in hist_4h:
                continue

            # Re-entry cooldown: skip if last loss was < 12h ago
            if pair in last_loss_time:
                hours_since_loss = (ts - last_loss_time[pair]).total_seconds() / 3600
                if hours_since_loss < 12:
                    continue

            df1 = hist_1h[pair]
            df4 = hist_4h[pair]

            # Get row at this timestamp
            if ts not in df1.index:
                continue
            row1 = df1.loc[ts]

            # Get 4H slice up to this timestamp
            df4_sl = df4[df4.index <= ts].tail(50)
            if len(df4_sl) < 5:
                continue

            # ── Daily trend filter: require 1D close > 1D EMA21 ──────────────
            # Blocks entries during macro downtrends even when 1H/4H look OK
            if pair in hist_1d:
                df1d_sl = hist_1d[pair][hist_1d[pair].index <= ts].tail(60)
                if len(df1d_sl) >= 21:
                    last_1d = df1d_sl.iloc[-1]
                    daily_ema21 = last_1d.get("ema20", np.nan)  # ema20≈ema21
                    daily_ema50 = last_1d.get("ema50", np.nan)
                    daily_close = last_1d["close"]
                    # Skip if price is below daily EMA21 OR daily EMA21 < daily EMA50
                    if (not np.isnan(daily_ema21) and daily_close < daily_ema21):
                        continue
                    if (not np.isnan(daily_ema21) and not np.isnan(daily_ema50)
                            and daily_ema21 < daily_ema50):
                        continue

            # Detect regime
            regime = detect_regime_bar(row1, df4_sl)
            if regime == "RANGING":
                continue

            # Score signals
            score, sigs = score_bar(row1, df4_sl, regime)
            threshold   = thresholds.get(regime, 999)

            if score < threshold:
                continue

            # Valid entry — size position
            entry     = row1["close"]
            atr_val   = row1["atr"]
            raw_sl    = entry - atr_val * SL_ATR_MULT
            sized     = size_spot(entry, raw_sl, atr_val, equity)

            # Entry is at OPEN of next 1H bar
            next_bars = df1[df1.index > ts].head(1)
            if next_bars.empty:
                continue
            actual_entry = float(next_bars.iloc[0]["open"])
            entry_time   = next_bars.index[0]

            # Recalculate SL/TP from actual entry
            risk_dist = actual_entry - sized["sl"] + (entry - actual_entry)
            if risk_dist <= 0:
                risk_dist = actual_entry * SPOT_MIN_STOP
            sl  = actual_entry - risk_dist
            tp1 = actual_entry + risk_dist * TP1_R
            tp2 = actual_entry + risk_dist * TP2_R

            open_trades[pair] = {
                "pair":       pair,
                "regime":     regime,
                "score":      score,
                "threshold":  threshold,
                "signal_ts":  ts,
                "entry_time": entry_time,
                "entry":      actual_entry,
                "sl":         sl,
                "tp1":        tp1,
                "tp2":        tp2,
                "risk_dist":  risk_dist,
                "risk_usdt":  sized["risk_usdt"],
                "notional":   sized["notional"],
                "qty":        sized["qty"],
                "running_sl": sl,
                "tp1_done":   False,
                "signals":    sigs,
                "atr":        atr_val,
            }

    # ── Force-close any still-open trades at last bar price ──────────────────
    last_ts = eval_idx[-1]
    for pair, tr in open_trades.items():
        if pair not in hist_1h:
            continue
        last_price = float(hist_1h[pair]["close"].iloc[-1])
        result_r   = (last_price - tr["entry"]) / tr["risk_dist"]
        if tr["tp1_done"]:
            result_r = TP1_R * 0.5 + result_r * 0.5
        pnl_usdt = tr["risk_usdt"] * result_r
        equity  += pnl_usdt
        equity_curve.append((last_ts, equity))
        all_results.append({
            **tr,
            "outcome":    "STILL_OPEN",
            "close_px":   last_price,
            "close_time": last_ts,
            "r":          round(result_r, 3),
            "pnl_usdt":   round(pnl_usdt, 2),
            "tp1":        tr["tp1_done"], "tp2": False,
        })

    # ── Statistics ────────────────────────────────────────────────────────────
    wins       = [r for r in all_results if r["r"] > 0]
    losses     = [r for r in all_results if r["r"] < 0]
    breakevens = [r for r in all_results if r["r"] == 0]
    n          = len(all_results)
    win_rate   = len(wins) / n * 100 if n else 0
    avg_w      = np.mean([r["r"] for r in wins])  if wins   else 0
    avg_l      = np.mean([r["r"] for r in losses]) if losses else 0
    expectancy = (win_rate/100 * avg_w) + ((1-win_rate/100) * avg_l)
    total_pnl  = sum(r["pnl_usdt"] for r in all_results)
    final_acc  = ACCOUNT_USDT + total_pnl

    # Max drawdown from equity curve
    eq_vals = [e for _, e in equity_curve]
    peak, max_dd = eq_vals[0], 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd

    # Regime breakdown
    by_regime = defaultdict(lambda: {"n": 0, "r": 0.0, "wins": 0})
    for r in all_results:
        rg = r["regime"]
        by_regime[rg]["n"]    += 1
        by_regime[rg]["r"]    += r["r"]
        by_regime[rg]["wins"] += int(r["r"] > 0)

    # Pair breakdown
    by_pair = defaultdict(lambda: {"n": 0, "r": 0.0, "wins": 0})
    for r in all_results:
        p = r["pair"]
        by_pair[p]["n"]    += 1
        by_pair[p]["r"]    += r["r"]
        by_pair[p]["wins"] += int(r["r"] > 0)

    # ── Console output ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  TRADE LOG")
    print("="*65)
    for i, r in enumerate(all_results, 1):
        icon = {"TP2": "🏆", "TP1": "✅", "TIME_STOP": "⏰", "SL": "❌", "STILL_OPEN": "⏳"}.get(r["outcome"], "➖")
        print(f"{icon} #{i:2d} {r['pair']:10s} {r['regime']:8s} "
              f"score={r['score']}/{r['threshold']} "
              f"entry=${r['entry']:.2f} → {r['outcome']:8s} "
              f"{r['r']:+.2f}R  ${r['pnl_usdt']:+.2f}  "
              f"[{r['signal_ts'].strftime('%m-%d %H:%M')}→{r['close_time'].strftime('%m-%d %H:%M')}]")

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Period:       {eval_idx[0].strftime('%Y-%m-%d')} → {eval_idx[-1].strftime('%Y-%m-%d')}")
    print(f"  Total trades: {n}")
    print(f"  Winners:      {len(wins)} ({win_rate:.0f}%)  |  Losers: {len(losses)}  |  B/E: {len(breakevens)}  |  Open: {len([r for r in all_results if r['outcome']=='STILL_OPEN'])}")
    print(f"  Avg winner:   {avg_w:+.2f}R  |  Avg loser: {avg_l:+.2f}R")
    print(f"  Expectancy:   {expectancy:+.3f}R per trade")
    print(f"  Max drawdown: {max_dd:.1f}%")
    print(f"  Start:  ${ACCOUNT_USDT:,.2f}  →  End: ${final_acc:,.2f}  (P&L: ${total_pnl:+.2f} = {total_pnl/ACCOUNT_USDT*100:+.1f}%)")
    print(f"\n  REGIME BREAKDOWN:")
    for rg, s in by_regime.items():
        wr = s["wins"]/s["n"]*100 if s["n"] else 0
        print(f"    {rg:10s}  {s['n']:2d} trades  win={wr:.0f}%  avg={s['r']/s['n']:+.2f}R")
    print(f"\n  PAIR BREAKDOWN:")
    for p, s in sorted(by_pair.items(), key=lambda x: -x[1]["r"]):
        wr = s["wins"]/s["n"]*100 if s["n"] else 0
        print(f"    {p:10s}  {s['n']:2d} trades  win={wr:.0f}%  total={s['r']:+.2f}R")

    # ── Build Telegram report ─────────────────────────────────────────────────
    tg_lines = [
        f"<b>ATLAS LEAN — {LOOKBACK_DAYS}-DAY BACKTEST RESULTS</b>",
        f"<i>{eval_idx[0].strftime('%Y-%m-%d')} → {eval_idx[-1].strftime('%Y-%m-%d')} UTC</i>",
        f"<i>SPOT mode | $1,000 account | 2% risk | 30% max allocation | Long-only</i>",
        "",
        f"<b>TRADE RESULTS ({n} total)</b>",
    ]
    for i, r in enumerate(all_results, 1):
        icon = {"TP2":"🏆","TP1_SL":"✅","SL":"❌","STILL_OPEN":"⏳"}.get(r["outcome"],"➖")
        sign = "+" if r["pnl_usdt"] >= 0 else ""
        tg_lines.append(
            f"{icon} #{i} <b>{r['pair']}</b> {r['regime']} {r['score']}/{r['threshold']} "
            f"→ {r['outcome']} <b>{r['r']:+.2f}R  ${sign}{r['pnl_usdt']:.2f}</b>"
        )
    tg_lines += [
        "",
        "━━━ SUMMARY ━━━",
        f"Trades: <b>{n}</b>  |  Win rate: <b>{win_rate:.0f}%</b>",
        f"Winners: <b>{len(wins)}</b>  |  Losers: <b>{len(losses)}</b>  |  Still open: <b>{len([r for r in all_results if r['outcome']=='STILL_OPEN'])}</b>",
        f"Avg winner: <b>{avg_w:+.2f}R</b>  |  Avg loser: <b>{avg_l:+.2f}R</b>",
        f"Expectancy: <b>{expectancy:+.3f}R</b> per trade",
        f"Max drawdown: <b>{max_dd:.1f}%</b>",
        "",
        f"Start: <b>${ACCOUNT_USDT:,.2f}</b>  →  End: <b>${final_acc:,.2f}</b>",
        f"Total P&amp;L: <b>${total_pnl:+.2f}  ({total_pnl/ACCOUNT_USDT*100:+.1f}%)</b>",
        "",
        "━━━ BY REGIME ━━━",
    ]
    for rg, s in by_regime.items():
        wr = s["wins"]/s["n"]*100 if s["n"] else 0
        tg_lines.append(f"  {rg}: {s['n']} trades  win={wr:.0f}%  avg={s['r']/s['n']:+.2f}R")
    tg_lines.append("")
    tg_lines.append("━━━ BY PAIR ━━━")
    for p, s in sorted(by_pair.items(), key=lambda x: -x[1]["r"]):
        wr = s["wins"]/s["n"]*100 if s["n"] else 0
        tg_lines.append(f"  {p}: {s['n']} trades  win={wr:.0f}%  total={s['r']:+.2f}R")
    tg_lines += [
        "",
        "<i>Signals: 8/9 scored (order book excluded — not available historically). "
        "Taker buy proxied via candle buy volume / total volume.</i>",
        "<i>Entry at next-bar open. SL = entry - 1.5×ATR (min 3%). TP1=1.5R full close. TP2=3R.</i>",
        "<i>Filters: daily EMA21 uptrend + 4H EMA50>EMA200 for PULLBACK + 12h loss cooldown per pair + 48h time stop.</i>",
    ]

    print("\nSending to Telegram...")
    await tg("\n".join(tg_lines))
    print("Done.")
    return all_results, equity_curve


if __name__ == "__main__":
    asyncio.run(run_backtest())
