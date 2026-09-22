"""
ATLAS Lean v2.0 — Walk-Forward Backtest

Replays every 1H candle as if it were "now" and simulates all qualifying
entries forward to SL/TP resolution.

THE ENGINE IS SHARED WITH LIVE. Indicators, regime detection, signal scoring
and entry filters are imported from indicators.py / regime_detector.py /
signals.py — the same modules main.py runs. This file contributes only the
historical data feed, the fill simulator and the reporting.

Assumptions:
- Entry at OPEN of the candle following the signal candle
- SL and TP calculated from ATR at signal time (× 1.5 stop, 1.5R/3.0R targets)
- Max 3 simultaneous trades, 1 per pair at a time
- Long-only (direction="LONG" is passed explicitly to the scorer)
- Risk: 2% of account per trade (SPOT mode: capped at 30% allocation)
- Account: $1,000 USDT

Known divergence from live (one, unavoidable):
- Binance serves no historical order-book depth, so signal 8 always scores 0
  and the max score is 8/9. Regime thresholds are reduced by 1 to compensate.
  Live, with real depth, can reach 9/9.

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
# The backtest runs the SAME code as the live scanner. Indicators, regime
# detection, signal scoring and entry filters are all imported — not
# reimplemented — so a backtest number describes the strategy that actually
# trades. (Previously this file carried its own copies with different EMA
# periods, RSI bands and structure rules.)
from indicators import compute_indicators, compute_volume_metrics
from regime_detector import detect_regime
from signals import score_pair, passes_daily_trend_filter

# ── Settings ─────────────────────────────────────────────────────────────────
ACCOUNT_USDT   = 1000.0
RISK_PCT       = 0.02          # 2% per trade
SL_ATR_MULT    = 1.5
TP1_R          = 1.5
TP2_R          = 3.0
MAX_TRADES     = 3
SPOT_ALLOC_PCT = 0.30          # max 30% of account per trade (SPOT)
SPOT_MIN_STOP  = 0.03          # min 3% stop distance (SPOT floor)
LOOKBACK_DAYS  = 7             # default; overridden by --days when run as a script
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
    # Column names match scanner.fetch_klines() exactly so the frames can be
    # handed straight to indicators.compute_indicators().
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.drop(columns=["ignore"], inplace=True)
    df.set_index("open_time", inplace=True)
    return df


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
    print(f"  Engine:  shared with live — indicators.py / regime_detector.py / signals.py")
    print(f"  Filters: daily trend={config.DAILY_TREND_FILTER}, "
          f"re-entry cooldown={config.REENTRY_COOLDOWN_HOURS}h")
    print(f"  Note:    Order book signal always 0 (depth not available historically)")
    print(f"           Max score = 8/9; thresholds reduced by 1 for each regime")
    print()

    # The ONLY deliberate divergence from live: Binance serves no historical
    # order-book depth, so signal 8 can never score. Thresholds drop by 1 to
    # keep the bar equivalent. Every other rule is the live code, unmodified.
    thresholds = {
        "PULLBACK": config.SCORE_THRESHOLD_PULLBACK - 1,
        "TRENDING": config.SCORE_THRESHOLD_TRENDING - 1,
        "BREAKOUT": config.SCORE_THRESHOLD_BREAKOUT - 1,
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
        # Indicators are computed ONCE over the full history using the live
        # engine. Every indicator here is causal (EMA / RSI / ATR / rolling
        # windows look only backwards), so computing on the full series and
        # then slicing to `ts` is identical to computing on the slice — no
        # lookahead is introduced.
        ind = compute_indicators({"1h": h1, "4h": h4, "1d": h1d})
        hist_1h[pair]  = ind.get("1h", h1)
        hist_4h[pair]  = ind.get("4h", h4)
        hist_1d[pair]  = ind.get("1d", h1d)
        hist_15m[pair] = h15   # raw — used only for intrabar SL/TP fills
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

            # Re-entry cooldown after a loss (config.REENTRY_COOLDOWN_HOURS).
            # Live reads the same window from the trades table via
            # db.hours_since_last_loss().
            if config.REENTRY_COOLDOWN_HOURS > 0 and pair in last_loss_time:
                hours_since_loss = (ts - last_loss_time[pair]).total_seconds() / 3600
                if hours_since_loss < config.REENTRY_COOLDOWN_HOURS:
                    continue

            df1 = hist_1h[pair]
            df4 = hist_4h[pair]

            # Get row at this timestamp
            if ts not in df1.index:
                continue

            # Build the same klines dict the live scanner passes around,
            # truncated to everything known at `ts` (no lookahead).
            df4_sl = df4[df4.index <= ts].tail(50)
            if len(df4_sl) < 5:
                continue
            klines = {
                "1h": df1.loc[:ts],
                "4h": df4_sl,
            }
            if pair in hist_1d:
                klines["1d"] = hist_1d[pair][hist_1d[pair].index <= ts].tail(60)

            # Volume metrics via the live helper. depth / book_ticker /
            # agg_trades / ticker24h are None because Binance serves no
            # historical snapshots — compute_volume_metrics then returns the
            # neutral defaults (ob_bid_pct = 0.5), so signal 8 scores 0.
            vol_metrics = compute_volume_metrics(
                klines["1h"], klines["4h"], None, None, None, None,
            )

            # Daily trend filter — shared implementation
            daily_ok, _ = passes_daily_trend_filter(klines)
            if not daily_ok:
                continue

            # Regime detection — live implementation
            regime = detect_regime(klines, vol_metrics).regime
            if regime == "RANGING":
                continue

            # Signal scoring — live implementation
            sig_result = score_pair(pair, klines, vol_metrics, regime, direction="LONG")
            score     = sig_result.score
            sigs      = {k: int(v) for k, v in sig_result.signals.items()}
            threshold = thresholds.get(regime, 999)

            if score < threshold:
                continue

            # Valid entry — size position
            row1      = df1.loc[ts]
            entry     = float(row1["close"])
            atr_val   = float(row1["atr14"])
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
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


def _parse_args():
    """Parse CLI args. Called only from __main__ so importing this module
    never touches sys.argv (which broke any caller with its own arguments)."""
    parser = argparse.ArgumentParser(description="ATLAS Lean Walk-Forward Backtest")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to backtest (default: 7)")
    return parser.parse_args()


if __name__ == "__main__":
    LOOKBACK_DAYS = _parse_args().days
    asyncio.run(run_backtest())
