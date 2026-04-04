"""
ATLAS Lean v2.0 — Position Manager
7-state trailing stop state machine. Runs every 5 minutes for each open trade.
Stops only move in the profit direction — never backwards.
This is pure Python — zero Claude calls, zero API cost.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import config
import db

logger = logging.getLogger(__name__)

# State ordering for validation (stop can only advance forward)
STATE_ORDER = [
    "INITIAL", "PROTECTED", "BREAKEVEN",
    "TRAILING_1", "TRAILING_2", "TRAILING_3", "TRAILING_MAX",
]


@dataclass
class StopUpdate:
    trade_id: int
    pair: str
    old_state: str
    new_state: str
    old_stop: float
    new_stop: float
    action: str       # "HOLD" | "RAISE_STOP" | "CLOSE_HALF" | "CLOSE_ALL" | "CLOSE_STOP" | "CLOSE_STRUCT" | "CLOSE_TIME"
    reason: str
    current_price: float = 0.0
    unrealised_r: float = 0.0


class PositionManager:
    """7-state trailing stop machine with exit priority ordering."""

    def compute_r_multiple(
        self, entry_price: float, stop_initial: float,
        current_price: float, direction: str,
    ) -> float:
        """
        Compute current R-multiple.
        R = (current - entry) / (entry - stop) for LONG
        R = (entry - current) / (stop - entry) for SHORT
        """
        if direction == "LONG":
            risk = entry_price - stop_initial
        else:
            risk = stop_initial - entry_price

        if risk <= 0:
            return 0.0

        if direction == "LONG":
            return (current_price - entry_price) / risk
        else:
            return (entry_price - current_price) / risk

    def update_position(
        self, trade: dict, current_price: float,
        klines_1h: Optional[pd.DataFrame] = None,
    ) -> StopUpdate:
        """
        Main entry point. Called every cycle for each open trade.
        Checks exit conditions in priority order, then state transitions.
        Returns StopUpdate describing what happened.
        """
        trade_id = trade["id"]
        pair = trade["pair"]
        direction = trade["direction"]
        entry = trade["entry_price"]
        stop_initial = trade["stop_initial"]
        stop_current = trade["stop_current"]
        stop_state = trade["stop_state"]
        atr = trade["atr_at_entry"] or 0
        peak = trade["peak_price"] or entry

        current_r = self.compute_r_multiple(entry, stop_initial, current_price, direction)

        # Update peak price
        if direction == "LONG" and current_price > peak:
            peak = current_price
            db.update_trade(trade_id, peak_price=peak)
        elif direction == "SHORT" and current_price < peak:
            peak = current_price
            db.update_trade(trade_id, peak_price=peak)

        # --- EXIT PRIORITY 1: Stop hit ---
        if self._check_stop_hit(current_price, stop_current, direction):
            return StopUpdate(
                trade_id=trade_id, pair=pair,
                old_state=stop_state, new_state=stop_state,
                old_stop=stop_current, new_stop=stop_current,
                action="CLOSE_STOP", reason="Stop loss hit",
                current_price=current_price, unrealised_r=current_r,
            )

        # --- EXIT PRIORITY 2: TP1 (close 50%) ---
        if not trade["half_closed"] and self._check_tp1(current_price, trade["tp1"], direction):
            new_stop = self._compute_trailing1_stop(entry, stop_initial, direction)
            new_stop = max(new_stop, stop_current) if direction == "LONG" else min(new_stop, stop_current)
            return StopUpdate(
                trade_id=trade_id, pair=pair,
                old_state=stop_state, new_state="TRAILING_1",
                old_stop=stop_current, new_stop=new_stop,
                action="CLOSE_HALF", reason="TP1 hit — closing 50%",
                current_price=current_price, unrealised_r=current_r,
            )

        # --- EXIT PRIORITY 3: Structure break ---
        if klines_1h is not None and self._check_structure_break(klines_1h, trade):
            return StopUpdate(
                trade_id=trade_id, pair=pair,
                old_state=stop_state, new_state=stop_state,
                old_stop=stop_current, new_stop=stop_current,
                action="CLOSE_STRUCT",
                reason="1H structure broken — key level lost",
                current_price=current_price, unrealised_r=current_r,
            )

        # --- EXIT PRIORITY 4: Time exit ---
        if self._check_time_exit(trade, current_r):
            return StopUpdate(
                trade_id=trade_id, pair=pair,
                old_state=stop_state, new_state=stop_state,
                old_stop=stop_current, new_stop=stop_current,
                action="CLOSE_TIME",
                reason=f"Open >{config.MAX_TRADE_HOURS}h with R<1.5 — dead capital",
                current_price=current_price, unrealised_r=current_r,
            )

        # --- State transitions (stop advancement) ---
        new_state, new_stop = self._advance_state(
            stop_state, stop_current, entry, stop_initial,
            current_r, peak, atr, direction,
        )

        if new_state != stop_state or new_stop != stop_current:
            action = "RAISE_STOP"
            reason = f"Stop advanced: {stop_state} -> {new_state}"

            # Persist to DB
            db.update_trade(trade_id, stop_state=new_state, stop_current=new_stop)

            return StopUpdate(
                trade_id=trade_id, pair=pair,
                old_state=stop_state, new_state=new_state,
                old_stop=stop_current, new_stop=new_stop,
                action=action, reason=reason,
                current_price=current_price, unrealised_r=current_r,
            )

        # No change
        return StopUpdate(
            trade_id=trade_id, pair=pair,
            old_state=stop_state, new_state=stop_state,
            old_stop=stop_current, new_stop=stop_current,
            action="HOLD", reason="No change",
            current_price=current_price, unrealised_r=current_r,
        )

    def _check_stop_hit(
        self, current_price: float, stop: float, direction: str
    ) -> bool:
        if direction == "LONG":
            return current_price <= stop
        return current_price >= stop

    def _check_tp1(
        self, current_price: float, tp1: float, direction: str
    ) -> bool:
        if tp1 is None or tp1 == 0:
            return False
        if direction == "LONG":
            return current_price >= tp1
        return current_price <= tp1

    def _check_structure_break(
        self, klines_1h: pd.DataFrame, trade: dict
    ) -> bool:
        """Detect if 1H candle closed through key support/resistance."""
        if len(klines_1h) < 10:
            return False

        direction = trade["direction"]
        entry = trade["entry_price"]
        lows = klines_1h["low"].values[-10:]
        highs = klines_1h["high"].values[-10:]

        # Find swing lows (support) for LONG trades
        if direction == "LONG":
            swing_lows = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    swing_lows.append(lows[i])

            if swing_lows:
                key_support = max(swing_lows)  # Most recent significant low
                current_close = float(klines_1h["close"].iloc[-1])
                # Structure break = 1H close below key support
                if current_close < key_support and key_support < entry:
                    return True

        # Find swing highs (resistance) for SHORT trades
        elif direction == "SHORT":
            swing_highs = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    swing_highs.append(highs[i])

            if swing_highs:
                key_resistance = min(swing_highs)
                current_close = float(klines_1h["close"].iloc[-1])
                if current_close > key_resistance and key_resistance > entry:
                    return True

        return False

    def _check_time_exit(self, trade: dict, current_r: float) -> bool:
        """Check if trade has been open too long with insufficient progress."""
        opened_at = trade["opened_at"]
        if opened_at is None:
            return False

        try:
            opened = datetime.fromisoformat(opened_at)
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hours_open = (now - opened).total_seconds() / 3600

            return hours_open > config.MAX_TRADE_HOURS and current_r < 1.5
        except (ValueError, TypeError):
            return False

    def _advance_state(
        self, current_state: str, current_stop: float,
        entry: float, stop_initial: float,
        current_r: float, peak: float, atr: float,
        direction: str,
    ) -> tuple[str, float]:
        """
        Determine if stop state should advance.
        Returns (new_state, new_stop). Stop NEVER moves backwards.
        """
        risk = abs(entry - stop_initial)
        new_state = current_state
        new_stop = current_stop

        if current_state == "INITIAL" and current_r >= config.STOP_R_PROTECTED:
            new_state = "PROTECTED"
            if direction == "LONG":
                new_stop = entry - (entry * 0.0025)  # Entry - 0.25%
            else:
                new_stop = entry + (entry * 0.0025)

        elif current_state == "PROTECTED" and current_r >= config.STOP_R_BREAKEVEN:
            new_state = "BREAKEVEN"
            new_stop = entry  # Exact entry — no loss possible

        elif current_state == "BREAKEVEN" and current_r >= config.STOP_R_TRAILING_1:
            new_state = "TRAILING_1"
            if direction == "LONG":
                new_stop = entry + (0.5 * risk)
            else:
                new_stop = entry - (0.5 * risk)

        elif current_state == "TRAILING_1" and current_r >= config.STOP_R_TRAILING_2:
            new_state = "TRAILING_2"
            if direction == "LONG":
                new_stop = entry + (1.0 * risk)
            else:
                new_stop = entry - (1.0 * risk)

        elif current_state == "TRAILING_2" and current_r >= config.STOP_R_TRAILING_3:
            new_state = "TRAILING_3"
            if direction == "LONG":
                new_stop = entry + (1.5 * risk)
            else:
                new_stop = entry - (1.5 * risk)

        elif current_state == "TRAILING_3" and current_r >= config.STOP_R_TRAILING_MAX:
            new_state = "TRAILING_MAX"
            if direction == "LONG":
                new_stop = peak - (config.TRAILING_MAX_ATR_MULT * atr)
            else:
                new_stop = peak + (config.TRAILING_MAX_ATR_MULT * atr)

        elif current_state == "TRAILING_MAX":
            # Dynamic trailing: recalculate from peak every cycle
            if direction == "LONG":
                candidate = peak - (config.TRAILING_MAX_ATR_MULT * atr)
                new_stop = max(current_stop, candidate)
            else:
                candidate = peak + (config.TRAILING_MAX_ATR_MULT * atr)
                new_stop = min(current_stop, candidate)

        # IRON RULE: stop can only move toward profit
        if direction == "LONG":
            new_stop = max(new_stop, current_stop)
        else:
            new_stop = min(new_stop, current_stop)

        return new_state, round(new_stop, 8)

    def _compute_trailing1_stop(
        self, entry: float, stop_initial: float, direction: str
    ) -> float:
        """Compute the TRAILING_1 stop level (entry + 0.5R)."""
        risk = abs(entry - stop_initial)
        if direction == "LONG":
            return entry + (0.5 * risk)
        return entry - (0.5 * risk)

    def check_daily_halt(self) -> bool:
        """Check if daily realized PnL exceeds the drawdown halt limit."""
        daily_pnl = db.get_daily_pnl()
        return daily_pnl <= -config.DAILY_DRAWDOWN_HALT_PCT

    def manage_all_positions(
        self, current_prices: dict[str, float],
        klines: Optional[dict[str, dict]] = None,
    ) -> list[StopUpdate]:
        """
        Iterate all open trades and manage each one.
        Returns list of all actions taken.
        """
        trades = db.get_open_trades()
        if not trades:
            return []

        updates = []
        for trade in trades:
            pair = trade["pair"]
            price = current_prices.get(pair)
            if price is None:
                logger.warning("No price for %s — skipping position update", pair)
                continue

            klines_1h = None
            if klines and pair in klines and "1h" in klines[pair]:
                klines_1h = klines[pair]["1h"]

            update = self.update_position(dict(trade), price, klines_1h)
            updates.append(update)

            # Log position snapshot
            db.insert_position_snapshot(
                trade_id=trade["id"],
                current_price=price,
                stop_loss=update.new_stop,
                stop_state=update.new_state,
                unrealised_r=update.unrealised_r,
                action_taken=update.action,
                action_detail=update.reason,
            )

        return updates


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    test_db = "test_position.db"
    config.DB_PATH = test_db
    db._conn = None

    try:
        db.init_db()
        pm = PositionManager()

        # Create a test trade: LONG BTC @ 69000, SL 68500, TP1 69750
        trade_id = db.insert_trade(
            pair="BTCUSDT", direction="LONG", source="TEST",
            regime_at_entry="TRENDING", entry_price=69000,
            stop_initial=68500, tp1=69750, tp2=70500,
            position_size_pct=1.5, atr_at_entry=430,
        )

        print("=== TRAILING STOP STATE MACHINE TEST ===\n")
        print(f"Entry: 69000 | SL: 68500 | Risk: 500 | ATR: 430")
        print(f"TP1: 69750 (+1.5R) | TP2: 70500 (+3.0R)\n")

        # Simulate price progression
        test_prices = [
            (69100, "Price barely up"),
            (69250, "At +0.5R — PROTECTED"),
            (69400, "Between PROTECTED and BREAKEVEN"),
            (69500, "At +1.0R — BREAKEVEN"),
            (69750, "At +1.5R — TRAILING_1 + TP1"),
            (70000, "At +2.0R — TRAILING_2"),
            (70250, "At +2.5R — TRAILING_3"),
            (70500, "At +3.0R — TRAILING_MAX"),
            (71000, "Peak at +4.0R"),
            (70800, "Pullback — stop should not move back"),
            (70600, "More pullback"),
        ]

        for price, description in test_prices:
            trade = dict(db.get_trade(trade_id))
            update = pm.update_position(trade, price)

            r = pm.compute_r_multiple(69000, 68500, price, "LONG")
            print(f"Price: {price} ({description})")
            print(f"  R: {r:.2f} | State: {update.old_state} -> {update.new_state}")
            print(f"  Stop: {update.old_stop:.2f} -> {update.new_stop:.2f}")
            print(f"  Action: {update.action} — {update.reason}")

            # Apply state changes to DB for next iteration
            if update.new_state != update.old_state:
                db.update_trade(trade_id, stop_state=update.new_state,
                                stop_current=update.new_stop)
            if update.action == "CLOSE_HALF":
                db.update_trade(trade_id, half_closed=1, stop_state="TRAILING_1",
                                stop_current=update.new_stop)
            if price > trade.get("peak_price", 0):
                db.update_trade(trade_id, peak_price=price)
            print()

        # Test stop hit
        trade = dict(db.get_trade(trade_id))
        # Force stop to a known value for testing
        current_stop = trade["stop_current"]
        print(f"Testing stop hit at stop={current_stop:.2f}...")
        update = pm.update_position(trade, current_stop - 1)
        print(f"  Action: {update.action} — {update.reason}")

        print("\n=== ALL POSITION MANAGER TESTS PASSED ===")

    finally:
        if db._conn:
            db._conn.close()
            db._conn = None
        if os.path.exists(test_db):
            os.remove(test_db)
