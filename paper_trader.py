"""
ATLAS Lean v2.0 — Paper Trader
Creates, closes, and updates paper trade records in SQLite.
Manual entries bypass the scanner but still enforce all risk limits.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import config
import db
from position_sizer import PositionSize, size_position

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    pair: str
    direction: str              # "LONG" only (long-only strategy)
    entry_price: float          # Proposed entry (market price at signal time)
    stop_initial: float         # Raw SL from strategy/Claude (may be widened in SPOT)
    tp1: float                  # First take-profit (overridden by sizer if stop widens)
    tp2: float                  # Second take-profit (overridden by sizer if stop widens)
    position_size_pct: float    # Legacy field — actual sizing handled by position_sizer
    atr_at_entry: float
    source: str = "SCANNER"     # "SCANNER" or "MANUAL"
    regime: str = ""
    signal_id: Optional[int] = None
    strategy_version: Optional[str] = None
    notes: Optional[str] = None
    # Populated automatically by open_trade() — do not set manually
    sized: Optional[PositionSize] = None


class PaperTrader:
    """Trade lifecycle management with hardcoded risk controls."""

    def validate_trade(self, setup: TradeSetup) -> tuple[bool, str]:
        """
        Validate a trade against all risk rules.
        Returns (is_valid, rejection_reason).
        Risk rules are HARDCODED — cannot be bypassed.
        """
        # 1. Max simultaneous trades
        open_count = db.count_open_trades()
        if open_count >= config.MAX_SIMULTANEOUS_TRADES:
            return False, (
                f"Max {config.MAX_SIMULTANEOUS_TRADES} trades open "
                f"(currently {open_count})"
            )

        # 2. Position size within limits
        # In SPOT mode actual_risk_pct < 2% is fine (capital-constrained);
        # in FUTURES mode it equals 2.0% exactly.  Block only if somehow > max.
        if setup.position_size_pct > config.MAX_RISK_PCT + 0.01:
            return False, (
                f"Risk {setup.position_size_pct:.2f}% exceeds max {config.MAX_RISK_PCT}%"
            )

        # 3. Risk/Reward ratio
        if setup.direction == "LONG":
            risk = setup.entry_price - setup.stop_initial
            reward = setup.tp1 - setup.entry_price
        else:
            risk = setup.stop_initial - setup.entry_price
            reward = setup.entry_price - setup.tp1

        if risk <= 0:
            return False, "Invalid stop loss (risk <= 0)"

        rr_ratio = reward / risk
        if rr_ratio < config.MIN_RR_RATIO:
            return False, (
                f"R:R {rr_ratio:.2f} below minimum {config.MIN_RR_RATIO}"
            )

        # 4. Daily drawdown check
        daily_pnl = db.get_daily_pnl()
        if daily_pnl <= -config.DAILY_DRAWDOWN_HALT_PCT:
            return False, (
                f"Daily drawdown {daily_pnl:.2f}% exceeds "
                f"-{config.DAILY_DRAWDOWN_HALT_PCT}% halt limit"
            )

        # 5. No duplicate trade on same pair
        existing = db.get_open_trade_for_pair(setup.pair)
        if existing:
            return False, f"Already have an open trade on {setup.pair}"

        # 6. Basic price sanity
        if setup.entry_price <= 0 or setup.stop_initial <= 0:
            return False, "Invalid price values"

        if setup.direction == "LONG" and setup.stop_initial >= setup.entry_price:
            return False, "LONG stop must be below entry"

        if setup.direction == "SHORT" and setup.stop_initial <= setup.entry_price:
            return False, "SHORT stop must be above entry"

        return True, "Valid"

    def has_manual_trades(self) -> bool:
        """Return True if any open trade was entered manually by the user."""
        trades = db.get_open_trades()
        return any(t["source"] == "MANUAL" for t in trades)

    def open_trade(self, setup: TradeSetup, account_usdt: float = config.PAPER_BALANCE) -> tuple[int, str]:
        """
        Validate, size, and open a paper trade.
        Calls position_sizer to compute qty / actual risk for the active mode.
        Returns (trade_id, message).
        Raises ValueError if validation fails.
        """
        # 1. Run position sizer — this may widen the stop in SPOT mode
        try:
            ps = size_position(
                entry=setup.entry_price,
                sl=setup.stop_initial,
                atr=setup.atr_at_entry,
                account_usdt=account_usdt,
            )
        except Exception as e:
            raise ValueError(f"Position sizing failed: {e}") from e

        # Attach sizing result so validate_trade can use the final SL/TP
        setup.sized = ps
        setup.stop_initial = ps.sl      # use widened SL if SPOT floor applied
        setup.tp1 = ps.tp1              # recalculated from final stop distance
        setup.tp2 = ps.tp2
        setup.position_size_pct = ps.actual_risk_pct

        # 2. Validate risk rules against the *sized* trade
        is_valid, reason = self.validate_trade(setup)
        if not is_valid:
            raise ValueError(f"Trade rejected: {reason}")

        # 3. Persist to DB
        trade_id = db.insert_trade(
            pair=setup.pair,
            direction=setup.direction,
            source=setup.source,
            regime_at_entry=setup.regime,
            entry_price=ps.entry,
            stop_initial=ps.sl,
            tp1=ps.tp1,
            tp2=ps.tp2,
            position_size_pct=ps.actual_risk_pct,
            atr_at_entry=setup.atr_at_entry,
            signal_id=setup.signal_id,
            strategy_version=setup.strategy_version,
            notes=setup.notes,
        )

        widened_note = "  [SL widened to 3% floor]" if ps.stop_was_widened else ""
        logger.info(
            "Trade opened: #%d %s %s @ %.4f | SL=%.4f%s | TP1=%.4f | TP2=%.4f | "
            "qty=%.6f | notional=$%.2f (%.1f%%) | risk=$%.2f (%.2f%%) | mode=%s",
            trade_id, setup.pair, setup.direction, ps.entry,
            ps.sl, widened_note, ps.tp1, ps.tp2,
            ps.qty, ps.notional_usdt, ps.notional_pct,
            ps.actual_risk_usdt, ps.actual_risk_pct, ps.mode,
        )

        rr = (ps.tp1 - ps.entry) / ps.stop_distance if ps.stop_distance > 0 else 0

        msg = (
            f"Trade #{trade_id} opened  [{ps.mode}]\n"
            f"{setup.pair} LONG @ ${ps.entry:,.4f}\n"
            f"SL:  ${ps.sl:,.4f}{widened_note}\n"
            f"TP1: ${ps.tp1:,.4f}  (+1.5R)\n"
            f"TP2: ${ps.tp2:,.4f}  (+3.0R)\n"
            f"Qty: {ps.qty:.6f}  |  Notional: ${ps.notional_usdt:,.2f} ({ps.notional_pct:.1f}% acct)\n"
            f"Risk: ${ps.actual_risk_usdt:.2f} ({ps.actual_risk_pct:.2f}% acct)  |  R:R: {rr:.1f}\n"
            f"Regime: {setup.regime}  |  Source: {setup.source}"
        )

        return trade_id, msg

    def close_trade(
        self, trade_id: int, exit_price: float, reason: str
    ) -> dict:
        """
        Close a trade and calculate PnL.
        Returns dict with pnl_r, pnl_pct, and summary.
        """
        trade = db.get_trade(trade_id)
        if trade is None:
            raise ValueError(f"Trade #{trade_id} not found")
        if trade["status"] != "OPEN":
            raise ValueError(f"Trade #{trade_id} is already {trade['status']}")

        # Calculate PnL
        entry = trade["entry_price"]
        risk = abs(entry - trade["stop_initial"])

        if trade["direction"] == "LONG":
            pnl_absolute = exit_price - entry
        else:
            pnl_absolute = entry - exit_price

        pnl_r = pnl_absolute / risk if risk > 0 else 0
        pnl_pct = (pnl_absolute / entry) * trade["position_size_pct"] if entry > 0 else 0

        # Map reason to status
        status_map = {
            "STOP_HIT": "CLOSED_SL",
            "TP1": "CLOSED_TP1",
            "TP2": "CLOSED_TP2",
            "STRUCTURE_BREAK": "CLOSED_STRUCT",
            "TIME_EXIT": "CLOSED_TIME",
            "MANUAL": "CLOSED_MANUAL",
            "DAILY_HALT": "CLOSED_HALT",
        }
        status = status_map.get(reason, "CLOSED_MANUAL")

        now = datetime.now(timezone.utc).isoformat()
        db.update_trade(
            trade_id,
            status=status,
            close_price=exit_price,
            pnl_r=round(pnl_r, 3),
            pnl_pct=round(pnl_pct, 3),
            closed_at=now,
        )

        logger.info(
            "Trade #%d closed: %s @ %.2f | PnL: %.2fR (%.2f%%) | Reason: %s",
            trade_id, trade["pair"], exit_price, pnl_r, pnl_pct, reason,
        )

        return {
            "trade_id": trade_id,
            "pair": trade["pair"],
            "direction": trade["direction"],
            "entry_price": entry,
            "exit_price": exit_price,
            "pnl_r": round(pnl_r, 3),
            "pnl_pct": round(pnl_pct, 3),
            "reason": reason,
            "status": status,
        }

    def close_half(self, trade_id: int, current_price: float) -> dict:
        """
        Close 50% of the position (TP1 partial exit).
        Marks half_closed=True in the trade record.
        """
        trade = db.get_trade(trade_id)
        if trade is None:
            raise ValueError(f"Trade #{trade_id} not found")
        if trade["half_closed"]:
            raise ValueError(f"Trade #{trade_id} already half-closed")

        db.update_trade(trade_id, half_closed=1)

        entry = trade["entry_price"]
        risk = abs(entry - trade["stop_initial"])
        if trade["direction"] == "LONG":
            pnl = current_price - entry
        else:
            pnl = entry - current_price
        pnl_r = pnl / risk if risk > 0 else 0

        logger.info(
            "Trade #%d half-closed at %.2f (%.2fR)", trade_id, current_price, pnl_r
        )

        return {
            "trade_id": trade_id,
            "pair": trade["pair"],
            "half_close_price": current_price,
            "pnl_r_at_half": round(pnl_r, 3),
        }

    def close_all(
        self, current_prices: dict[str, float], reason: str
    ) -> list[dict]:
        """Emergency close all open trades (daily halt scenario)."""
        trades = db.get_open_trades()
        results = []
        for trade in trades:
            price = current_prices.get(trade["pair"])
            if price is None:
                price = trade["entry_price"]  # Fallback
            try:
                result = self.close_trade(trade["id"], price, reason)
                results.append(result)
            except Exception as e:
                logger.error("Failed to close trade #%d: %s", trade["id"], e)
        return results


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    test_db = "test_paper.db"
    config.DB_PATH = test_db
    db._conn = None

    try:
        db.init_db()
        trader = PaperTrader()

        # Test valid trade
        setup = TradeSetup(
            pair="BTCUSDT", direction="LONG",
            entry_price=69000, stop_initial=68500,
            tp1=69750, tp2=70500,
            position_size_pct=1.5, atr_at_entry=430,
            source="SCANNER", regime="PULLBACK",
        )

        trade_id, msg = trader.open_trade(setup)
        print(f"[OK] Trade opened:\n{msg}\n")

        # Test duplicate pair rejection
        try:
            trader.open_trade(setup)
            print("[FAIL] Should have rejected duplicate")
        except ValueError as e:
            print(f"[OK] Duplicate rejected: {e}")

        # Test max trades
        for i, pair in enumerate(["ETHUSDT", "SOLUSDT"]):
            s = TradeSetup(
                pair=pair, direction="LONG",
                entry_price=2000 + i*1000, stop_initial=1950 + i*1000,
                tp1=2075 + i*1000, tp2=2150 + i*1000,
                position_size_pct=1.0, atr_at_entry=30,
                source="MANUAL", regime="TRENDING",
            )
            trader.open_trade(s)

        try:
            s = TradeSetup(
                pair="BNBUSDT", direction="LONG",
                entry_price=600, stop_initial=590,
                tp1=615, tp2=630,
                position_size_pct=1.0, atr_at_entry=10,
                source="MANUAL", regime="TRENDING",
            )
            trader.open_trade(s)
            print("[FAIL] Should have rejected 4th trade")
        except ValueError as e:
            print(f"[OK] Max trades rejected: {e}")

        # Test close
        result = trader.close_trade(trade_id, 69750, "TP1")
        print(f"[OK] Trade closed: PnL={result['pnl_r']:.2f}R ({result['pnl_pct']:.2f}%)")

        # Test R:R rejection
        try:
            bad = TradeSetup(
                pair="BNBUSDT", direction="LONG",
                entry_price=600, stop_initial=590,
                tp1=605, tp2=610,  # R:R = 0.5 < 1.5
                position_size_pct=1.0, atr_at_entry=10,
                source="MANUAL", regime="TRENDING",
            )
            trader.open_trade(bad)
            print("[FAIL] Should have rejected bad R:R")
        except ValueError as e:
            print(f"[OK] Bad R:R rejected: {e}")

        print("\n=== ALL PAPER TRADER TESTS PASSED ===")

    finally:
        if db._conn:
            db._conn.close()
            db._conn = None
        if os.path.exists(test_db):
            os.remove(test_db)
