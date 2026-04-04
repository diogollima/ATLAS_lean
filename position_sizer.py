"""
ATLAS Lean v2.0 — Position Sizer
Calculates position size, actual risk, and notional exposure for a given
trade setup.  Supports two modes, switchable via config.TRADING_MODE:

  SPOT    — capital-constrained; allocation ceiling enforced; stop distance
             has a minimum floor so notional never exceeds per-trade budget.

  FUTURES — risk-first; exact 2% loss on every SL; ATR stops intact;
             notional can exceed account (leverage handles the difference).

Usage:
    from position_sizer import size_position, PositionSize
    ps = size_position(entry=68794, sl=68037, atr=505, account_usdt=1152)
    print(ps)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """All sizing outputs for one trade."""
    mode: str               # "SPOT" | "FUTURES"

    # Prices (SL may be widened in SPOT mode to meet the 3% floor)
    entry: float
    sl: float               # Final stop-loss price (may differ from raw input)
    tp1: float
    tp2: float
    stop_distance: float    # entry - sl  (always positive for LONG)

    # Quantity & exposure
    qty: float              # Units of the base asset to buy
    notional_usdt: float    # qty × entry
    notional_pct: float     # notional as % of account

    # Risk
    target_risk_pct: float  # What we aimed for (config.MAX_RISK_PCT)
    actual_risk_usdt: float # qty × stop_distance  (real $ at risk)
    actual_risk_pct: float  # actual_risk_usdt / account × 100

    # Only meaningful for SPOT
    allocation_ceiling_usdt: float  # hard per-trade cap
    stop_was_widened: bool          # True if raw SL was tighter than 3% floor

    def summary_lines(self) -> list[str]:
        """Human-readable lines for Telegram / logs."""
        widened = "  [stop widened to 3% floor]" if self.stop_was_widened else ""
        return [
            f"Mode:           {self.mode}",
            f"Entry:          ${self.entry:,.4f}",
            f"Stop Loss:      ${self.sl:,.4f}  ({self.stop_distance/self.entry*100:.2f}% from entry){widened}",
            f"TP1 (+1.5R):    ${self.tp1:,.4f}",
            f"TP2 (+3.0R):    ${self.tp2:,.4f}",
            f"Qty:            {self.qty:.6f}",
            f"Notional:       ${self.notional_usdt:,.2f}  ({self.notional_pct:.1f}% of account)",
            f"Risk targeted:  {self.target_risk_pct:.1f}%",
            f"Risk actual:    ${self.actual_risk_usdt:.2f}  ({self.actual_risk_pct:.2f}% of account)",
        ]

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


def size_position(
    entry: float,
    sl: float,
    atr: float,
    account_usdt: float,
    mode: str | None = None,
) -> PositionSize:
    """
    Main entry point.  Returns a PositionSize for a LONG trade.

    Parameters
    ----------
    entry         : float  — proposed entry price
    sl            : float  — raw stop-loss price from strategy/Claude
    atr           : float  — ATR(1H) at signal time
    account_usdt  : float  — current paper (or live) balance in USDT
    mode          : str    — override config.TRADING_MODE if supplied
    """
    if entry <= 0 or account_usdt <= 0:
        raise ValueError("entry and account_usdt must be positive")

    resolved_mode = (mode or config.TRADING_MODE).upper()

    if resolved_mode == "SPOT":
        return _size_spot(entry, sl, atr, account_usdt)
    elif resolved_mode == "FUTURES":
        return _size_futures(entry, sl, atr, account_usdt)
    else:
        raise ValueError(f"Unknown TRADING_MODE: {resolved_mode}")


# ---------------------------------------------------------------------------
# SPOT
# ---------------------------------------------------------------------------

def _size_spot(
    entry: float, raw_sl: float, atr: float, account_usdt: float
) -> PositionSize:
    """
    SPOT position sizing.

    Rules:
      1. Stop distance floor  = max(entry - raw_sl, entry × SPOT_MIN_STOP_PCT%)
         If the ATR-based stop is tighter than 3%, we widen it to 3% so the
         risk-based quantity doesn't overshoot the allocation ceiling.

      2. Allocation ceiling   = account × (1 − fee_reserve%) / MAX_TRADES
         E.g. $1,152 × 0.9 / 3 = $345.69 max notional per trade.

      3. Risk-based quantity  = (account × TARGET_RISK%) / stop_distance
         This is what we'd buy if there were no capital constraint.

      4. Final qty            = min(risk_based_qty, ceiling / entry)
         On spot we can't spend more than the ceiling, so we cap it.

      5. Actual risk          = final_qty × stop_distance
         Will equal TARGET_RISK when the ceiling isn't binding (small coins),
         or be lower than TARGET_RISK when the ceiling binds (BTC / BNB).
    """
    target_risk_pct  = config.MAX_RISK_PCT               # e.g. 2.0
    min_stop_pct     = config.SPOT_MIN_STOP_PCT           # e.g. 3.0
    max_alloc_pct    = config.SPOT_MAX_ALLOC_PCT          # e.g. 30.0
    fee_reserve_pct  = config.SPOT_FEE_RESERVE_PCT        # e.g. 10.0
    max_trades       = config.MAX_SIMULTANEOUS_TRADES     # e.g. 3

    # 1. Enforce minimum stop distance
    raw_stop_dist    = entry - raw_sl
    min_stop_dist    = entry * (min_stop_pct / 100)
    stop_dist        = max(raw_stop_dist, min_stop_dist)
    widened          = stop_dist > raw_stop_dist + 1e-8
    sl               = entry - stop_dist

    # 2. Allocation ceiling (per trade)
    deployable       = account_usdt * (1 - fee_reserve_pct / 100)
    ceiling_usdt     = deployable / max_trades
    # Also enforce the config hard ceiling
    ceiling_usdt     = min(ceiling_usdt, account_usdt * (max_alloc_pct / 100))

    # 3. Risk-based quantity (uncapped)
    risk_usdt        = account_usdt * (target_risk_pct / 100)
    qty_risk         = risk_usdt / stop_dist

    # 4. Cap to allocation ceiling
    qty_max          = ceiling_usdt / entry
    qty              = min(qty_risk, qty_max)

    # 5. Derived values
    notional         = qty * entry
    actual_risk_usdt = qty * stop_dist
    actual_risk_pct  = (actual_risk_usdt / account_usdt) * 100
    notional_pct     = (notional / account_usdt) * 100

    tp1 = entry + stop_dist * 1.5
    tp2 = entry + stop_dist * 3.0

    if widened:
        logger.debug(
            "SPOT: stop widened %.4f → %.4f (%.2f%% floor applied)",
            raw_sl, sl, min_stop_pct,
        )

    return PositionSize(
        mode="SPOT",
        entry=entry, sl=sl, tp1=tp1, tp2=tp2,
        stop_distance=stop_dist,
        qty=qty, notional_usdt=notional, notional_pct=notional_pct,
        target_risk_pct=target_risk_pct,
        actual_risk_usdt=actual_risk_usdt, actual_risk_pct=actual_risk_pct,
        allocation_ceiling_usdt=ceiling_usdt,
        stop_was_widened=widened,
    )


# ---------------------------------------------------------------------------
# FUTURES
# ---------------------------------------------------------------------------

def _size_futures(
    entry: float, sl: float, atr: float, account_usdt: float
) -> PositionSize:
    """
    FUTURES position sizing.

    Exact 2% risk per trade.  No capital ceiling — leverage covers the gap.
    ATR-based stop unchanged.  No minimum stop floor needed.
    """
    target_risk_pct  = config.MAX_RISK_PCT
    stop_dist        = entry - sl
    if stop_dist <= 0:
        raise ValueError(f"FUTURES: SL ({sl}) must be below entry ({entry}) for LONG")

    risk_usdt        = account_usdt * (target_risk_pct / 100)
    qty              = risk_usdt / stop_dist
    notional         = qty * entry
    notional_pct     = (notional / account_usdt) * 100
    actual_risk_usdt = qty * stop_dist   # == risk_usdt exactly
    actual_risk_pct  = target_risk_pct

    tp1 = entry + stop_dist * 1.5
    tp2 = entry + stop_dist * 3.0

    return PositionSize(
        mode="FUTURES",
        entry=entry, sl=sl, tp1=tp1, tp2=tp2,
        stop_distance=stop_dist,
        qty=qty, notional_usdt=notional, notional_pct=notional_pct,
        target_risk_pct=target_risk_pct,
        actual_risk_usdt=actual_risk_usdt, actual_risk_pct=actual_risk_pct,
        allocation_ceiling_usdt=0.0,   # Not applicable for futures
        stop_was_widened=False,
    )


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    EUR_USDT   = 1.1523
    ACCOUNT    = 1000 * EUR_USDT   # EUR 1,000 → USDT

    retrotest_trades = [
        dict(pair="BTCUSDT", entry=68794.24, raw_sl=68036.62, atr=505.08),
        dict(pair="BNBUSDT", entry=629.18,   raw_sl=623.96,   atr=3.48),
        dict(pair="SOLUSDT", entry=86.76,    raw_sl=85.485,   atr=0.85),
    ]

    for mode in ("SPOT", "FUTURES"):
        print(f"\n{'='*65}")
        print(f"  MODE: {mode}   |   Account: ${ACCOUNT:,.2f} (EUR 1,000 @ {EUR_USDT})")
        print(f"{'='*65}")

        total_notional = 0.0
        total_max_loss = 0.0
        total_best     = 0.0

        for t in retrotest_trades:
            ps = size_position(t["entry"], t["raw_sl"], t["atr"], ACCOUNT, mode=mode)
            profit_full = (ps.qty * 0.5 * (ps.tp1 - ps.entry)
                         + ps.qty * 0.5 * (ps.tp2 - ps.entry))

            print(f"\n  {t['pair']}")
            print(f"  {'-'*60}")
            for line in ps.summary_lines():
                print(f"    {line}")
            print(f"    Profit at TP2:  ${profit_full:.2f}  (+{profit_full/ACCOUNT*100:.2f}% acct)")

            total_notional += ps.notional_usdt
            total_max_loss += ps.actual_risk_usdt
            total_best     += profit_full

        print(f"\n  {'─'*60}")
        print(f"  ALL 3 COMBINED:")
        print(f"    Total notional  : ${total_notional:,.2f}  ({total_notional/ACCOUNT*100:.1f}% of account)")
        print(f"    Max loss (all SL): -${total_max_loss:.2f}  (-{total_max_loss/ACCOUNT*100:.2f}% account)")
        print(f"    Best case (all TP2): +${total_best:.2f}  (+{total_best/ACCOUNT*100:.2f}% account)")

    print(f"\n{'='*65}")
    print("  KEY DIFFERENCES")
    print(f"{'='*65}")
    print("  SPOT   — stops widened to 3% floor on BTC/BNB (were <1.5%)")
    print("           notional capped at 30% of account per trade")
    print("           actual risk < 2% when ceiling binds (BTC/BNB: ~0.6%)")
    print("           actual risk = 2% when ceiling does NOT bind (SOL)")
    print()
    print("  FUTURES— ATR stops unchanged, 2% risk enforced exactly")
    print("           notional exceeds account (leverage covers the gap)")
    print("           identical R:R profile regardless of price level")
    print("           funding fees: ~0.01%/8h on open futures longs")
