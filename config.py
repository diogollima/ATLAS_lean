"""
ATLAS Lean v2.0 — Configuration
All constants, thresholds, and risk parameters.
Loads secrets from .env via python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys (from .env)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: int = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
PAIRS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# ---------------------------------------------------------------------------
# Timeframes & Kline Limits
# ---------------------------------------------------------------------------
TIMEFRAMES: list[str] = ["15m", "1h", "4h", "1d", "1w"]
KLINE_LIMITS: dict[str, int] = {
    "15m": 100,
    "1h": 100,
    "4h": 100,
    "1d": 100,
    "1w": 50,
}

# ---------------------------------------------------------------------------
# Binance REST Endpoints
# ---------------------------------------------------------------------------
BINANCE_BASE: str = "https://api.binance.com"
EP_KLINES: str = "/api/v3/klines"
EP_TICKER24H: str = "/api/v3/ticker/24hr"
EP_DEPTH: str = "/api/v3/depth"
EP_BOOK_TICKER: str = "/api/v3/ticker/bookTicker"
EP_AGG_TRADES: str = "/api/v3/aggTrades"

# ---------------------------------------------------------------------------
# Risk Controls — HARDCODED, NON-OVERRIDABLE
# These are Python code, not strategy file text. Claude cannot override them.
# Telegram commands cannot override them. Weekly review cannot modify them.
# ---------------------------------------------------------------------------
MAX_RISK_PCT: float = 2.0                # Max risk per trade (% of account)
MAX_SIMULTANEOUS_TRADES: int = 3          # Max open trades at any time
DAILY_DRAWDOWN_HALT_PCT: float = 4.0      # Close all + halt if daily P&L < -4%
MIN_RR_RATIO: float = 1.5                # Minimum risk:reward ratio

# ---------------------------------------------------------------------------
# Signal Scoring Thresholds
# ---------------------------------------------------------------------------
SCORE_THRESHOLD_TRENDING: int = 6         # Min score to trigger Claude in TRENDING
SCORE_THRESHOLD_PULLBACK: int = 5         # Min score to trigger Claude in PULLBACK
SCORE_THRESHOLD_BREAKOUT: int = 6         # Min score to trigger Claude in BREAKOUT
MIN_SCORE_LOG: int = 3                    # Min score to log signal to DB

VOLUME_RATIO_THRESHOLD: float = 1.3       # 130% of 20-period avg for standard regimes
VOLUME_RATIO_BREAKOUT: float = 1.5        # 150% for breakout regime

TAKER_BIAS_LONG: float = 0.55            # Taker buy ratio threshold for long
TAKER_BIAS_SHORT: float = 0.45           # Taker buy ratio threshold for short
TAKER_BIAS_BREAKOUT_LONG: float = 0.60   # Breakout taker threshold for long
TAKER_BIAS_BREAKOUT_SHORT: float = 0.40  # Breakout taker threshold for short

OB_IMBALANCE_THRESHOLD: float = 0.58     # Order book imbalance (58/42)
OB_IMBALANCE_BREAKOUT: float = 0.60      # Order book imbalance for breakout (60/40)

# ---------------------------------------------------------------------------
# Trailing Stop R-Levels
# ---------------------------------------------------------------------------
STOP_R_PROTECTED: float = 0.5            # Move to protected at +0.5R
STOP_R_BREAKEVEN: float = 1.0            # Move to breakeven at +1.0R
STOP_R_TRAILING_1: float = 1.5           # Close 50%, trail at +1.5R
STOP_R_TRAILING_2: float = 2.0           # Trail at +2.0R
STOP_R_TRAILING_3: float = 2.5           # Trail at +2.5R
STOP_R_TRAILING_MAX: float = 3.0         # Dynamic trail at +3.0R

TP1_CLOSE_PCT: float = 0.50              # Close 50% at TP1 (TRAILING_1)
TRAILING_MAX_ATR_MULT: float = 0.5       # Stop = peak - 0.5 * ATR in TRAILING_MAX

# ---------------------------------------------------------------------------
# Time Limits
# ---------------------------------------------------------------------------
MAX_TRADE_HOURS: int = 48                # Close dead trades after 48h
SCAN_INTERVAL_SECONDS: int = 300         # 5-minute scan cycle

# ---------------------------------------------------------------------------
# Claude Models
# ---------------------------------------------------------------------------
HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
SONNET_MODEL: str = "claude-sonnet-4-6-20250514"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas.db")

# ---------------------------------------------------------------------------
# Trading Mode — switch between SPOT and FUTURES at any time
# Set in .env as TRADING_MODE=SPOT or TRADING_MODE=FUTURES
# ---------------------------------------------------------------------------
TRADING_MODE: str = os.getenv("TRADING_MODE", "SPOT").upper()   # "SPOT" | "FUTURES"

# ---------------------------------------------------------------------------
# SPOT Mode Parameters
# ---------------------------------------------------------------------------
# On spot you can only spend capital you own.  No leverage means position
# sizing is constrained by available balance, not just risk %.
#
# Fee reserve  — kept in cash at all times for trading fees & slippage.
SPOT_FEE_RESERVE_PCT: float = 10.0     # Reserve 10 % of account; never trade it.
#
# Per-trade allocation ceiling  — max % of account in one position.
# With MAX_SIMULTANEOUS_TRADES=3 and 10 % reserve:
#   (100% - 10%) / 3 = 30% per trade  → 90% fully deployed across 3 trades.
SPOT_MAX_ALLOC_PCT: float = 30.0       # Hard ceiling: never put > 30% on one trade.
#
# Minimum stop distance  — prevents the ATR formula from producing a stop so
# tight that the required notional exceeds the per-trade allocation.
# 3% of entry price → for BTC @ $69k: SL at $66,930, risk dist $2,070.
# Risk-based qty = (1% of $1,000 USDT) / $2,070 = 0.0048 BTC = $333 notional ✓
SPOT_MIN_STOP_PCT: float = 3.0         # Floor on stop distance = 3% of entry.

# ---------------------------------------------------------------------------
# FUTURES Mode Parameters
# ---------------------------------------------------------------------------
# On futures the 2 % risk rule is enforced exactly — ATR-based stops work as
# designed.  A small amount of isolated margin covers the position.
FUTURES_LEVERAGE: int = int(os.getenv("FUTURES_LEVERAGE", "3"))  # Default 3x

# ---------------------------------------------------------------------------
# Paper Trading
# ---------------------------------------------------------------------------
PAPER_BALANCE: float = 1000.0           # Starting paper balance in USDT (≈ EUR 1,000)

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
STRATEGY_DIR: str = os.path.dirname(os.path.abspath(__file__))
ACTIVE_STRATEGY_FILE: str = "strategy_v1.md"


def get_score_threshold(regime: str) -> int:
    """Return the minimum signal score to trigger Claude for a given regime."""
    thresholds = {
        "TRENDING": SCORE_THRESHOLD_TRENDING,
        "PULLBACK": SCORE_THRESHOLD_PULLBACK,
        "BREAKOUT": SCORE_THRESHOLD_BREAKOUT,
        "RANGING": 999,  # Never trigger Claude for RANGING
    }
    return thresholds.get(regime, 999)
