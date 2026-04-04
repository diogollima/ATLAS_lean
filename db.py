"""
ATLAS Lean v2.0 — Database Layer
All SQLite read/write operations. Single module for all DB access.
5 tables: signals, trades, position_history, strategy_versions, weekly_reviews.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)

_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    """Return singleton SQLite connection with WAL mode and Row factory."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_PATH)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            pair            TEXT NOT NULL,
            timeframe       TEXT DEFAULT '5m',
            regime          TEXT NOT NULL,
            score           INTEGER NOT NULL,
            signals_json    TEXT,
            indicators_json TEXT,
            claude_called   INTEGER DEFAULT 0,
            claude_decision TEXT,
            claude_reasoning TEXT,
            claude_confidence TEXT,
            strategy_version TEXT,
            source          TEXT DEFAULT 'SCANNER',
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trades (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id         INTEGER REFERENCES signals(id),
            pair              TEXT NOT NULL,
            direction         TEXT NOT NULL DEFAULT 'LONG',
            source            TEXT NOT NULL DEFAULT 'SCANNER',
            regime_at_entry   TEXT,
            entry_price       REAL NOT NULL,
            stop_initial      REAL NOT NULL,
            stop_current      REAL NOT NULL,
            stop_state        TEXT NOT NULL DEFAULT 'INITIAL',
            tp1               REAL,
            tp2               REAL,
            position_size_pct REAL NOT NULL,
            peak_price        REAL,
            half_closed       INTEGER DEFAULT 0,
            status            TEXT NOT NULL DEFAULT 'OPEN',
            close_price       REAL,
            pnl_r             REAL,
            pnl_pct           REAL,
            opened_at         TEXT DEFAULT (datetime('now')),
            closed_at         TEXT,
            atr_at_entry      REAL,
            strategy_version  TEXT,
            notes             TEXT
        );

        CREATE TABLE IF NOT EXISTS position_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER NOT NULL REFERENCES trades(id),
            current_price   REAL NOT NULL,
            stop_loss       REAL NOT NULL,
            stop_state      TEXT NOT NULL,
            unrealised_r    REAL,
            action_taken    TEXT NOT NULL DEFAULT 'NONE',
            action_detail   TEXT,
            recorded_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS strategy_versions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            version             TEXT NOT NULL UNIQUE,
            created_at          TEXT DEFAULT (datetime('now')),
            created_by          TEXT NOT NULL,
            rules_file          TEXT NOT NULL,
            summary             TEXT,
            changes_json        TEXT,
            status              TEXT NOT NULL DEFAULT 'PROPOSED',
            activated_at        TEXT,
            performance_snapshot TEXT
        );

        CREATE TABLE IF NOT EXISTS weekly_reviews (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start        TEXT NOT NULL,
            created_at        TEXT DEFAULT (datetime('now')),
            signals_total     INTEGER,
            claude_calls      INTEGER,
            entries_taken     INTEGER,
            win_rate          REAL,
            avg_r             REAL,
            proposed_version  TEXT,
            recommendation    TEXT,
            full_analysis     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_signals_pair_created
            ON signals(pair, created_at);
        CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_pair_status
            ON trades(pair, status);
        CREATE INDEX IF NOT EXISTS idx_position_history_trade
            ON position_history(trade_id, recorded_at);
    """)
    conn.commit()
    logger.info("Database initialized: %s", config.DB_PATH)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def insert_signal(
    pair: str,
    regime: str,
    score: int,
    signals_dict: dict,
    indicators_dict: dict,
    claude_called: bool = False,
    claude_decision: Optional[str] = None,
    claude_reasoning: Optional[str] = None,
    claude_confidence: Optional[str] = None,
    strategy_version: Optional[str] = None,
    source: str = "SCANNER",
) -> int:
    """Insert a signal record. Returns the new signal ID."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO signals
           (pair, regime, score, signals_json, indicators_json,
            claude_called, claude_decision, claude_reasoning,
            claude_confidence, strategy_version, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pair, regime, score,
            json.dumps(signals_dict),
            json.dumps(indicators_dict),
            int(claude_called),
            claude_decision,
            claude_reasoning,
            claude_confidence,
            strategy_version,
            source,
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_signal_claude(
    signal_id: int,
    claude_decision: str,
    claude_reasoning: str,
    claude_confidence: str,
) -> None:
    """Update a signal record with Claude's decision after the call."""
    conn = get_conn()
    conn.execute(
        """UPDATE signals
           SET claude_called=1, claude_decision=?, claude_reasoning=?,
               claude_confidence=?
           WHERE id=?""",
        (claude_decision, claude_reasoning, claude_confidence, signal_id),
    )
    conn.commit()


def get_signals_since(since: datetime) -> list[sqlite3.Row]:
    """Return all signals since a given datetime."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM signals WHERE created_at >= ? ORDER BY created_at DESC",
        (since.isoformat(),),
    ).fetchall()


def get_signals_for_review(since: datetime) -> list[sqlite3.Row]:
    """Return signals with score >= MIN_SCORE_LOG for weekly review."""
    conn = get_conn()
    return conn.execute(
        """SELECT * FROM signals
           WHERE created_at >= ? AND score >= ?
           ORDER BY created_at""",
        (since.isoformat(), config.MIN_SCORE_LOG),
    ).fetchall()


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def insert_trade(
    pair: str,
    direction: str,
    source: str,
    regime_at_entry: str,
    entry_price: float,
    stop_initial: float,
    tp1: float,
    tp2: float,
    position_size_pct: float,
    atr_at_entry: float,
    signal_id: Optional[int] = None,
    strategy_version: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """Insert a new trade. Returns the trade ID."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO trades
           (signal_id, pair, direction, source, regime_at_entry,
            entry_price, stop_initial, stop_current, tp1, tp2,
            position_size_pct, peak_price, atr_at_entry,
            strategy_version, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            signal_id, pair, direction, source, regime_at_entry,
            entry_price, stop_initial, stop_initial,  # stop_current = stop_initial
            tp1, tp2, position_size_pct,
            entry_price,  # peak_price starts at entry
            atr_at_entry, strategy_version, notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_trade(trade_id: int, **kwargs) -> None:
    """Update specific columns on a trade. Pass column=value as kwargs."""
    if not kwargs:
        return
    conn = get_conn()
    set_clause = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [trade_id]
    conn.execute(f"UPDATE trades SET {set_clause} WHERE id=?", values)
    conn.commit()


def get_open_trades() -> list[sqlite3.Row]:
    """Return all trades with status='OPEN'."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM trades WHERE status='OPEN' ORDER BY opened_at"
    ).fetchall()


def get_trade(trade_id: int) -> Optional[sqlite3.Row]:
    """Return a single trade by ID."""
    conn = get_conn()
    return conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()


def count_open_trades() -> int:
    """Return the number of currently open trades."""
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM trades WHERE status='OPEN'").fetchone()
    return row["cnt"]


def get_closed_trades(since: Optional[datetime] = None, limit: int = 50) -> list[sqlite3.Row]:
    """Return closed trades, optionally filtered by date."""
    conn = get_conn()
    if since:
        return conn.execute(
            """SELECT * FROM trades
               WHERE status != 'OPEN' AND closed_at >= ?
               ORDER BY closed_at DESC LIMIT ?""",
            (since.isoformat(), limit),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM trades WHERE status != 'OPEN' ORDER BY closed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def get_open_trade_for_pair(pair: str) -> Optional[sqlite3.Row]:
    """Return the open trade for a given pair, if any."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM trades WHERE pair=? AND status='OPEN'", (pair,)
    ).fetchone()


# ---------------------------------------------------------------------------
# Position History
# ---------------------------------------------------------------------------

def insert_position_snapshot(
    trade_id: int,
    current_price: float,
    stop_loss: float,
    stop_state: str,
    unrealised_r: float,
    action_taken: str = "NONE",
    action_detail: Optional[str] = None,
) -> None:
    """Insert a position history snapshot."""
    conn = get_conn()
    conn.execute(
        """INSERT INTO position_history
           (trade_id, current_price, stop_loss, stop_state,
            unrealised_r, action_taken, action_detail)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (trade_id, current_price, stop_loss, stop_state,
         unrealised_r, action_taken, action_detail),
    )
    conn.commit()


def get_position_history(trade_id: int) -> list[sqlite3.Row]:
    """Return full position history for a trade."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM position_history WHERE trade_id=? ORDER BY recorded_at",
        (trade_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Strategy Versions
# ---------------------------------------------------------------------------

def insert_strategy_version(
    version: str,
    created_by: str,
    rules_file: str,
    summary: str,
    changes_json: str,
    status: str = "PROPOSED",
) -> int:
    """Insert a new strategy version. Returns the version ID."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO strategy_versions
           (version, created_by, rules_file, summary, changes_json, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (version, created_by, rules_file, summary, changes_json, status),
    )
    conn.commit()
    return cur.lastrowid


def get_active_strategy() -> Optional[sqlite3.Row]:
    """Return the currently active strategy version."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM strategy_versions WHERE status='ACTIVE' ORDER BY activated_at DESC LIMIT 1"
    ).fetchone()


def get_strategy_version(version: str) -> Optional[sqlite3.Row]:
    """Return a specific strategy version by name."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM strategy_versions WHERE version=?", (version,)
    ).fetchone()


def get_strategy_versions() -> list[sqlite3.Row]:
    """Return all strategy versions."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM strategy_versions ORDER BY created_at DESC"
    ).fetchall()


def update_strategy_status(version: str, status: str) -> None:
    """Update the status of a strategy version."""
    conn = get_conn()
    updates = {"status": status}
    if status == "ACTIVE":
        updates["activated_at"] = datetime.now(timezone.utc).isoformat()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [version]
    conn.execute(f"UPDATE strategy_versions SET {set_clause} WHERE version=?", values)
    conn.commit()


# ---------------------------------------------------------------------------
# Weekly Reviews
# ---------------------------------------------------------------------------

def insert_weekly_review(
    week_start: str,
    signals_total: int,
    claude_calls: int,
    entries_taken: int,
    win_rate: float,
    avg_r: float,
    proposed_version: Optional[str],
    recommendation: str,
    full_analysis: str,
) -> int:
    """Insert a weekly review record. Returns the review ID."""
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO weekly_reviews
           (week_start, signals_total, claude_calls, entries_taken,
            win_rate, avg_r, proposed_version, recommendation, full_analysis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (week_start, signals_total, claude_calls, entries_taken,
         win_rate, avg_r, proposed_version, recommendation, full_analysis),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_review() -> Optional[sqlite3.Row]:
    """Return the most recent weekly review."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM weekly_reviews ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def get_reviews(limit: int = 10) -> list[sqlite3.Row]:
    """Return recent weekly reviews."""
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM weekly_reviews ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()


# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------

def get_daily_pnl() -> float:
    """Return sum of realized PnL (%) for trades closed today."""
    conn = get_conn()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        """SELECT COALESCE(SUM(pnl_pct), 0.0) as total
           FROM trades
           WHERE status != 'OPEN' AND closed_at >= ?""",
        (today,),
    ).fetchone()
    return row["total"]


def get_daily_trade_count() -> int:
    """Return number of trades opened today."""
    conn = get_conn()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM trades WHERE opened_at >= ?", (today,)
    ).fetchone()
    return row["cnt"]


def get_performance_stats(days: int = 7) -> dict:
    """Return performance statistics over the given number of days."""
    conn = get_conn()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    trades = conn.execute(
        """SELECT * FROM trades
           WHERE status != 'OPEN' AND closed_at >= ?""",
        (since,),
    ).fetchall()

    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "best_r": 0.0,
            "worst_r": 0.0,
            "total_pnl_pct": 0.0,
            "profit_factor": 0.0,
        }

    total = len(trades)
    winners = [t for t in trades if (t["pnl_r"] or 0) > 0]
    losers = [t for t in trades if (t["pnl_r"] or 0) <= 0]
    all_r = [t["pnl_r"] or 0 for t in trades]

    gross_profit = sum(t["pnl_r"] for t in winners) if winners else 0
    gross_loss = abs(sum(t["pnl_r"] for t in losers)) if losers else 0

    return {
        "total_trades": total,
        "win_rate": len(winners) / total * 100 if total else 0,
        "avg_r": sum(all_r) / total if total else 0,
        "best_r": max(all_r) if all_r else 0,
        "worst_r": min(all_r) if all_r else 0,
        "total_pnl_pct": sum(t["pnl_pct"] or 0 for t in trades),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    test_db = "test_atlas.db"
    config.DB_PATH = test_db

    # Reset connection for test DB
    _conn = None

    try:
        init_db()
        print("[OK] Database initialized")

        # Test signals
        sid = insert_signal("BTCUSDT", "TRENDING", 7,
                            {"trend": True, "ema_stack": True},
                            {"rsi_1h": 55, "ema50_4h": 67000},
                            strategy_version="v1")
        print(f"[OK] Signal inserted: id={sid}")

        signals = get_signals_since(datetime(2020, 1, 1))
        assert len(signals) == 1
        print(f"[OK] Signal queried: {dict(signals[0])}")

        # Test trades
        tid = insert_trade("BTCUSDT", "LONG", "SCANNER", "TRENDING",
                           67000.0, 66500.0, 67500.0, 68500.0,
                           1.5, 450.0, signal_id=sid, strategy_version="v1")
        print(f"[OK] Trade inserted: id={tid}")

        assert count_open_trades() == 1
        print("[OK] Open trade count: 1")

        update_trade(tid, stop_current=66800.0, stop_state="PROTECTED")
        trade = get_trade(tid)
        assert trade["stop_current"] == 66800.0
        assert trade["stop_state"] == "PROTECTED"
        print("[OK] Trade updated")

        # Test position history
        insert_position_snapshot(tid, 67200.0, 66800.0, "PROTECTED", 0.4, "RAISE_STOP")
        history = get_position_history(tid)
        assert len(history) == 1
        print("[OK] Position snapshot inserted")

        # Close trade
        update_trade(tid, status="CLOSED_SL", close_price=66800.0,
                     pnl_r=-0.4, pnl_pct=-0.6,
                     closed_at=datetime.now(timezone.utc).isoformat())
        assert count_open_trades() == 0
        print("[OK] Trade closed")

        # Test strategy versions
        sv_id = insert_strategy_version("v1", "MANUAL", "strategy_v1.md",
                                        "Initial baseline strategy",
                                        "[]", "ACTIVE")
        print(f"[OK] Strategy version inserted: id={sv_id}")

        active = get_active_strategy()
        assert active["version"] == "v1"
        print("[OK] Active strategy: v1")

        # Test weekly reviews
        wr_id = insert_weekly_review("2026-03-23", 50, 10, 5,
                                     60.0, 0.8, "v2", "ADOPT",
                                     "Full analysis text here")
        print(f"[OK] Weekly review inserted: id={wr_id}")

        # Test performance stats
        stats = get_performance_stats(days=30)
        print(f"[OK] Performance stats: {stats}")

        print("\n=== ALL TESTS PASSED ===")

    finally:
        if _conn:
            _conn.close()
            _conn = None
        if os.path.exists(test_db):
            os.remove(test_db)
