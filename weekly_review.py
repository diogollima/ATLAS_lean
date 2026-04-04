"""
ATLAS Lean v2.0 — Weekly Review
Gathers 7 days of trading data, calls Claude Sonnet 4.6 for analysis,
saves proposed strategy version. Auto-triggered Sunday 00:00 UTC or via /review.
"""

import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

import config
import db
from claude_client import ClaudeClient
from strategy_manager import StrategyManager

logger = logging.getLogger(__name__)


class WeeklyReviewer:
    """Orchestrates the weekly self-improvement review cycle."""

    def __init__(self) -> None:
        self.claude = ClaudeClient()
        self.sm = StrategyManager()

    def _gather_performance_data(self) -> dict:
        """
        Collect 7 days of signals, trades, and outcomes.
        Produces a structured summary (~4000 tokens) for Claude.
        """
        since = datetime.now(timezone.utc) - timedelta(days=7)

        # Closed trades
        closed = db.get_closed_trades(since=since, limit=200)
        trades_list = []
        for t in closed:
            trades_list.append({
                "id": t["id"],
                "pair": t["pair"],
                "direction": t["direction"],
                "regime": t["regime_at_entry"],
                "entry": t["entry_price"],
                "exit": t["close_price"],
                "pnl_r": t["pnl_r"],
                "pnl_pct": t["pnl_pct"],
                "status": t["status"],
                "stop_state": t["stop_state"],
                "source": t["source"],
                "strategy_version": t["strategy_version"],
            })

        # Signals logged this week
        signals = db.get_signals_for_review(since=since)
        signal_summary = {
            "total": len(signals),
            "by_regime": {},
            "claude_called": 0,
            "claude_entered": 0,
            "skipped_score": 0,
        }
        for s in signals:
            regime = s["regime"]
            signal_summary["by_regime"].setdefault(regime, {"count": 0, "avg_score": 0.0})
            signal_summary["by_regime"][regime]["count"] += 1
            signal_summary["by_regime"][regime]["avg_score"] = round(
                (signal_summary["by_regime"][regime]["avg_score"] * (
                    signal_summary["by_regime"][regime]["count"] - 1
                ) + (s["score"] or 0)) / signal_summary["by_regime"][regime]["count"], 2
            )
            if s["claude_called"]:
                signal_summary["claude_called"] += 1
                if s["claude_decision"] == "ENTER":
                    signal_summary["claude_entered"] += 1
            else:
                signal_summary["skipped_score"] += 1

        # Performance stats
        stats = db.get_performance_stats(days=7)

        # Recent weekly reviews for context
        prior_review = db.get_latest_review()
        prior_summary = None
        if prior_review:
            prior_summary = {
                "date": prior_review["created_at"],
                "recommendation": prior_review["recommendation"],
                "outcome": prior_review.get("outcome"),
            }

        return {
            "period": {
                "from": since.isoformat(),
                "to": datetime.now(timezone.utc).isoformat(),
            },
            "performance": stats,
            "trades": trades_list,
            "signals": signal_summary,
            "active_strategy": config.ACTIVE_STRATEGY_FILE,
            "prior_review": prior_summary,
        }

    def _next_version_name(self) -> str:
        """Generate the next strategy version filename."""
        versions = db.get_strategy_versions()
        # Find highest v-number
        max_v = 1
        for v in versions:
            name = v["version"]
            if name.startswith("strategy_v") and name.endswith(".md"):
                try:
                    n = int(name[len("strategy_v"):-len(".md")].split("_")[0])
                    max_v = max(max_v, n)
                except ValueError:
                    pass
        now = datetime.now(timezone.utc)
        return f"strategy_v{max_v + 1}_{now.strftime('%Y%m%d')}.md"

    async def run_review(self) -> Optional[dict]:
        """
        Full review cycle:
        1. Gather 7-day performance data
        2. Call Claude Sonnet 4.6
        3. Save proposed strategy version if recommendation is ADOPT or EXTEND
        Returns result dict or None on failure.
        """
        logger.info("Starting weekly review...")

        # Need at least a few trades to review
        since = datetime.now(timezone.utc) - timedelta(days=7)
        closed = db.get_closed_trades(since=since, limit=200)
        if len(closed) < 3:
            logger.warning(
                "Only %d closed trades this week — skipping review (need >=3)", len(closed)
            )
            return None

        perf_data = self._gather_performance_data()
        current_strategy = self.sm.load_active_strategy()

        logger.info(
            "Gathered %d trades, %d signals — calling Sonnet 4.6",
            len(perf_data["trades"]),
            perf_data["signals"]["total"],
        )

        result = await self.claude.weekly_review(perf_data, current_strategy)

        if "error" in result:
            logger.error("Weekly review Claude call failed: %s", result["error"])
            return None

        recommendation = result.get("recommendation", "KEEP")
        logger.info(
            "Weekly review complete — recommendation: %s", recommendation
        )

        proposed_version_name = None

        # Save proposed strategy if Claude recommends change
        if recommendation in ("ADOPT", "EXTEND") and result.get("proposed_strategy"):
            version_name = self._next_version_name()
            changes = "\n".join(result.get("proposed_changes", []))
            summary = result.get("performance_summary", "")

            self.sm.save_new_version(
                name=version_name,
                text=result["proposed_strategy"],
                created_by=f"Claude {config.SONNET_MODEL}",
                summary=summary[:500],
                changes=changes,
            )
            proposed_version_name = version_name
            logger.info("Proposed version saved: %s", version_name)

        # Save weekly review record
        db.insert_weekly_review(
            week_start=perf_data["period"]["from"],
            signals_total=perf_data["signals"]["total"],
            claude_calls=perf_data["signals"]["claude_called"],
            entries_taken=perf_data["signals"]["claude_entered"],
            win_rate=perf_data["performance"].get("win_rate", 0),
            avg_r=perf_data["performance"].get("total_pnl_r", 0),
            proposed_version=proposed_version_name,
            recommendation=recommendation,
            full_analysis=json.dumps(result, default=str),
        )

        return {
            "recommendation": recommendation,
            "proposed_version": proposed_version_name,
            "performance_summary": result.get("performance_summary", ""),
            "proposed_changes": result.get("proposed_changes", []),
            "expected_impact": result.get("expected_impact", ""),
        }

    def should_run_now(self) -> bool:
        """
        Check if weekly review should trigger now.
        Runs Sunday 00:00–01:00 UTC (checked every 5 min by main loop).
        Also enforces minimum 6 days between reviews.
        """
        now = datetime.now(timezone.utc)

        # Sunday = weekday 6
        if now.weekday() != 6 or now.hour != 0:
            return False

        # Check if already ran this week
        last = db.get_latest_review()
        if last:
            last_dt_str = last["created_at"]
            try:
                last_dt = datetime.fromisoformat(last_dt_str.replace("Z", "+00:00"))
            except ValueError:
                last_dt = datetime.now(timezone.utc) - timedelta(days=8)
            if (now - last_dt).days < 6:
                return False

        return True


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_db = "test_review.db"
    config.DB_PATH = test_db
    db._conn = None

    try:
        db.init_db()
        reviewer = WeeklyReviewer()

        print("=== WEEKLY REVIEW TESTS ===")

        # Test _gather_performance_data (empty DB)
        data = reviewer._gather_performance_data()
        assert "period" in data
        assert "performance" in data
        assert "trades" in data
        assert "signals" in data
        print("[OK] Performance data gathered (empty DB)")

        # Test should_run_now (deterministic — just checks it returns bool)
        result = reviewer.should_run_now()
        assert isinstance(result, bool)
        print(f"[OK] should_run_now returns bool: {result}")

        # Test version name generation
        name = reviewer._next_version_name()
        assert name.startswith("strategy_v2_")
        assert name.endswith(".md")
        print(f"[OK] Next version name: {name}")

        # Test that run_review skips with insufficient data
        async def test_skip():
            result = await reviewer.run_review()
            assert result is None, "Should return None with <3 trades"
            print("[OK] run_review correctly skipped with insufficient data")

        asyncio.run(test_skip())

        print("\n=== ALL WEEKLY REVIEW TESTS PASSED ===")

    finally:
        if db._conn:
            db._conn.close()
            db._conn = None
        if os.path.exists(test_db):
            os.remove(test_db)
