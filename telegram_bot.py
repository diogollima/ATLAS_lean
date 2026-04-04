"""
ATLAS Lean v2.0 — Telegram Bot
Commands + notifications interface using python-telegram-bot v21 async.
All commands gated by TELEGRAM_CHAT_ID for security.
"""

import json
import logging
from datetime import datetime, timezone, timedelta

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

import config
import db
from paper_trader import PaperTrader, TradeSetup
from indicators import compute_indicators, compute_volume_metrics, summarize_indicators
from regime_detector import detect_regime

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram command interface + notification sender."""

    def __init__(self, atlas=None) -> None:
        self.atlas = atlas  # Main AtlasLean instance (injected)
        self.trader = PaperTrader()
        self._app = None

    def _auth(self, update: Update) -> bool:
        """Check if message is from authorized chat."""
        if config.TELEGRAM_CHAT_ID == 0:
            return True  # Not configured — allow all (dev mode)
        return update.effective_chat.id == config.TELEGRAM_CHAT_ID

    async def start(self) -> None:
        """Initialize and start the Telegram bot (non-blocking polling)."""
        if not config.TELEGRAM_BOT_TOKEN:
            logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
            return

        self._app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .build()
        )

        # Status commands
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("history", self._cmd_history))
        self._app.add_handler(CommandHandler("performance", self._cmd_performance))
        self._app.add_handler(CommandHandler("regime", self._cmd_regime))
        self._app.add_handler(CommandHandler("help", self._cmd_help))

        # Trade commands
        self._app.add_handler(CommandHandler("enter", self._cmd_enter))
        self._app.add_handler(CommandHandler("close", self._cmd_close))
        self._app.add_handler(CommandHandler("closehalf", self._cmd_closehalf))
        self._app.add_handler(CommandHandler("movestop", self._cmd_movestop))
        self._app.add_handler(CommandHandler("pause", self._cmd_pause))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("halt", self._cmd_halt))

        # Strategy commands
        self._app.add_handler(CommandHandler("strategy", self._cmd_strategy))
        self._app.add_handler(CommandHandler("versions", self._cmd_versions))
        self._app.add_handler(CommandHandler("review", self._cmd_review))

        # Remote-control commands
        self._app.add_handler(CommandHandler("live", self._cmd_live))
        self._app.add_handler(CommandHandler("dryrun", self._cmd_dryrun))
        self._app.add_handler(CommandHandler("scan", self._cmd_scan))
        self._app.add_handler(CommandHandler("addpair", self._cmd_addpair))
        self._app.add_handler(CommandHandler("removepair", self._cmd_removepair))
        self._app.add_handler(CommandHandler("config", self._cmd_config))
        self._app.add_handler(CommandHandler("setthreshold", self._cmd_setthreshold))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    async def send_notification(self, message: str) -> None:
        """Send a notification message to the configured chat."""
        if not self._app or config.TELEGRAM_CHAT_ID == 0:
            logger.debug("Notification skipped (bot not configured): %s", message[:50])
            return
        try:
            await self._app.bot.send_message(
                chat_id=config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("Failed to send notification: %s", e)

    # ------------------------------------------------------------------
    # Status Commands
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return
        await update.message.reply_text(
            "ATLAS Lean v2.0\n"
            f"Monitoring: {', '.join(config.PAIRS)}\n"
            f"Max trades: {config.MAX_SIMULTANEOUS_TRADES}\n"
            f"Max risk: {config.MAX_RISK_PCT}%\n"
            "Type /help for commands"
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return
        await update.message.reply_text(
            "<b>Status</b>\n"
            "/status — System overview\n"
            "/positions — Open trades\n"
            "/history [n] — Last n closed trades\n"
            "/performance — 7-day stats\n"
            "/regime — Current regime per pair\n\n"
            "<b>Trade</b>\n"
            "/enter PAIR DIR ENTRY SL TP1 TP2 RISK%\n"
            "/close PAIR — Close trade on pair\n"
            "/closehalf PAIR — Close 50% on pair\n"
            "/movestop PAIR PRICE — Move stop\n"
            "/pause — Pause scanning\n"
            "/resume — Resume scanning\n"
            "/halt — Emergency halt\n\n"
            "<b>Remote Control</b>\n"
            "/live — Enable Claude-powered entries\n"
            "/dryrun — Analysis only (no entries)\n"
            "/scan — Trigger immediate scan cycle\n"
            "/addpair PAIR — Add pair to watchlist\n"
            "/removepair PAIR — Remove pair from watchlist\n"
            "/config — Show current runtime config\n"
            "/setthreshold REGIME VALUE — Adjust score threshold\n\n"
            "<b>Strategy</b>\n"
            "/strategy — Show active strategy\n"
            "/versions — List strategy versions\n"
            "/review — Trigger weekly review",
            parse_mode="HTML",
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        open_trades = db.get_open_trades()
        daily_pnl = db.get_daily_pnl()
        daily_count = db.get_daily_trade_count()

        paused = self.atlas.paused if self.atlas else False
        halted = self.atlas.halted if self.atlas else False
        cycle = self.atlas._cycle_count if self.atlas else 0

        if halted:
            state = "HALTED"
        elif paused:
            state = "PAUSED"
        else:
            state = "RUNNING"

        msg = (
            f"<b>ATLAS Lean Status</b>\n"
            f"State: {state}\n"
            f"Cycle: #{cycle}\n"
            f"Open trades: {len(open_trades)}/{config.MAX_SIMULTANEOUS_TRADES}\n"
            f"Daily PnL: {daily_pnl:.2f}%\n"
            f"Trades today: {daily_count}\n"
            f"Halt threshold: -{config.DAILY_DRAWDOWN_HALT_PCT}%"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        trades = db.get_open_trades()
        if not trades:
            await update.message.reply_text("No open positions.")
            return

        lines = ["<b>Open Positions</b>"]
        for t in trades:
            entry = t["entry_price"]
            stop = t["stop_current"] or t["stop_initial"]
            risk = abs(entry - t["stop_initial"])
            half = " [50% closed]" if t["half_closed"] else ""

            lines.append(
                f"\n#{t['id']} {t['pair']} {t['direction']}{half}\n"
                f"  Entry: {entry:.2f} | Stop: {stop:.2f}\n"
                f"  TP1: {t['tp1']:.2f} | TP2: {t['tp2']:.2f}\n"
                f"  State: {t['stop_state'] or 'INITIAL'} | Risk: {t['position_size_pct']:.1f}%"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_history(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        args = ctx.args
        limit = int(args[0]) if args else 10

        trades = db.get_closed_trades(limit=limit)
        if not trades:
            await update.message.reply_text("No closed trades.")
            return

        lines = [f"<b>Last {len(trades)} Closed Trades</b>"]
        for t in trades:
            pnl_r = t["pnl_r"] or 0
            pnl_pct = t["pnl_pct"] or 0
            emoji = "+" if pnl_r >= 0 else ""
            lines.append(
                f"#{t['id']} {t['pair']} {t['direction']} "
                f"{emoji}{pnl_r:.2f}R ({emoji}{pnl_pct:.2f}%) "
                f"[{t['status']}]"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_performance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        stats = db.get_performance_stats(days=7)
        msg = (
            f"<b>7-Day Performance</b>\n"
            f"Trades: {stats['total_trades']}\n"
            f"Win rate: {stats['win_rate']:.1f}%\n"
            f"Total PnL: {stats['total_pnl_r']:.2f}R ({stats['total_pnl_pct']:.2f}%)\n"
            f"Avg winner: {stats['avg_winner_r']:.2f}R\n"
            f"Avg loser: {stats['avg_loser_r']:.2f}R\n"
            f"Best: {stats['best_trade_r']:.2f}R\n"
            f"Worst: {stats['worst_trade_r']:.2f}R"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_regime(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        await update.message.reply_text("Scanning regimes... (next cycle)")

        if not self.atlas:
            return

        try:
            market_data = await self.atlas.scanner.scan_all()
            lines = ["<b>Current Regimes</b>"]

            for pair in config.PAIRS:
                pair_data = market_data.get(pair)
                if pair_data is None:
                    lines.append(f"{pair}: NO DATA")
                    continue

                klines = compute_indicators(pair_data["klines"])
                vol = compute_volume_metrics(
                    klines.get("1h"), klines.get("4h"),
                    pair_data.get("depth"), pair_data.get("book_ticker"),
                    pair_data.get("agg_trades"), pair_data.get("ticker24h"),
                )
                result = detect_regime(klines, vol)
                lines.append(
                    f"{pair}: {result.regime} ({result.confidence:.0%})"
                )

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"Regime scan failed: {e}")

    # ------------------------------------------------------------------
    # Trade Commands
    # ------------------------------------------------------------------

    async def _cmd_enter(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Manual entry: /enter PAIR DIR ENTRY SL TP1 TP2 RISK%"""
        if not self._auth(update):
            return

        args = ctx.args
        if not args or len(args) < 7:
            await update.message.reply_text(
                "Usage: /enter PAIR DIR ENTRY SL TP1 TP2 RISK%\n"
                "Example: /enter BTCUSDT LONG 69000 68500 69750 70500 1.5"
            )
            return

        try:
            pair = args[0].upper()
            direction = args[1].upper()
            entry = float(args[2])
            sl = float(args[3])
            tp1 = float(args[4])
            tp2 = float(args[5])
            risk_pct = float(args[6])

            if pair not in config.PAIRS:
                await update.message.reply_text(
                    f"Unknown pair: {pair}\nAllowed: {', '.join(config.PAIRS)}"
                )
                return

            if direction not in ("LONG", "SHORT"):
                await update.message.reply_text("Direction must be LONG or SHORT")
                return

            setup = TradeSetup(
                pair=pair,
                direction=direction,
                entry_price=entry,
                stop_initial=sl,
                tp1=tp1,
                tp2=tp2,
                position_size_pct=risk_pct,
                atr_at_entry=0,
                source="MANUAL",
                regime="MANUAL",
                notes="Manual Telegram entry",
            )

            trade_id, msg = self.trader.open_trade(setup)
            await update.message.reply_text(f"Trade opened!\n{msg}")

        except ValueError as e:
            await update.message.reply_text(f"Rejected: {e}")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Close trade: /close PAIR"""
        if not self._auth(update):
            return

        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /close PAIR")
            return

        pair = args[0].upper()
        trade = db.get_open_trade_for_pair(pair)
        if not trade:
            await update.message.reply_text(f"No open trade for {pair}")
            return

        # Use entry price as exit for manual close (paper trading)
        # In reality, user should provide price or we fetch it
        exit_price = trade["entry_price"]
        if self.atlas:
            try:
                data = await self.atlas.scanner.scan_all()
                pd = data.get(pair, {}).get("ticker24h", {})
                if pd and "lastPrice" in pd:
                    exit_price = float(pd["lastPrice"])
            except Exception:
                pass

        try:
            result = self.trader.close_trade(trade["id"], exit_price, "MANUAL")
            await update.message.reply_text(
                f"Trade #{result['trade_id']} closed\n"
                f"{pair} @ {exit_price:.2f}\n"
                f"PnL: {result['pnl_r']:.2f}R ({result['pnl_pct']:.2f}%)"
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_closehalf(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Close half: /closehalf PAIR"""
        if not self._auth(update):
            return

        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /closehalf PAIR")
            return

        pair = args[0].upper()
        trade = db.get_open_trade_for_pair(pair)
        if not trade:
            await update.message.reply_text(f"No open trade for {pair}")
            return

        # Fetch current price
        current_price = trade["entry_price"]
        if self.atlas:
            try:
                data = await self.atlas.scanner.scan_all()
                pd = data.get(pair, {}).get("ticker24h", {})
                if pd and "lastPrice" in pd:
                    current_price = float(pd["lastPrice"])
            except Exception:
                pass

        try:
            result = self.trader.close_half(trade["id"], current_price)
            await update.message.reply_text(
                f"50% closed on #{result['trade_id']} {pair}\n"
                f"@ {current_price:.2f} ({result['pnl_r_at_half']:.2f}R)"
            )
        except ValueError as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_movestop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Move stop: /movestop PAIR PRICE"""
        if not self._auth(update):
            return

        args = ctx.args
        if not args or len(args) < 2:
            await update.message.reply_text("Usage: /movestop PAIR PRICE")
            return

        pair = args[0].upper()
        new_stop = float(args[1])

        trade = db.get_open_trade_for_pair(pair)
        if not trade:
            await update.message.reply_text(f"No open trade for {pair}")
            return

        # Enforce IRON RULE: stop only moves toward profit
        current_stop = trade["stop_current"] or trade["stop_initial"]
        if trade["direction"] == "LONG" and new_stop < current_stop:
            await update.message.reply_text(
                f"REJECTED: LONG stop can only move UP\n"
                f"Current: {current_stop:.2f} | Requested: {new_stop:.2f}"
            )
            return
        if trade["direction"] == "SHORT" and new_stop > current_stop:
            await update.message.reply_text(
                f"REJECTED: SHORT stop can only move DOWN\n"
                f"Current: {current_stop:.2f} | Requested: {new_stop:.2f}"
            )
            return

        db.update_trade(trade["id"], stop_current=new_stop)
        await update.message.reply_text(
            f"Stop moved on #{trade['id']} {pair}\n"
            f"{current_stop:.2f} -> {new_stop:.2f}"
        )

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return
        if self.atlas:
            self.atlas.paused = True
        await update.message.reply_text("System PAUSED. Use /resume to restart.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return
        if self.atlas:
            self.atlas.paused = False
            self.atlas.halted = False
        await update.message.reply_text("System RESUMED.")

    async def _cmd_halt(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return
        if self.atlas:
            self.atlas.halted = True
        await update.message.reply_text(
            "EMERGENCY HALT activated.\n"
            "No new scans or trades. Use /resume to restart."
        )

    # ------------------------------------------------------------------
    # Remote Control Commands
    # ------------------------------------------------------------------

    async def _cmd_live(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Enable live mode: Claude evaluates setups and opens trades."""
        if not self._auth(update):
            return
        if not self.atlas:
            await update.message.reply_text("Atlas not connected.")
            return
        self.atlas.live_mode = True
        await update.message.reply_text(
            "LIVE MODE enabled.\n"
            "Claude will now evaluate setups and open paper trades.\n"
            "Use /dryrun to return to analysis-only mode."
        )

    async def _cmd_dryrun(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Disable live mode: analysis and Telegram alerts only, no entries."""
        if not self._auth(update):
            return
        if not self.atlas:
            await update.message.reply_text("Atlas not connected.")
            return
        self.atlas.live_mode = False
        await update.message.reply_text(
            "DRY-RUN mode enabled.\n"
            "Claude calls are disabled — analysis and alerts only."
        )

    async def _cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Request an immediate scan cycle (skips the 5-minute wait)."""
        if not self._auth(update):
            return
        if not self.atlas:
            await update.message.reply_text("Atlas not connected.")
            return
        if self.atlas.halted:
            await update.message.reply_text("System is HALTED. Use /resume first.")
            return
        if self.atlas.paused:
            await update.message.reply_text("System is PAUSED. Use /resume first.")
            return
        self.atlas._scan_requested = True
        await update.message.reply_text("Immediate scan requested. Results incoming shortly.")

    async def _cmd_addpair(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Add a pair to the watchlist: /addpair PAIR"""
        if not self._auth(update):
            return
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /addpair PAIR\nExample: /addpair XRPUSDT")
            return
        pair = args[0].upper()
        if pair in config.PAIRS:
            await update.message.reply_text(f"{pair} is already in the watchlist.")
            return
        config.PAIRS.append(pair)
        await update.message.reply_text(
            f"{pair} added to watchlist.\n"
            f"Now monitoring: {', '.join(config.PAIRS)}"
        )

    async def _cmd_removepair(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Remove a pair from the watchlist: /removepair PAIR"""
        if not self._auth(update):
            return
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /removepair PAIR\nExample: /removepair XRPUSDT")
            return
        pair = args[0].upper()
        if pair not in config.PAIRS:
            await update.message.reply_text(f"{pair} is not in the watchlist.")
            return
        if len(config.PAIRS) <= 1:
            await update.message.reply_text("Cannot remove last pair from watchlist.")
            return
        config.PAIRS.remove(pair)
        await update.message.reply_text(
            f"{pair} removed from watchlist.\n"
            f"Now monitoring: {', '.join(config.PAIRS)}"
        )

    async def _cmd_config(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Show current runtime configuration."""
        if not self._auth(update):
            return
        live = self.atlas.live_mode if self.atlas else False
        state = "HALTED" if (self.atlas and self.atlas.halted) else \
                "PAUSED" if (self.atlas and self.atlas.paused) else "RUNNING"
        msg = (
            f"<b>ATLAS Runtime Config</b>\n"
            f"Mode: {'LIVE' if live else 'DRY-RUN'}\n"
            f"State: {state}\n"
            f"Pairs: {', '.join(config.PAIRS)}\n\n"
            f"<b>Score Thresholds</b>\n"
            f"TRENDING: {config.SCORE_THRESHOLD_TRENDING}/9\n"
            f"PULLBACK: {config.SCORE_THRESHOLD_PULLBACK}/9\n"
            f"BREAKOUT: {config.SCORE_THRESHOLD_BREAKOUT}/9\n\n"
            f"<b>Risk (hardcoded)</b>\n"
            f"Max risk/trade: {config.MAX_RISK_PCT}%\n"
            f"Max open trades: {config.MAX_SIMULTANEOUS_TRADES}\n"
            f"Daily halt: -{config.DAILY_DRAWDOWN_HALT_PCT}%\n"
            f"Min R:R: {config.MIN_RR_RATIO}\n\n"
            f"<b>Timing</b>\n"
            f"Scan interval: {config.SCAN_INTERVAL_SECONDS}s\n"
            f"Max trade hours: {config.MAX_TRADE_HOURS}h\n"
            f"Active strategy: {config.ACTIVE_STRATEGY_FILE}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_setthreshold(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Adjust a score threshold: /setthreshold REGIME VALUE"""
        if not self._auth(update):
            return
        args = ctx.args
        if not args or len(args) < 2:
            await update.message.reply_text(
                "Usage: /setthreshold REGIME VALUE\n"
                "Regimes: TRENDING, PULLBACK, BREAKOUT\n"
                "Example: /setthreshold PULLBACK 5"
            )
            return
        regime = args[0].upper()
        try:
            value = int(args[1])
        except ValueError:
            await update.message.reply_text("Value must be an integer (1-9).")
            return
        if value < 1 or value > 9:
            await update.message.reply_text("Threshold must be between 1 and 9.")
            return
        if regime == "TRENDING":
            config.SCORE_THRESHOLD_TRENDING = value
        elif regime == "PULLBACK":
            config.SCORE_THRESHOLD_PULLBACK = value
        elif regime == "BREAKOUT":
            config.SCORE_THRESHOLD_BREAKOUT = value
        else:
            await update.message.reply_text(
                f"Unknown regime: {regime}\nValid: TRENDING, PULLBACK, BREAKOUT"
            )
            return
        await update.message.reply_text(
            f"Threshold updated: {regime} → {value}/9\n"
            f"TRENDING={config.SCORE_THRESHOLD_TRENDING} "
            f"PULLBACK={config.SCORE_THRESHOLD_PULLBACK} "
            f"BREAKOUT={config.SCORE_THRESHOLD_BREAKOUT}"
        )

    # ------------------------------------------------------------------
    # Strategy Commands
    # ------------------------------------------------------------------

    async def _cmd_strategy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        import os
        path = os.path.join(config.STRATEGY_DIR, config.ACTIVE_STRATEGY_FILE)
        try:
            with open(path, "r") as f:
                content = f.read()
            # Telegram message limit is 4096 chars
            if len(content) > 3900:
                content = content[:3900] + "\n\n... (truncated)"
            await update.message.reply_text(
                f"<b>Active: {config.ACTIVE_STRATEGY_FILE}</b>\n\n"
                f"<pre>{content}</pre>",
                parse_mode="HTML",
            )
        except Exception as e:
            await update.message.reply_text(f"Error reading strategy: {e}")

    async def _cmd_versions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        versions = db.get_strategy_versions()
        if not versions:
            await update.message.reply_text("No strategy versions in DB.")
            return

        lines = ["<b>Strategy Versions</b>"]
        for v in versions:
            lines.append(
                f"\n{v['version']} [{v['status']}]\n"
                f"  By: {v['created_by']} | {v['created_at']}\n"
                f"  {v['summary'] or 'No summary'}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_review(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._auth(update):
            return

        await update.message.reply_text("Triggering weekly review... This may take 30-60s.")

        try:
            # Import here to avoid circular import at module level
            from weekly_review import WeeklyReviewer
            reviewer = WeeklyReviewer()
            result = await reviewer.run_review()

            if result:
                await update.message.reply_text(
                    f"<b>Weekly Review Complete</b>\n\n"
                    f"Recommendation: {result.get('recommendation', 'N/A')}\n"
                    f"Proposed version: {result.get('proposed_version', 'None')}\n\n"
                    f"Use /versions to see details.",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text("Review completed but no changes proposed.")

        except Exception as e:
            await update.message.reply_text(f"Review failed: {e}")


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== TELEGRAM BOT MODULE TEST ===")
    print("[OK] Module imports successfully")

    bot = TelegramBot()
    print(f"[OK] TelegramBot created (app={bot._app})")
    print(f"[OK] Token configured: {'yes' if config.TELEGRAM_BOT_TOKEN else 'no'}")
    print(f"[OK] Chat ID: {config.TELEGRAM_CHAT_ID}")

    # Verify all handler methods exist
    handlers = [
        "_cmd_start", "_cmd_help", "_cmd_status", "_cmd_positions",
        "_cmd_history", "_cmd_performance", "_cmd_regime",
        "_cmd_enter", "_cmd_close", "_cmd_closehalf", "_cmd_movestop",
        "_cmd_pause", "_cmd_resume", "_cmd_halt",
        "_cmd_live", "_cmd_dryrun", "_cmd_scan",
        "_cmd_addpair", "_cmd_removepair", "_cmd_config", "_cmd_setthreshold",
        "_cmd_strategy", "_cmd_versions", "_cmd_review",
    ]
    for h in handlers:
        assert hasattr(bot, h), f"Missing handler: {h}"
    print(f"[OK] All {len(handlers)} command handlers present")

    print("\n=== ALL TELEGRAM BOT TESTS PASSED ===")
