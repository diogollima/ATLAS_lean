"""
ATLAS Lean v2.0 — Main Orchestrator
5-minute loop: scan → indicators → regime → signals → Claude → manage positions.
Entry point for the entire system.
"""

import asyncio
import html
import json
import logging
import time
from datetime import datetime, timezone

import config
import db
from scanner import BinanceScanner
from indicators import compute_indicators, compute_volume_metrics, summarize_indicators
from regime_detector import detect_regime
from signals import score_pair
from claude_client import ClaudeClient, ClaudeDecision
from paper_trader import PaperTrader, TradeSetup
from position_manager import PositionManager

logger = logging.getLogger(__name__)


class AtlasLean:
    """Main orchestrator — runs the 5-minute trading loop."""

    def __init__(self) -> None:
        db.init_db()
        self.scanner = BinanceScanner()
        self.claude = ClaudeClient()
        self.trader = PaperTrader()
        self.pm = PositionManager()
        self.paused = False
        self.halted = False
        self.live_mode = False  # False = analysis only; True = Claude calls + trade entry
        self._telegram = None  # Set externally by telegram_bot
        self._cycle_count = 0
        self._scan_requested = False  # Set by /scan command for immediate cycle

    def set_telegram(self, telegram_bot) -> None:
        """Inject Telegram bot for notifications."""
        self._telegram = telegram_bot

    async def notify(self, message: str) -> None:
        """Send notification via Telegram if available."""
        if self._telegram:
            try:
                await self._telegram.send_notification(message)
            except Exception as e:
                logger.error("Telegram notification failed: %s", e)

    async def run_cycle(self) -> dict:
        """
        Execute one complete 5-minute cycle.
        Returns summary dict of actions taken.
        """
        self._cycle_count += 1
        cycle_start = time.monotonic()
        summary = {
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals_logged": 0,
            "claude_calls": 0,
            "trades_opened": 0,
            "position_updates": [],
            "errors": [],
        }

        # Check halt/pause
        if self.halted:
            logger.info("System HALTED — skipping cycle")
            return summary
        if self.paused:
            logger.info("System PAUSED — skipping cycle")
            return summary

        try:
            # Step 1: Scan all pairs
            logger.info("Cycle #%d — scanning %d pairs...", self._cycle_count, len(config.PAIRS))
            market_data = await self.scanner.scan_all()

            if not market_data:
                logger.warning("No market data returned — skipping cycle")
                summary["errors"].append("No market data")
                return summary

            # Collect current prices for position management
            current_prices = {}
            all_klines = {}
            cycle_results = []  # Per-pair results for cycle summary

            # Step 2-5: Process each pair
            for pair in config.PAIRS:
                try:
                    pair_data = market_data.get(pair)
                    if pair_data is None:
                        logger.warning("No data for %s — skipping", pair)
                        continue

                    # Step 2: Compute indicators
                    klines = compute_indicators(pair_data["klines"])
                    vol_metrics = compute_volume_metrics(
                        klines.get("1h"), klines.get("4h"),
                        pair_data.get("depth"), pair_data.get("book_ticker"),
                        pair_data.get("agg_trades"), pair_data.get("ticker24h"),
                    )

                    # Get current price
                    if "1h" in klines and not klines["1h"].empty:
                        current_prices[pair] = float(klines["1h"]["close"].iloc[-1])
                    all_klines[pair] = klines

                    # Step 3: Detect regime
                    regime_result = detect_regime(klines, vol_metrics)
                    regime = regime_result.regime

                    # Step 4: Score signals
                    signal_result = score_pair(pair, klines, vol_metrics, regime)
                    score = signal_result.score

                    # Always compute indicators summary (needed for summary + analysis)
                    indicators_summary = summarize_indicators(klines, vol_metrics, pair)
                    threshold = config.get_score_threshold(regime)
                    open_count = db.count_open_trades()
                    would_call_claude = (
                        score >= threshold
                        and open_count < config.MAX_SIMULTANEOUS_TRADES
                        and not db.get_open_trade_for_pair(pair)
                    )

                    # Collect for cycle summary (all pairs)
                    cycle_results.append({
                        "pair": pair,
                        "regime": regime,
                        "score": score,
                        "would_call_claude": would_call_claude,
                    })

                    # Log signal to DB if score >= MIN_SCORE_LOG
                    signal_id = None
                    if score >= config.MIN_SCORE_LOG:
                        signal_id = db.insert_signal(
                            pair=pair,
                            regime=regime,
                            score=score,
                            signals_dict={k: int(v) for k, v in signal_result.signals.items()},
                            indicators_dict=indicators_summary,
                            strategy_version=config.ACTIVE_STRATEGY_FILE,
                        )
                        summary["signals_logged"] += 1

                    # Block automated entries if user has manual SPOT trades open.
                    # Scanner keeps running and reporting — only entry is suppressed.
                    if would_call_claude and self.trader.has_manual_trades():
                        logger.info(
                            "%s would call Claude but manual trade(s) active — auto-entry suppressed",
                            pair,
                        )
                        would_call_claude = False

                    # Step 5: Call Claude (live mode) or send analysis (dry-run)
                    if would_call_claude:
                        if self.live_mode:
                            decision = await self.claude.evaluate_entry(
                                pair=pair,
                                regime=regime,
                                score=score,
                                signals=signal_result.signals,
                                indicators=indicators_summary,
                                open_trades_count=open_count,
                            )
                            summary["claude_calls"] += 1
                            if signal_id is not None:
                                db.update_signal_claude(
                                    signal_id,
                                    decision.action,
                                    decision.reasoning,
                                    decision.confidence,
                                )
                            if decision.action == "ENTER":
                                await self._open_trade_from_decision(
                                    pair, regime, decision,
                                    signal_id or 0, indicators_summary, summary,
                                )
                            await self._send_analysis(
                                pair, regime, regime_result,
                                signal_result, indicators_summary,
                                threshold, would_call_claude, decision,
                            )
                        else:
                            await self._send_analysis(
                                pair, regime, regime_result,
                                signal_result, indicators_summary,
                                threshold, would_call_claude,
                            )

                except Exception as e:
                    logger.error("Error processing %s: %s", pair, e, exc_info=True)
                    summary["errors"].append(f"{pair}: {e}")

            # Step 6: Send cycle summary to Telegram
            if cycle_results:
                await self._send_cycle_summary(cycle_results)

            # Step 7: Manage open positions
            if current_prices:
                await self._manage_positions(current_prices, all_klines, summary)

            # Step 8: Weekly review check (Sunday 00:00 UTC)
            try:
                from weekly_review import WeeklyReviewer
                reviewer = WeeklyReviewer()
                if reviewer.should_run_now():
                    logger.info("Weekly review triggered")
                    result = await reviewer.run_review()
                    if result:
                        await self.notify(
                            f"WEEKLY REVIEW COMPLETE\n"
                            f"Recommendation: {result['recommendation']}\n"
                            f"Proposed version: {result.get('proposed_version') or 'None'}"
                        )
            except Exception as e:
                logger.error("Weekly review error: %s", e)

            # Step 9: Check daily halt
            if self.pm.check_daily_halt():
                logger.warning("DAILY DRAWDOWN HALT TRIGGERED")
                self.halted = True
                await self._emergency_halt(current_prices, summary)

        except Exception as e:
            logger.error("Cycle #%d failed: %s", self._cycle_count, e, exc_info=True)
            summary["errors"].append(f"Cycle error: {e}")

        elapsed = time.monotonic() - cycle_start
        logger.info(
            "Cycle #%d complete in %.1fs — signals=%d claude=%d trades=%d updates=%d errors=%d",
            self._cycle_count, elapsed,
            summary["signals_logged"], summary["claude_calls"],
            summary["trades_opened"], len(summary["position_updates"]),
            len(summary["errors"]),
        )

        return summary

    async def _send_analysis(
        self, pair: str, regime: str, regime_result,
        signal_result, indicators: dict,
        threshold: int, would_call_claude: bool,
        decision=None,
    ) -> None:
        """Format and send full signal analysis to Telegram."""
        score = signal_result.score
        price = indicators.get("price", 0)
        conf = regime_result.confidence

        # Header — show Claude's actual verdict in live mode, else dry-run label
        if decision is not None:
            verdict = f"CLAUDE: {decision.action} ({decision.confidence})"
        elif would_call_claude:
            verdict = "DRY-RUN (analysis only)"
        else:
            verdict = f"score {score}/{threshold} needed"
        lines = [
            f"<b>{'🔔 ' if would_call_claude else ''}SIGNAL ANALYSIS — {pair}</b>",
            f"Price: <b>${price:,.2f}</b>  |  Cycle #{self._cycle_count}",
            "",
            f"<b>REGIME: {regime}</b>  ({conf:.0%} confidence)",
        ]

        # Regime detail reasons
        rd = regime_result.details
        if rd:
            for k, v in list(rd.items())[:4]:
                lines.append(f"  • {html.escape(str(k))}: {html.escape(str(v))}")

        lines.append("")
        lines.append(f"<b>SCORE: {score}/9</b>  (threshold: {threshold})  → {verdict}")
        lines.append("")

        # Signal breakdown
        signal_names = {
            "trend_alignment": "Trend Alignment",
            "ema_stack":        "EMA Stack",
            "rsi_mode":         "RSI Mode",
            "macd":             "MACD",
            "structure_level":  "Structure Level",
            "volume_ratio":     "Volume Ratio",
            "taker_buy_bias":   "Taker Buy Bias",
            "order_book":       "Order Book",
            "regime_fit":       "Regime Fit",
        }
        lines.append("<b>SIGNALS:</b>")
        for key, label in signal_names.items():
            passed = signal_result.signals.get(key, False)
            detail = html.escape(signal_result.details.get(key, ""))
            icon = "✅" if passed else "❌"
            lines.append(f"  {icon} <b>{label}</b>: {detail}")

        lines.append("")
        lines.append("<b>KEY INDICATORS:</b>")

        # Trend
        ema50 = indicators.get("ema50_4h")
        ema200 = indicators.get("ema200_4h")
        stack = indicators.get("ema_stack_1h", "?")
        struct = indicators.get("structure_4h", "?")
        lines.append(f"  EMA50/200 (4H): {ema50:.0f} / {ema200:.0f}" if ema50 and ema200 else "  EMA: N/A")
        lines.append(f"  EMA Stack (1H): {html.escape(str(stack))}  |  Structure (4H): {html.escape(str(struct))}")

        # Momentum
        rsi15 = indicators.get("rsi_15m")
        rsi1h = indicators.get("rsi_1h")
        rsi4h = indicators.get("rsi_4h")
        macd_dir = html.escape(str(indicators.get("macd_direction_1h", "?")))
        stoch = indicators.get("stoch_k_4h")
        rsi_str = f"RSI: {rsi15:.0f}(15m) / {rsi1h:.0f}(1H) / {rsi4h:.0f}(4H)" if all(x is not None for x in [rsi15, rsi1h, rsi4h]) else "RSI: N/A"
        lines.append(f"  {rsi_str}")
        lines.append(f"  MACD (1H): {macd_dir}  |  Stoch K (4H): {stoch:.0f}" if stoch else f"  MACD (1H): {macd_dir}")

        # Volume
        vol_ratio = indicators.get("volume_ratio_1h")
        taker = indicators.get("taker_buy_ratio_1h")
        ob_bid = indicators.get("ob_bid_pct")
        ob_ask = 1 - ob_bid if ob_bid else None
        vol_str = f"Vol ratio: {vol_ratio:.2f}x" if vol_ratio else "Vol: N/A"
        taker_str = f"Taker buy: {taker:.1%}" if taker else ""
        ob_str = f"OB bid/ask: {ob_bid:.1%}/{ob_ask:.1%}" if ob_bid else ""
        lines.append(f"  {vol_str}  |  {taker_str}")
        if ob_str:
            lines.append(f"  {ob_str}")

        # Volatility
        atr = indicators.get("atr14_1h")
        atr_pct = indicators.get("atr_pct_1h")
        bb_pctb = indicators.get("bb_pctb_1h")
        if atr:
            lines.append(f"  ATR (1H): {atr:.2f} ({atr_pct:.2f}%)  |  BB %B: {bb_pctb:.2f}" if bb_pctb else f"  ATR (1H): {atr:.2f} ({atr_pct:.2f}%)")

        # 24h change
        chg = indicators.get("price_change_24h_pct")
        if chg is not None:
            lines.append(f"  24H change: {chg:+.2f}%")

        msg = "\n".join(lines)
        # Telegram message limit is 4096 chars
        if len(msg) > 4000:
            msg = msg[:4000] + "\n..."
        await self.notify(msg)

    async def _send_cycle_summary(self, cycle_results: list[dict]) -> None:
        """Send a compact per-cycle overview of all pairs to Telegram."""
        lines = [f"<b>CYCLE #{self._cycle_count} SUMMARY</b>  {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]
        lines.append("")
        for r in cycle_results:
            score_bar = "█" * r["score"] + "░" * (9 - r["score"])
            flag = " 🔔" if r["would_call_claude"] else ""
            lines.append(
                f"<b>{r['pair']}</b>  {r['regime'][:4]}  [{score_bar}] {r['score']}/9{flag}"
            )
        await self.notify("\n".join(lines))

    async def _open_trade_from_decision(
        self, pair: str, regime: str, decision: ClaudeDecision,
        signal_id: int, indicators: dict, summary: dict,
    ) -> None:
        """Validate and open a trade from a Claude ENTER decision."""
        try:
            atr = indicators.get("atr14_1h", 0)
            setup = TradeSetup(
                pair=pair,
                direction=decision.direction,
                entry_price=decision.entry_price,
                stop_initial=decision.stop_loss,
                tp1=decision.tp1,
                tp2=decision.tp2,
                position_size_pct=decision.position_size_pct,
                atr_at_entry=atr,
                source="SCANNER",
                regime=regime,
                signal_id=signal_id,
                strategy_version=config.ACTIVE_STRATEGY_FILE,
                notes=decision.reasoning[:200],
            )

            trade_id, msg = self.trader.open_trade(setup)
            summary["trades_opened"] += 1
            logger.info("Trade opened: #%d %s %s", trade_id, pair, decision.direction)
            await self.notify(f"NEW TRADE\n{msg}")

        except ValueError as e:
            logger.warning("Trade rejected for %s: %s", pair, e)
            await self.notify(f"Trade rejected: {pair}\n{e}")
        except Exception as e:
            logger.error("Failed to open trade for %s: %s", pair, e, exc_info=True)
            summary["errors"].append(f"Trade open error {pair}: {e}")

    async def _manage_positions(
        self, current_prices: dict, all_klines: dict, summary: dict,
    ) -> None:
        """Run position manager on all open trades."""
        # Build klines dict for structure break detection
        klines_map = {}
        for pair, klines in all_klines.items():
            if "1h" in klines:
                klines_map[pair] = {"1h": klines["1h"]}

        updates = self.pm.manage_all_positions(current_prices, klines_map or None)

        for update in updates:
            summary["position_updates"].append({
                "trade_id": update.trade_id,
                "pair": update.pair,
                "action": update.action,
                "state": f"{update.old_state}->{update.new_state}",
                "stop": f"{update.old_stop:.2f}->{update.new_stop:.2f}",
                "r": f"{update.unrealised_r:.2f}R",
            })

            # Execute actions
            if update.action == "CLOSE_STOP":
                result = self.trader.close_trade(
                    update.trade_id, update.current_price, "STOP_HIT",
                )
                await self.notify(
                    f"STOP HIT\n{update.pair} closed @ {update.current_price:.2f}\n"
                    f"PnL: {result['pnl_r']:.2f}R ({result['pnl_pct']:.2f}%)"
                )

            elif update.action == "CLOSE_HALF":
                self.trader.close_half(update.trade_id, update.current_price)
                db.update_trade(
                    update.trade_id,
                    stop_state=update.new_state,
                    stop_current=update.new_stop,
                )
                await self.notify(
                    f"TP1 HIT — 50% closed\n{update.pair} @ {update.current_price:.2f}\n"
                    f"Stop raised to {update.new_stop:.2f} ({update.new_state})"
                )

            elif update.action == "CLOSE_STRUCT":
                result = self.trader.close_trade(
                    update.trade_id, update.current_price, "STRUCTURE_BREAK",
                )
                await self.notify(
                    f"STRUCTURE BREAK\n{update.pair} closed @ {update.current_price:.2f}\n"
                    f"PnL: {result['pnl_r']:.2f}R ({result['pnl_pct']:.2f}%)"
                )

            elif update.action == "CLOSE_TIME":
                result = self.trader.close_trade(
                    update.trade_id, update.current_price, "TIME_EXIT",
                )
                await self.notify(
                    f"TIME EXIT\n{update.pair} closed @ {update.current_price:.2f}\n"
                    f"PnL: {result['pnl_r']:.2f}R ({result['pnl_pct']:.2f}%)"
                )

            elif update.action == "RAISE_STOP":
                await self.notify(
                    f"STOP RAISED\n{update.pair}: {update.old_state} -> {update.new_state}\n"
                    f"Stop: {update.old_stop:.2f} -> {update.new_stop:.2f}\n"
                    f"Current R: {update.unrealised_r:.2f}"
                )

    async def _emergency_halt(
        self, current_prices: dict, summary: dict,
    ) -> None:
        """Daily drawdown halt — close all trades and halt system."""
        logger.warning("EMERGENCY HALT — closing all positions")
        results = self.trader.close_all(current_prices, "DAILY_HALT")

        msg_lines = ["DAILY HALT TRIGGERED\nAll positions closed:"]
        for r in results:
            msg_lines.append(
                f"  {r['pair']}: {r['pnl_r']:.2f}R ({r['pnl_pct']:.2f}%)"
            )
        msg_lines.append("\nSystem halted. Use /resume to restart.")

        await self.notify("\n".join(msg_lines))
        summary["errors"].append("DAILY HALT TRIGGERED")

    async def run(self, max_runtime: int = 0) -> None:
        """
        Main entry point — runs the 5-minute loop.
        max_runtime: seconds before auto-stop (0 = run forever).
        """
        logger.info("ATLAS Lean v2.0 starting...")
        logger.info("Pairs: %s", config.PAIRS)
        logger.info("Scan interval: %ds", config.SCAN_INTERVAL_SECONDS)
        logger.info("Max trades: %d | Max risk: %.1f%% | Min R:R: %.1f",
                     config.MAX_SIMULTANEOUS_TRADES, config.MAX_RISK_PCT, config.MIN_RR_RATIO)
        if max_runtime:
            logger.info("Auto-stop in %ds", max_runtime)

        start_time = time.monotonic()
        run_label = "LIVE (Claude-powered entries)" if self.live_mode else "DRY-RUN (analysis only)"
        await self.notify(
            f"ATLAS Lean v2.0 started — {run_label}\n"
            f"Monitoring: {', '.join(config.PAIRS)}\n"
            f"Every 5 min you'll receive:\n"
            f"  • Cycle summary (all pairs)\n"
            f"  • Full analysis for pairs hitting threshold\n"
            + (f"Auto-stop in {max_runtime // 60}min." if max_runtime else "Running indefinitely.")
        )

        while True:
            cycle_start = time.monotonic()
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error("Unhandled cycle error: %s", e, exc_info=True)
                await self.notify(f"CYCLE ERROR: {e}")

            # Auto-stop after max_runtime
            if max_runtime and (time.monotonic() - start_time) >= max_runtime:
                logger.info("Max runtime reached — stopping")
                await self.notify(
                    f"Auto-stop reached.\n"
                    f"Ran {self._cycle_count} cycles.\n"
                    f"ATLAS stopped."
                )
                break

            # Sleep for remainder of interval, waking early if /scan was requested
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, config.SCAN_INTERVAL_SECONDS - elapsed)
            slept = 0.0
            while slept < sleep_time:
                if self._scan_requested:
                    self._scan_requested = False
                    logger.info("Immediate scan requested via Telegram")
                    break
                await asyncio.sleep(1)
                slept += 1

    async def close(self) -> None:
        """Cleanup resources."""
        await self.scanner.close()
        logger.info("ATLAS Lean shutdown complete")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    atlas = AtlasLean()

    async def run_all():
        try:
            from telegram_bot import TelegramBot
            bot = TelegramBot(atlas)
            atlas.set_telegram(bot)
            await bot.start()

            await atlas.run(max_runtime=3600)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await atlas.close()

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
