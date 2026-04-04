"""
ATLAS Lean v2.0 — Claude Integration
Calls Claude Haiku 4.5 only when Python has confirmed a candidate setup.
Uses prompt caching on system prompt + strategy rules (~70% input savings).
Returns structured JSON decisions.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic

import config

logger = logging.getLogger(__name__)


@dataclass
class ClaudeDecision:
    action: str             # "ENTER" or "SKIP"
    direction: str = "LONG"
    confidence: str = "medium"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    position_size_pct: float = 1.5
    r_ratio: float = 0.0
    reasoning: str = ""
    key_risk: str = ""
    flags: list[str] = field(default_factory=list)
    raw_response: str = ""


class ClaudeClient:
    """Manages Claude API calls with prompt caching for cost optimization."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._strategy_text = self._load_strategy()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read = 0
        self._total_cache_create = 0
        self._call_count = 0

    def _load_strategy(self) -> str:
        """Load the active strategy file."""
        path = os.path.join(config.STRATEGY_DIR, config.ACTIVE_STRATEGY_FILE)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Strategy file not found: %s", path)
            return "No strategy file loaded."

    def reload_strategy(self, strategy_text: Optional[str] = None) -> None:
        """Reload strategy from file or from provided text."""
        if strategy_text:
            self._strategy_text = strategy_text
        else:
            self._strategy_text = self._load_strategy()
        logger.info("Strategy reloaded (%d chars)", len(self._strategy_text))

    def _build_system_prompt(self) -> list[dict]:
        """
        Build system prompt with cache control.
        Block 1: Role + risk rules (static)
        Block 2: Strategy rules (cached with ephemeral TTL)
        """
        role_block = {
            "type": "text",
            "text": (
                "You are ATLAS, an autonomous crypto trading analyst. "
                "You receive pre-scored market data for Binance pairs and decide whether to ENTER or SKIP.\n\n"
                "RULES (NON-NEGOTIABLE):\n"
                f"- Max risk per trade: {config.MAX_RISK_PCT}% of account\n"
                f"- Max simultaneous trades: {config.MAX_SIMULTANEOUS_TRADES}\n"
                f"- Minimum R:R ratio: {config.MIN_RR_RATIO}\n"
                f"- Daily drawdown halt: {config.DAILY_DRAWDOWN_HALT_PCT}%\n"
                "- Stop loss must be at a structural level (no arbitrary pips)\n"
                "- Stop can only move toward profit, never backwards\n\n"
                "RESPONSE FORMAT: Return ONLY valid JSON with these fields:\n"
                '{"decision":"ENTER"|"SKIP","confidence":"high"|"medium"|"low",'
                '"direction":"LONG"|"SHORT","entry":<price>,"stop_loss":<price>,'
                '"take_profit_1":<price>,"take_profit_2":<price>,'
                '"position_size_pct":<float>,"r_ratio":<float>,'
                '"reasoning":"<1-3 sentences>","key_risk":"<1 sentence>",'
                '"flags":[]}\n\n'
                "For SKIP decisions, include: decision, confidence, reasoning, retry_condition, flags."
            ),
        }

        strategy_block = {
            "type": "text",
            "text": f"ACTIVE STRATEGY RULES:\n\n{self._strategy_text}",
            "cache_control": {"type": "ephemeral"},
        }

        return [role_block, strategy_block]

    def _build_user_message(
        self,
        pair: str,
        regime: str,
        score: int,
        signals: dict[str, bool],
        indicators: dict,
        open_trades_count: int,
        account_balance: float,
    ) -> str:
        """Build compact JSON payload for Claude."""
        # Extract key indicators for the payload (minimize tokens)
        payload = {
            "pair": pair,
            "regime": regime,
            "score": f"{score}/9",
            "signals": {k: int(v) for k, v in signals.items()},
            "price": indicators.get("price"),
            "trend": {
                "ema50_4h": indicators.get("ema50_4h"),
                "ema200_4h": indicators.get("ema200_4h"),
                "ema_stack_1h": indicators.get("ema_stack_1h"),
                "structure_4h": indicators.get("structure_4h"),
                "above_ema50_4h": indicators.get("price_above_ema50_4h"),
                "above_200d_ma": indicators.get("above_200d_ma"),
            },
            "momentum": {
                "rsi_15m": indicators.get("rsi_15m"),
                "rsi_1h": indicators.get("rsi_1h"),
                "rsi_4h": indicators.get("rsi_4h"),
                "macd_direction_1h": indicators.get("macd_direction_1h"),
                "macd_hist_4h": indicators.get("macd_hist_4h"),
                "stoch_k_4h": indicators.get("stoch_k_4h"),
                "roc10_1h": indicators.get("roc10_1h"),
            },
            "volume": {
                "ratio_1h": indicators.get("volume_ratio_1h"),
                "taker_buy_1h": indicators.get("taker_buy_ratio_1h"),
                "taker_buy_agg": indicators.get("taker_buy_ratio_agg"),
                "ob_bid_pct": indicators.get("ob_bid_pct"),
                "trend": indicators.get("volume_trend_1h"),
                "large_trades": indicators.get("large_trades_count"),
                "change_24h": indicators.get("price_change_24h_pct"),
            },
            "volatility": {
                "atr_1h": indicators.get("atr14_1h"),
                "atr_pct_1h": indicators.get("atr_pct_1h"),
                "bb_pctb_1h": indicators.get("bb_pctb_1h"),
                "compression_ratio": indicators.get("compression_ratio_1h"),
            },
            "account": {
                "balance": account_balance,
                "open_trades": open_trades_count,
                "max_risk_pct": config.MAX_RISK_PCT,
            },
        }

        # Add interpretations if available
        interps = indicators.get("interpretations")
        if interps:
            payload["context"] = interps

        return json.dumps(payload, default=str)

    async def evaluate_entry(
        self,
        pair: str,
        regime: str,
        score: int,
        signals: dict[str, bool],
        indicators: dict,
        open_trades_count: int,
        account_balance: float = None,
    ) -> ClaudeDecision:
        """
        Call Claude Haiku 4.5 to evaluate a trade entry.
        Returns ClaudeDecision.
        """
        if account_balance is None:
            account_balance = config.PAPER_BALANCE

        user_msg = self._build_user_message(
            pair, regime, score, signals, indicators,
            open_trades_count, account_balance,
        )

        try:
            response = self._client.messages.create(
                model=config.HAIKU_MODEL,
                max_tokens=500,
                system=self._build_system_prompt(),
                messages=[{"role": "user", "content": user_msg}],
            )

            # Track token usage
            usage = response.usage
            self._call_count += 1
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens
            if hasattr(usage, "cache_read_input_tokens"):
                self._total_cache_read += usage.cache_read_input_tokens or 0
            if hasattr(usage, "cache_creation_input_tokens"):
                self._total_cache_create += usage.cache_creation_input_tokens or 0

            logger.info(
                "Claude call #%d: in=%d out=%d cache_read=%d cache_create=%d",
                self._call_count, usage.input_tokens, usage.output_tokens,
                getattr(usage, "cache_read_input_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
            )

            # Parse response
            raw_text = response.content[0].text.strip()
            return self._parse_decision(raw_text, pair)

        except anthropic.APIError as e:
            logger.error("Claude API error: %s", e)
            return ClaudeDecision(
                action="SKIP",
                reasoning=f"API error: {e}",
                raw_response=str(e),
            )
        except Exception as e:
            logger.error("Claude call failed: %s", e)
            return ClaudeDecision(
                action="SKIP",
                reasoning=f"Error: {e}",
                raw_response=str(e),
            )

    def _parse_decision(self, raw_text: str, pair: str) -> ClaudeDecision:
        """Parse Claude's JSON response into a ClaudeDecision."""
        try:
            # Try to extract JSON from the response
            text = raw_text
            if "```" in text:
                # Extract from code block
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            elif not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

            data = json.loads(text)

            decision = ClaudeDecision(
                action=data.get("decision", "SKIP").upper(),
                direction=data.get("direction", "LONG").upper(),
                confidence=data.get("confidence", "medium"),
                entry_price=float(data.get("entry", 0)),
                stop_loss=float(data.get("stop_loss", 0)),
                tp1=float(data.get("take_profit_1", 0)),
                tp2=float(data.get("take_profit_2", 0)),
                position_size_pct=float(data.get("position_size_pct", 1.5)),
                r_ratio=float(data.get("r_ratio", 0)),
                reasoning=data.get("reasoning", ""),
                key_risk=data.get("key_risk", ""),
                flags=data.get("flags", []),
                raw_response=raw_text,
            )

            # Validate R:R ratio for ENTER decisions
            if decision.action == "ENTER" and decision.r_ratio < config.MIN_RR_RATIO:
                logger.warning(
                    "%s: Claude suggested R:R %.1f < min %.1f — overriding to SKIP",
                    pair, decision.r_ratio, config.MIN_RR_RATIO,
                )
                decision.action = "SKIP"
                decision.reasoning += f" [OVERRIDDEN: R:R {decision.r_ratio} < {config.MIN_RR_RATIO}]"

            # Validate position size
            if decision.position_size_pct > config.MAX_RISK_PCT:
                decision.position_size_pct = config.MAX_RISK_PCT

            return decision

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("Failed to parse Claude response: %s\nRaw: %s", e, raw_text)
            return ClaudeDecision(
                action="SKIP",
                reasoning=f"Parse error: {e}",
                raw_response=raw_text,
            )

    async def weekly_review(
        self, performance_data: dict, current_strategy: str
    ) -> dict:
        """
        Call Claude Sonnet 4.6 for weekly strategy review.
        Returns dict with recommendation, proposed changes, and analysis.
        """
        prompt = (
            "You are ATLAS, reviewing this week's trading performance.\n\n"
            f"PERFORMANCE DATA:\n{json.dumps(performance_data, indent=2, default=str)}\n\n"
            f"CURRENT STRATEGY:\n{current_strategy}\n\n"
            "Analyze the performance and propose improvements. Return JSON with:\n"
            '{"recommendation":"ADOPT"|"EXTEND"|"KEEP",'
            '"performance_summary":"<2-3 sentences>",'
            '"signal_analysis":["<bullet1>","<bullet2>",...],'
            '"volume_findings":"<2-3 sentences>",'
            '"regime_effectiveness":"<2-3 sentences>",'
            '"proposed_changes":["<change1>","<change2>","<change3>"],'
            '"proposed_strategy":"<complete new strategy markdown>",'
            '"expected_impact":"<1 sentence>"}'
        )

        try:
            response = self._client.messages.create(
                model=config.SONNET_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            # Extract JSON
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
            return {"error": "No JSON in response", "raw": raw}

        except Exception as e:
            logger.error("Weekly review failed: %s", e)
            return {"error": str(e)}

    def get_usage_stats(self) -> dict:
        """Return cumulative token usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cache_read": self._total_cache_read,
            "total_cache_create": self._total_cache_create,
        }


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def test():
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY.startswith("sk-ant-your"):
            print("Set ANTHROPIC_API_KEY in .env to test Claude integration")
            return

        client = ClaudeClient()

        # Test with mock data
        mock_signals = {
            "trend_alignment": True, "ema_stack": True, "rsi_mode": True,
            "macd": False, "structure_level": True, "volume_ratio": False,
            "taker_buy_bias": True, "order_book": True, "regime_fit": True,
        }
        mock_indicators = {
            "pair": "BTCUSDT", "price": 69000, "rsi_1h": 44,
            "ema50_4h": 68500, "ema_stack_1h": "mixed",
            "macd_direction_1h": "bearish contracting",
            "volume_ratio_1h": 0.98, "taker_buy_ratio_1h": 0.58,
            "ob_bid_pct": 0.62, "atr14_1h": 430, "atr_pct_1h": 0.62,
            "bb_pctb_1h": 0.22, "interpretations": {},
        }

        print("Calling Claude Haiku 4.5...")
        decision = await client.evaluate_entry(
            "BTCUSDT", "PULLBACK", 6, mock_signals, mock_indicators, 1
        )

        print(f"\nDecision: {decision.action}")
        print(f"Direction: {decision.direction}")
        print(f"Confidence: {decision.confidence}")
        print(f"Entry: {decision.entry_price}")
        print(f"Stop: {decision.stop_loss}")
        print(f"TP1: {decision.tp1}")
        print(f"TP2: {decision.tp2}")
        print(f"R:R: {decision.r_ratio}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Key Risk: {decision.key_risk}")
        print(f"\nUsage: {client.get_usage_stats()}")

    asyncio.run(test())
