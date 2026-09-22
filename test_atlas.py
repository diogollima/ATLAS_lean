"""
ATLAS Lean — regression tests.

Every test here corresponds to a bug that was shipped at some point. They exist
so those bugs cannot come back silently.

Run any of:
    python test_atlas.py
    python -m unittest test_atlas
    pytest test_atlas.py

No test framework beyond the stdlib is required. Tests that need the live
indicator engine (pandas-ta) skip automatically when it is not installed, so
the pure-logic tests always run.
"""

import ast
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import config  # noqa: E402

try:
    import pandas_ta  # noqa: F401
    HAS_PANDAS_TA = True
except Exception:
    HAS_PANDAS_TA = False

needs_pandas_ta = unittest.skipUnless(
    HAS_PANDAS_TA, "pandas-ta not installed (pip install -r requirements.txt)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_tree(filename):
    """Parse a project file into an AST."""
    with open(os.path.join(PROJECT_DIR, filename), "r", encoding="utf-8") as f:
        return ast.parse(f.read())


def _ohlcv(n, freq_hours, seed=7, drift=0.0004, start=60000.0):
    """Synthetic OHLCV in exactly the shape scanner.fetch_klines() returns."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, 0.006, n)))
    vol = rng.lognormal(6, 0.4, n)
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{freq_hours}h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * (1 + np.abs(rng.normal(0, 0.003, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.003, n))),
            "close": close,
            "volume": vol,
            "close_time": idx,
            "quote_volume": vol * close,
            "trades": rng.integers(100, 900, n),
            "taker_buy_volume": vol * rng.uniform(0.35, 0.68, n),
            "taker_buy_quote_volume": vol * close * 0.5,
        },
        index=idx,
    )


def _frame_with(n=60, **columns):
    """Minimal frame carrying pre-computed indicator columns."""
    base = {
        "open": np.linspace(100, 110, n), "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n), "close": np.linspace(100, 110, n),
        "volume": np.full(n, 1000.0),
    }
    base.update({k: np.full(n, v) for k, v in columns.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# The ADX/CHOP gates used to raise instead of returning
# ---------------------------------------------------------------------------

class TestRegimeGatesDoNotRaise(unittest.TestCase):
    """
    Both gates built their rejection message with an invalid f-string format
    spec — `f"{adx:.1f if adx else 'N/A'}"`. That raised ValueError (or
    TypeError when the value was None) on the *common* path: a failing gate.
    main.py swallowed it as a per-pair exception, so affected pairs vanished
    from the cycle with no error surfaced.
    """

    def _trending(self, adx, chop):
        from regime_detector import _check_trending
        df = _frame_with(ema21=96.0, ema50=95.0, ema200=90.0,
                         adx14=adx, chop14=chop, atr14=1.0,
                         atr14_sma20=0.9, vol_ratio=1.2)
        return _check_trending(df, df, {})

    def _pullback(self, adx, chop):
        from regime_detector import _check_pullback
        df = _frame_with(ema21=96.0, ema50=95.0, ema200=90.0,
                         adx14=adx, chop14=chop, rsi14=40.0, bb_pctb=0.3,
                         atr14=1.0, atr14_sma20=0.9, vol_ratio=1.2)
        return _check_pullback(df, df, {}, True)

    def test_trending_gate_failure_returns_cleanly(self):
        for adx, chop in [(18.0, 55.0), (30.0, 50.0), (10.0, 90.0)]:
            with self.subTest(adx=adx, chop=chop):
                passed, conf, details = self._trending(adx, chop)
                self.assertFalse(passed)
                self.assertIn("reason", details)
                self.assertIn("ADX", details["reason"])

    def test_trending_gate_handles_missing_indicators(self):
        """NaN ADX/CHOP (warmup period) must not raise."""
        passed, conf, details = self._trending(np.nan, np.nan)
        self.assertFalse(passed)
        self.assertIn("N/A", details["reason"])

    def test_pullback_gate_failure_returns_cleanly(self):
        passed, conf, details = self._pullback(40.0, 20.0)
        self.assertFalse(passed)
        self.assertIn("ADX", details["reason"])

    def test_adx_of_exactly_zero_is_not_dropped(self):
        """`if adx_4h` treated a legitimate 0.0 reading as missing."""
        _, _, details = self._trending(0.0, 55.0)
        self.assertEqual(details["adx14_4h"], 0.0)

    def test_detect_regime_survives_a_failing_gate(self):
        """End to end: the entry point main.py calls must not raise."""
        from regime_detector import detect_regime
        df = _frame_with(ema21=96.0, ema50=95.0, ema200=90.0,
                         adx14=18.0, chop14=55.0, rsi14=40.0, bb_pctb=0.3,
                         atr14=1.0, atr14_sma20=0.9, vol_ratio=1.2,
                         bb_width=2.0, bb_width_sma20=2.0,
                         compression_ratio=1.0)
        result = detect_regime({"1h": df, "4h": df}, {"volume_ratio_1h": 1.0})
        self.assertIn(result.regime,
                      {"TRENDING", "PULLBACK", "BREAKOUT", "RANGING"})


# ---------------------------------------------------------------------------
# Order-flow signals must not reward flow pushing against the trade
# ---------------------------------------------------------------------------

class TestDirectionalOrderFlow(unittest.TestCase):
    """
    Signals 7 and 8 scored aggression in EITHER direction, so on a long-only
    system a market being actively dumped earned +2/9.
    """

    SOLD_OFF = {"taker_buy_ratio_1h": 0.30, "ob_bid_pct": 0.25}
    BID_UP = {"taker_buy_ratio_1h": 0.62, "ob_bid_pct": 0.65}
    NEUTRAL = {"taker_buy_ratio_1h": 0.50, "ob_bid_pct": 0.50}

    def _flow_score(self, metrics, direction):
        from signals import _sig_taker_buy_bias, _sig_orderbook_imbalance
        taker, _ = _sig_taker_buy_bias(metrics, "TRENDING", direction)
        book, _ = _sig_orderbook_imbalance(metrics, "TRENDING", direction)
        return int(taker) + int(book)

    def test_long_scores_nothing_on_sell_side_flow(self):
        self.assertEqual(self._flow_score(self.SOLD_OFF, "LONG"), 0)

    def test_long_still_scores_genuine_buy_side_flow(self):
        self.assertEqual(self._flow_score(self.BID_UP, "LONG"), 2)

    def test_short_scores_sell_side_flow(self):
        """The SHORT branch is retained for manual /enter."""
        self.assertEqual(self._flow_score(self.SOLD_OFF, "SHORT"), 2)

    def test_neutral_flow_scores_nothing_either_way(self):
        for direction in ("LONG", "SHORT"):
            with self.subTest(direction=direction):
                self.assertEqual(self._flow_score(self.NEUTRAL, direction), 0)

    def test_absent_order_book_scores_zero(self):
        """
        The backtest has no historical depth, so compute_volume_metrics
        returns ob_bid_pct = 0.5. That must score 0, not 1 — the -1 threshold
        adjustment in backtest.py depends on it.
        """
        from signals import _sig_orderbook_imbalance
        passed, _ = _sig_orderbook_imbalance({"ob_bid_pct": 0.5}, "TRENDING", "LONG")
        self.assertFalse(passed)

    def test_score_pair_defaults_to_long(self):
        import inspect
        from signals import score_pair
        default = inspect.signature(score_pair).parameters["direction"].default
        self.assertEqual(default, "LONG")


# ---------------------------------------------------------------------------
# Entry filters — must behave identically for live and backtest
# ---------------------------------------------------------------------------

class TestDailyTrendFilter(unittest.TestCase):

    def setUp(self):
        self._original = config.DAILY_TREND_FILTER
        config.DAILY_TREND_FILTER = True

    def tearDown(self):
        config.DAILY_TREND_FILTER = self._original

    def _daily(self, close, ema21, ema50):
        n = 30
        return pd.DataFrame({
            "close": np.full(n, close),
            "ema21": np.full(n, ema21),
            "ema50": np.full(n, ema50),
        })

    def test_uptrend_allows_entries(self):
        from signals import passes_daily_trend_filter
        ok, _ = passes_daily_trend_filter({"1d": self._daily(110, 100, 90)})
        self.assertTrue(ok)

    def test_close_below_ema21_blocks(self):
        from signals import passes_daily_trend_filter
        ok, reason = passes_daily_trend_filter({"1d": self._daily(95, 100, 90)})
        self.assertFalse(ok)
        self.assertIn("EMA21", reason)

    def test_ema21_below_ema50_blocks(self):
        from signals import passes_daily_trend_filter
        ok, reason = passes_daily_trend_filter({"1d": self._daily(110, 100, 105)})
        self.assertFalse(ok)
        self.assertIn("EMA50", reason)

    def test_missing_daily_data_fails_open(self):
        """A data gap must never silently block every entry."""
        from signals import passes_daily_trend_filter
        self.assertTrue(passes_daily_trend_filter({})[0])
        self.assertTrue(passes_daily_trend_filter({"1d": self._daily(1, 1, 1).head(3)})[0])

    def test_disabled_flag_is_honoured(self):
        from signals import passes_daily_trend_filter
        config.DAILY_TREND_FILTER = False
        ok, reason = passes_daily_trend_filter({"1d": self._daily(95, 100, 90)})
        self.assertTrue(ok)
        self.assertIn("disabled", reason)


class TestReentryCooldown(unittest.TestCase):
    """db.hours_since_last_loss() backs the live half of the cooldown."""

    def setUp(self):
        import db
        self._original_path = config.DB_PATH
        fd, self._path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self._path)
        config.DB_PATH = self._path
        db._conn = None
        db.init_db()
        self.db = db

    def tearDown(self):
        if self.db._conn:
            self.db._conn.close()
            self.db._conn = None
        config.DB_PATH = self._original_path
        for suffix in ("", "-wal", "-shm"):
            path = self._path + suffix
            if os.path.exists(path):
                os.unlink(path)

    def _closed_trade(self, pair, hours_ago, pnl_r):
        trade_id = self.db.insert_trade(
            pair=pair, direction="LONG", source="SCANNER",
            regime_at_entry="PULLBACK", entry_price=100.0, stop_initial=97.0,
            tp1=104.0, tp2=109.0, position_size_pct=1.0, atr_at_entry=1.0,
        )
        closed_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        self.db.update_trade(trade_id, status="CLOSED", pnl_r=pnl_r,
                             pnl_pct=pnl_r, closed_at=closed_at.isoformat())

    def test_no_history_means_no_cooldown(self):
        self.assertIsNone(self.db.hours_since_last_loss("BTCUSDT"))

    def test_recent_loss_is_within_cooldown(self):
        self._closed_trade("BTCUSDT", hours_ago=3, pnl_r=-1.0)
        elapsed = self.db.hours_since_last_loss("BTCUSDT")
        self.assertLess(elapsed, config.REENTRY_COOLDOWN_HOURS)

    def test_old_loss_is_outside_cooldown(self):
        self._closed_trade("ETHUSDT", hours_ago=30, pnl_r=-1.0)
        elapsed = self.db.hours_since_last_loss("ETHUSDT")
        self.assertGreater(elapsed, config.REENTRY_COOLDOWN_HOURS)

    def test_a_recent_win_does_not_trigger_cooldown(self):
        self._closed_trade("SOLUSDT", hours_ago=1, pnl_r=+1.5)
        self.assertIsNone(self.db.hours_since_last_loss("SOLUSDT"))

    def test_cooldown_is_per_pair(self):
        self._closed_trade("BTCUSDT", hours_ago=1, pnl_r=-1.0)
        self.assertIsNotNone(self.db.hours_since_last_loss("BTCUSDT"))
        self.assertIsNone(self.db.hours_since_last_loss("ETHUSDT"))


# ---------------------------------------------------------------------------
# Cross-module contracts that broke silently
# ---------------------------------------------------------------------------

class TestTickerPriceKeyContract(unittest.TestCase):
    """
    scanner.fetch_ticker_24h() returns "last_price"; telegram_bot looked up
    "lastPrice". The branch never ran, so /close and /closehalf fell back to
    the entry price and every manual close booked exactly 0.00 PnL.
    """

    def test_scanner_emits_snake_case_last_price(self):
        tree = _source_tree("scanner.py")
        keys = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("last_price", keys)

    def test_telegram_bot_reads_the_key_the_scanner_emits(self):
        with open(os.path.join(PROJECT_DIR, "telegram_bot.py"), encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn('"lastPrice"', source)
        self.assertIn('"last_price"', source)


class TestClaudeClientIsAsync(unittest.TestCase):
    """
    A synchronous anthropic.Anthropic client was called un-awaited from two
    coroutines, blocking the shared event loop — freezing the Telegram bot and
    stalling the scan cycle for the duration of every API call.
    """

    def _message_create_calls(self):
        tree = _source_tree("claude_client.py")

        def is_messages_create(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "messages"
            )

        total = sum(1 for n in ast.walk(tree) if is_messages_create(n))
        awaited = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Await) and is_messages_create(n.value)
        )
        return awaited, total

    def test_uses_the_async_client(self):
        tree = _source_tree("claude_client.py")
        names = {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn("AsyncAnthropic", names)
        self.assertNotIn("Anthropic", names - {"AsyncAnthropic"})

    def test_every_api_call_is_awaited(self):
        awaited, total = self._message_create_calls()
        self.assertGreaterEqual(total, 2, "expected entry + weekly-review calls")
        self.assertEqual(awaited, total, f"{total - awaited} un-awaited API call(s)")


class TestConfiguration(unittest.TestCase):

    def test_model_ids_carry_no_date_suffix(self):
        """SONNET_MODEL was "claude-sonnet-4-6-20250514", which does not exist."""
        for name in ("HAIKU_MODEL", "SONNET_MODEL"):
            model_id = getattr(config, name)
            with self.subTest(model=name):
                tail = model_id.rsplit("-", 1)[-1]
                self.assertFalse(
                    tail.isdigit() and len(tail) == 8,
                    f"{name}={model_id!r} has a date suffix",
                )

    def test_numpy_is_pinned_below_2(self):
        """pandas-ta 0.3.14b imports numpy.NaN, removed in numpy 2.0."""
        with open(os.path.join(PROJECT_DIR, "requirements.txt"), encoding="utf-8") as f:
            requirements = f.read()
        numpy_lines = [
            line for line in requirements.splitlines()
            if line.strip().startswith("numpy")
        ]
        self.assertTrue(numpy_lines, "numpy not pinned in requirements.txt")
        self.assertIn("<2", numpy_lines[0])

    def test_risk_limits_are_sane(self):
        self.assertGreater(config.MAX_RISK_PCT, 0)
        self.assertLessEqual(config.MAX_RISK_PCT, 5.0)
        self.assertGreaterEqual(config.MAX_SIMULTANEOUS_TRADES, 1)
        self.assertGreater(config.DAILY_DRAWDOWN_HALT_PCT, config.MAX_RISK_PCT)
        self.assertGreaterEqual(config.MIN_RR_RATIO, 1.0)

    def test_ranging_regime_never_triggers_entries(self):
        self.assertGreater(config.get_score_threshold("RANGING"), 9)


# ---------------------------------------------------------------------------
# Backtest and live must not drift apart again
# ---------------------------------------------------------------------------

class TestBacktestSharesTheLiveEngine(unittest.TestCase):
    """
    backtest.py used to carry its own indicator, regime and scoring code, with
    different EMA periods, RSI bands and structure rules. Backtest numbers
    therefore described a strategy that was never running.
    """

    def setUp(self):
        self.tree = _source_tree("backtest.py")

    def _imported_names(self):
        names = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
        return names

    def _defined_functions(self):
        return {
            node.name for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_imports_the_live_modules(self):
        imported = self._imported_names()
        for required in (
            "indicators.compute_indicators",
            "indicators.compute_volume_metrics",
            "regime_detector.detect_regime",
            "signals.score_pair",
            "signals.passes_daily_trend_filter",
        ):
            with self.subTest(symbol=required):
                self.assertIn(required, imported)

    def test_defines_no_parallel_implementation(self):
        defined = self._defined_functions()
        for forbidden in ("add_indicators", "score_bar", "detect_regime_bar",
                          "adx", "choppiness", "macd_hist", "bollinger"):
            with self.subTest(function=forbidden):
                self.assertNotIn(
                    forbidden, defined,
                    f"backtest.py redefines {forbidden}() — it must use the live engine",
                )

    def test_entry_filters_come_from_config(self):
        with open(os.path.join(PROJECT_DIR, "backtest.py"), encoding="utf-8") as f:
            source = f.read()
        self.assertIn("config.REENTRY_COOLDOWN_HOURS", source)
        self.assertIn("passes_daily_trend_filter", source)

    def test_argparse_is_not_run_at_import_time(self):
        """
        parse_args() at module scope consumed the argv of any process that
        imported this module — including a test runner.
        """
        module_level_calls = [
            node for node in self.tree.body
            if isinstance(node, ast.Expr) or isinstance(node, ast.Assign)
        ]
        for node in module_level_calls:
            for child in ast.walk(node):
                if (isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "parse_args"):
                    self.fail("parse_args() runs at import time")
        self.assertIn("_parse_args", self._defined_functions())


@needs_pandas_ta
class TestSharedPipelineEndToEnd(unittest.TestCase):
    """
    Walk synthetic data through the exact sequence backtest.py runs, proving
    the column contract between compute_indicators and the live scorers holds.
    """

    @classmethod
    def setUpClass(cls):
        from indicators import compute_indicators
        cls.ind = compute_indicators({
            "1h": _ohlcv(600, 1, drift=0.0004),
            "4h": _ohlcv(300, 4, drift=0.0016),
            "1d": _ohlcv(120, 24, drift=0.0090),
        })

    def test_compute_indicators_emits_every_column_the_scorers_read(self):
        required = {
            "ema21", "ema50", "ema200", "rsi14", "macd_hist", "atr14",
            "atr14_sma20", "bb_pctb", "bb_width", "bb_width_sma20",
            "vol_ratio", "vol_trend", "adx14", "chop14",
            "compression_ratio", "taker_buy_ratio",
        }
        self.assertEqual(required - set(self.ind["1h"].columns), set())

    def test_walk_forward_produces_valid_scores(self):
        from indicators import compute_volume_metrics
        from regime_detector import detect_regime
        from signals import score_pair, passes_daily_trend_filter

        seen_regimes, scored = set(), 0
        for ts in self.ind["1h"].index[210::7]:
            four_hour = self.ind["4h"][self.ind["4h"].index <= ts].tail(50)
            if len(four_hour) < 5:
                continue
            klines = {
                "1h": self.ind["1h"].loc[:ts],
                "4h": four_hour,
                "1d": self.ind["1d"][self.ind["1d"].index <= ts].tail(60),
            }
            metrics = compute_volume_metrics(
                klines["1h"], klines["4h"], None, None, None, None)

            # Absent depth must stay neutral, otherwise the backtest's -1
            # threshold adjustment is wrong.
            self.assertEqual(metrics["ob_bid_pct"], 0.5)

            passes_daily_trend_filter(klines)   # must not raise
            regime = detect_regime(klines, metrics).regime
            seen_regimes.add(regime)
            if regime == "RANGING":
                continue

            result = score_pair("BTCUSDT", klines, metrics, regime, direction="LONG")
            scored += 1
            self.assertEqual(len(result.signals), 9)
            self.assertTrue(0 <= result.score <= 9)
            self.assertFalse(result.signals["order_book"],
                             "signal 8 cannot score without depth data")

        self.assertTrue(seen_regimes, "walk-forward produced no regimes at all")
        self.assertGreater(scored, 0, "no bar was ever scored")


if __name__ == "__main__":
    print(f"pandas-ta {'available' if HAS_PANDAS_TA else 'MISSING — integration tests skipped'}")
    unittest.main(verbosity=2)
