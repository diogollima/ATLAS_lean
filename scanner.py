"""
ATLAS Lean v2.0 — Binance Data Scanner
Fetches all market data from Binance REST API every 5-minute cycle.
All endpoints are free. Rate limit consumption: ~96 weight / 6,000 limit (1.6%).
"""

import asyncio
import logging
from typing import Optional

import httpx
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


class BinanceScanner:
    """Async Binance REST data fetcher for all watchlist pairs."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=config.BINANCE_BASE,
            timeout=15.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Individual endpoint fetchers
    # ------------------------------------------------------------------

    async def fetch_klines(
        self, pair: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV klines for a pair/timeframe. Returns DataFrame or None on error."""
        try:
            resp = await self._client.get(
                config.EP_KLINES,
                params={"symbol": pair, "interval": interval, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ])
            # Cast numeric columns
            for col in ["open", "high", "low", "close", "volume",
                        "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
                df[col] = df[col].astype(float)
            df["trades"] = df["trades"].astype(int)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            df.drop(columns=["ignore"], inplace=True)
            return df

        except Exception as e:
            logger.warning("fetch_klines(%s, %s) failed: %s", pair, interval, e)
            return None

    async def fetch_ticker_24h(self, pair: str) -> Optional[dict]:
        """Fetch 24h ticker stats for a pair."""
        try:
            resp = await self._client.get(
                config.EP_TICKER24H,
                params={"symbol": pair},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "price_change_pct": float(data["priceChangePercent"]),
                "weighted_avg_price": float(data["weightedAvgPrice"]),
                "volume_24h": float(data["volume"]),
                "quote_volume_24h": float(data["quoteVolume"]),
                "trade_count_24h": int(data["count"]),
                "last_price": float(data["lastPrice"]),
            }
        except Exception as e:
            logger.warning("fetch_ticker_24h(%s) failed: %s", pair, e)
            return None

    async def fetch_depth(self, pair: str, limit: int = 20) -> Optional[dict]:
        """Fetch order book depth (top N levels)."""
        try:
            resp = await self._client.get(
                config.EP_DEPTH,
                params={"symbol": pair, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            bids = [[float(p), float(q)] for p, q in data["bids"]]
            asks = [[float(p), float(q)] for p, q in data["asks"]]
            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.warning("fetch_depth(%s) failed: %s", pair, e)
            return None

    async def fetch_book_ticker(self, pair: str) -> Optional[dict]:
        """Fetch best bid/ask price and quantity."""
        try:
            resp = await self._client.get(
                config.EP_BOOK_TICKER,
                params={"symbol": pair},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "bid_price": float(data["bidPrice"]),
                "bid_qty": float(data["bidQty"]),
                "ask_price": float(data["askPrice"]),
                "ask_qty": float(data["askQty"]),
            }
        except Exception as e:
            logger.warning("fetch_book_ticker(%s) failed: %s", pair, e)
            return None

    async def fetch_agg_trades(self, pair: str, limit: int = 200) -> Optional[list[dict]]:
        """Fetch recent aggregate trades for buyer/seller flow analysis."""
        try:
            resp = await self._client.get(
                config.EP_AGG_TRADES,
                params={"symbol": pair, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "timestamp": t["T"],
                    "is_buyer_maker": t["m"],
                }
                for t in data
            ]
        except Exception as e:
            logger.warning("fetch_agg_trades(%s) failed: %s", pair, e)
            return None

    # ------------------------------------------------------------------
    # Pair-level orchestration
    # ------------------------------------------------------------------

    async def scan_pair(self, pair: str) -> Optional[dict]:
        """
        Fetch all data for a single pair concurrently.
        Returns dict with keys: klines, ticker24h, depth, book_ticker, agg_trades.
        Returns None if critical data (klines) fails completely.
        """
        # Fetch all kline timeframes concurrently
        kline_tasks = {
            tf: self.fetch_klines(pair, tf, config.KLINE_LIMITS[tf])
            for tf in config.TIMEFRAMES
        }

        # Fetch supplementary data concurrently alongside klines
        all_tasks = list(kline_tasks.values()) + [
            self.fetch_ticker_24h(pair),
            self.fetch_depth(pair),
            self.fetch_book_ticker(pair),
            self.fetch_agg_trades(pair),
        ]

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Unpack klines
        klines = {}
        tf_list = list(kline_tasks.keys())
        for i, tf in enumerate(tf_list):
            result = results[i]
            if isinstance(result, Exception):
                logger.warning("Kline %s/%s exception: %s", pair, tf, result)
                continue
            if result is not None:
                klines[tf] = result

        # Must have at least 1h and 4h for meaningful analysis
        if "1h" not in klines or "4h" not in klines:
            logger.warning("Skipping %s: missing critical timeframes (have: %s)",
                           pair, list(klines.keys()))
            return None

        # Unpack supplementary data
        offset = len(tf_list)
        ticker24h = results[offset] if not isinstance(results[offset], Exception) else None
        depth = results[offset + 1] if not isinstance(results[offset + 1], Exception) else None
        book_ticker = results[offset + 2] if not isinstance(results[offset + 2], Exception) else None
        agg_trades = results[offset + 3] if not isinstance(results[offset + 3], Exception) else None

        return {
            "klines": klines,
            "ticker24h": ticker24h,
            "depth": depth,
            "book_ticker": book_ticker,
            "agg_trades": agg_trades,
        }

    # ------------------------------------------------------------------
    # Full watchlist scan
    # ------------------------------------------------------------------

    async def scan_all(self) -> dict[str, dict]:
        """
        Scan all watchlist pairs concurrently.
        Returns: {"BTCUSDT": {...}, "ETHUSDT": {...}, ...}
        Pairs that fail completely are excluded from results.
        """
        tasks = [self.scan_pair(pair) for pair in config.PAIRS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        market_data = {}
        for pair, result in zip(config.PAIRS, results):
            if isinstance(result, Exception):
                logger.error("scan_pair(%s) exception: %s", pair, result)
                continue
            if result is not None:
                market_data[pair] = result

        logger.info("Scan complete: %d/%d pairs fetched", len(market_data), len(config.PAIRS))
        return market_data


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    async def test():
        scanner = BinanceScanner()
        try:
            print("Scanning all pairs...")
            start = time.time()
            data = await scanner.scan_all()
            elapsed = time.time() - start
            print(f"Scan completed in {elapsed:.2f}s\n")

            for pair, pair_data in data.items():
                klines = pair_data["klines"]
                tf_list = list(klines.keys())
                shapes = {tf: klines[tf].shape for tf in tf_list}
                price = klines["1h"]["close"].iloc[-1] if "1h" in klines else "N/A"
                print(f"{pair}: price={price}, timeframes={tf_list}")
                for tf, shape in shapes.items():
                    print(f"  {tf}: {shape[0]} candles x {shape[1]} columns")

                t24 = pair_data.get("ticker24h")
                if t24:
                    print(f"  24h change: {t24['price_change_pct']}%, "
                          f"volume: {t24['volume_24h']:.0f}")

                depth = pair_data.get("depth")
                if depth:
                    bid_vol = sum(q for _, q in depth["bids"])
                    ask_vol = sum(q for _, q in depth["asks"])
                    total = bid_vol + ask_vol
                    bid_pct = bid_vol / total * 100 if total > 0 else 0
                    print(f"  Order book: bids {bid_pct:.1f}% / asks {100-bid_pct:.1f}%")

                bt = pair_data.get("book_ticker")
                if bt:
                    spread = (bt["ask_price"] - bt["bid_price"]) / bt["bid_price"] * 100
                    print(f"  Spread: {spread:.4f}%")

                trades = pair_data.get("agg_trades")
                if trades:
                    buyer_vol = sum(t["qty"] for t in trades if not t["is_buyer_maker"])
                    total_vol = sum(t["qty"] for t in trades)
                    taker_ratio = buyer_vol / total_vol if total_vol > 0 else 0
                    print(f"  Taker buy ratio (last {len(trades)} trades): {taker_ratio:.3f}")

                print()

        finally:
            await scanner.close()

    asyncio.run(test())
