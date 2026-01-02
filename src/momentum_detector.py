"""
Momentum detection for crypto spot prices.

Detects when BTC/ETH makes a move on Binance/Coinbase before
Polymarket 15-minute windows have adjusted their odds.

This is the edge gabagool22 exploits: spot moves first, PM odds lag.
"""

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

import websockets
from rich.console import Console

from .config import get_config
from .models import MarketLag, MomentumSignal, TradeDirection

console = Console()


class PriceBuffer:
    """Rolling buffer of price data for momentum calculation."""

    def __init__(self, window_seconds: int = 15, max_samples: int = 100):
        self.window_seconds = window_seconds
        self.max_samples = max_samples
        self.prices: deque[tuple[datetime, Decimal]] = deque(maxlen=max_samples)

    def add(self, price: Decimal, timestamp: Optional[datetime] = None):
        """Add a price point."""
        ts = timestamp or datetime.now(timezone.utc)
        self.prices.append((ts, price))

    def get_momentum(self) -> Optional[tuple[Decimal, TradeDirection]]:
        """
        Calculate momentum over the window.
        Returns (percent_change, direction) or None if insufficient data.
        """
        if len(self.prices) < 2:
            return None

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self.window_seconds

        # Get prices within window
        window_prices = [
            (ts, price) for ts, price in self.prices
            if ts.timestamp() >= cutoff
        ]

        if len(window_prices) < 2:
            return None

        first_price = window_prices[0][1]
        last_price = window_prices[-1][1]

        if first_price == 0:
            return None

        change_pct = ((last_price - first_price) / first_price) * 100
        direction = TradeDirection.UP if change_pct > 0 else TradeDirection.DOWN

        return (abs(change_pct), direction)

    @property
    def current_price(self) -> Optional[Decimal]:
        """Get the most recent price."""
        return self.prices[-1][1] if self.prices else None


class MomentumDetector:
    """
    Detect momentum signals from crypto spot markets.

    Monitors Binance and Coinbase websocket feeds for BTC and ETH,
    detects significant moves, and emits signals when the move
    exceeds threshold.
    """

    # Minimum move % to trigger a signal
    DEFAULT_THRESHOLD = Decimal("0.15")  # 0.15% move

    def __init__(
        self,
        threshold: Decimal = DEFAULT_THRESHOLD,
        window_seconds: int = 15,
    ):
        self.config = get_config()
        self.threshold = threshold
        self.window_seconds = window_seconds

        # Price buffers for each asset/source
        self.buffers: dict[str, PriceBuffer] = {
            "binance_btc": PriceBuffer(window_seconds),
            "binance_eth": PriceBuffer(window_seconds),
            "coinbase_btc": PriceBuffer(window_seconds),
            "coinbase_eth": PriceBuffer(window_seconds),
        }

        self._running = False
        self._callbacks: list[Callable[[MomentumSignal], None]] = []

    def on_signal(self, callback: Callable[[MomentumSignal], None]):
        """Register a callback for momentum signals."""
        self._callbacks.append(callback)

    def _emit_signal(self, signal: MomentumSignal):
        """Emit a signal to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                console.print(f"[red]Callback error: {e}[/red]")

    def _calculate_strength(self, change_pct: Decimal) -> Decimal:
        """
        Calculate signal strength (0-1) based on price change magnitude.
        Larger moves = higher confidence signal.
        """
        # Scale: 0.15% = 0.5 strength, 0.5% = 1.0 strength
        normalized = min(change_pct / Decimal("0.5"), Decimal("1.0"))
        return normalized

    async def _connect_binance(self):
        """Connect to Binance websocket for BTC and ETH trades."""
        streams = ["btcusdt@trade", "ethusdt@trade"]
        url = f"{self.config.spot_feeds.binance_ws_url}/{'/'.join(streams)}"

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    console.print("[green]Connected to Binance[/green]")

                    async for message in ws:
                        if not self._running:
                            break

                        data = json.loads(message)
                        if "stream" not in data:
                            continue

                        stream = data["stream"]
                        trade = data["data"]
                        price = Decimal(trade["p"])
                        ts = datetime.fromtimestamp(trade["T"] / 1000, tz=timezone.utc)

                        if "btcusdt" in stream:
                            self.buffers["binance_btc"].add(price, ts)
                            self._check_momentum("binance_btc", "BTC", "binance")
                        elif "ethusdt" in stream:
                            self.buffers["binance_eth"].add(price, ts)
                            self._check_momentum("binance_eth", "ETH", "binance")

            except Exception as e:
                if self._running:
                    console.print(f"[yellow]Binance reconnecting: {e}[/yellow]")
                    await asyncio.sleep(5)

    async def _connect_coinbase(self):
        """Connect to Coinbase websocket for BTC and ETH trades."""
        url = self.config.spot_feeds.coinbase_ws_url

        subscribe_msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD", "ETH-USD"],
            "channels": ["ticker"],
        }

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    console.print("[green]Connected to Coinbase[/green]")

                    async for message in ws:
                        if not self._running:
                            break

                        data = json.loads(message)
                        if data.get("type") != "ticker":
                            continue

                        product = data.get("product_id", "")
                        price = Decimal(data.get("price", "0"))
                        ts = datetime.now(timezone.utc)

                        if product == "BTC-USD":
                            self.buffers["coinbase_btc"].add(price, ts)
                            self._check_momentum("coinbase_btc", "BTC", "coinbase")
                        elif product == "ETH-USD":
                            self.buffers["coinbase_eth"].add(price, ts)
                            self._check_momentum("coinbase_eth", "ETH", "coinbase")

            except Exception as e:
                if self._running:
                    console.print(f"[yellow]Coinbase reconnecting: {e}[/yellow]")
                    await asyncio.sleep(5)

    def _check_momentum(self, buffer_key: str, asset: str, source: str):
        """Check if momentum exceeds threshold and emit signal."""
        buffer = self.buffers[buffer_key]
        momentum = buffer.get_momentum()

        if momentum is None:
            return

        change_pct, direction = momentum

        if change_pct >= self.threshold:
            strength = self._calculate_strength(change_pct)
            signal = MomentumSignal(
                asset=asset,
                direction=direction,
                spot_price=buffer.current_price or Decimal("0"),
                price_change_percent=change_pct,
                timestamp=datetime.now(timezone.utc),
                source=source,
                strength=strength,
            )
            self._emit_signal(signal)

    async def start(self):
        """Start monitoring spot price feeds."""
        self._running = True
        console.print("[cyan]Starting momentum detector...[/cyan]")

        await asyncio.gather(
            self._connect_binance(),
            self._connect_coinbase(),
        )

    def stop(self):
        """Stop monitoring."""
        self._running = False
        console.print("[cyan]Momentum detector stopped[/cyan]")

    def get_current_prices(self) -> dict[str, Optional[Decimal]]:
        """Get current prices from all sources."""
        return {
            key: buffer.current_price
            for key, buffer in self.buffers.items()
        }


class LagDetector:
    """
    Detect lag between spot prices and Polymarket odds.

    When spot moves hard, PM odds should follow. This measures
    how long that takes and estimates the available edge.
    """

    def __init__(self, momentum_detector: MomentumDetector):
        self.momentum = momentum_detector
        self.pending_signals: list[MomentumSignal] = []
        self.detected_lags: list[MarketLag] = []

    async def check_polymarket_odds(self, asset: str) -> Optional[Decimal]:
        """
        Fetch current Polymarket odds for the asset's 15-min window.
        Returns the 'Yes' (UP) probability.
        """
        # This would connect to Polymarket API to get current odds
        # For now, return None as placeholder
        # In production, query the active 15-min BTC/ETH markets
        return None

    def estimate_edge(
        self,
        spot_direction: TradeDirection,
        spot_change: Decimal,
        current_odds: Decimal,
    ) -> Decimal:
        """
        Estimate the edge based on spot move vs current odds.

        If spot is up 0.3% and odds are still at 50%, there's edge
        because odds should be higher (say 55-60%).
        """
        # Simplified edge calculation
        # Real implementation would use historical correlation data
        expected_odds_adjustment = spot_change * Decimal("10")  # 0.3% move = ~3% odds shift

        if spot_direction == TradeDirection.UP:
            expected_odds = min(Decimal("0.5") + expected_odds_adjustment / 100, Decimal("0.95"))
            edge = expected_odds - current_odds
        else:
            expected_odds = max(Decimal("0.5") - expected_odds_adjustment / 100, Decimal("0.05"))
            edge = current_odds - expected_odds

        return max(edge, Decimal("0"))

    def record_signal(self, signal: MomentumSignal):
        """Record a momentum signal for lag analysis."""
        self.pending_signals.append(signal)
        # Keep only recent signals
        if len(self.pending_signals) > 100:
            self.pending_signals = self.pending_signals[-50:]
