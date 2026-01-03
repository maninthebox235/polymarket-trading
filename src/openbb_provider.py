"""
OpenBB data provider for enhanced crypto market data.

Provides historical and real-time cryptocurrency data through OpenBB Platform,
offering an alternative/supplement to direct exchange WebSocket feeds.

Key capabilities:
- Historical BTC/ETH price data from multiple providers
- Backtesting support for momentum strategy validation
- Technical indicators and market analytics
- Multi-provider data aggregation
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import get_config
from .models import MarketType, MomentumSignal, TradeDirection

console = Console()


class OpenBBProvider(str, Enum):
    """Supported OpenBB data providers for crypto."""
    YFINANCE = "yfinance"
    FMP = "fmp"
    POLYGON = "polygon"
    TIINGO = "tiingo"


@dataclass
class PriceData:
    """Represents a single price data point."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Optional[Decimal] = None


@dataclass
class CryptoSnapshot:
    """Current crypto market snapshot."""
    symbol: str
    price: Decimal
    change_24h: Decimal
    change_percent_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    volume_24h: Optional[Decimal] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class OpenBBDataProvider:
    """
    OpenBB Platform integration for cryptocurrency data.

    Provides historical price data, real-time quotes, and technical
    analysis capabilities through the OpenBB ecosystem.
    """

    # Symbol mappings for different providers
    SYMBOL_MAP = {
        MarketType.BTC: {
            OpenBBProvider.YFINANCE: "BTC-USD",
            OpenBBProvider.FMP: "BTCUSD",
            OpenBBProvider.POLYGON: "X:BTCUSD",
            OpenBBProvider.TIINGO: "btcusd",
        },
        MarketType.ETH: {
            OpenBBProvider.YFINANCE: "ETH-USD",
            OpenBBProvider.FMP: "ETHUSD",
            OpenBBProvider.POLYGON: "X:ETHUSD",
            OpenBBProvider.TIINGO: "ethusd",
        },
    }

    def __init__(
        self,
        provider: OpenBBProvider = OpenBBProvider.YFINANCE,
    ):
        """
        Initialize OpenBB data provider.

        Args:
            provider: The data provider to use (default: yfinance for free access)
        """
        self.provider = provider
        self.config = get_config()
        self._obb = None

    def _get_obb(self):
        """Lazy initialization of OpenBB client."""
        if self._obb is None:
            try:
                from openbb import obb
                self._obb = obb
            except ImportError:
                raise ImportError(
                    "OpenBB is not installed. Run: pip install openbb"
                )
        return self._obb

    def _get_symbol(self, market_type: MarketType) -> str:
        """Get the correct symbol for the provider."""
        if market_type not in self.SYMBOL_MAP:
            raise ValueError(f"Unsupported market type: {market_type}")
        return self.SYMBOL_MAP[market_type].get(
            self.provider,
            self.SYMBOL_MAP[market_type][OpenBBProvider.YFINANCE]
        )

    def get_historical_prices(
        self,
        market_type: MarketType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetch historical price data for BTC or ETH.

        Args:
            market_type: BTC or ETH
            start_date: Start date (default: 7 days ago)
            end_date: End date (default: now)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)

        Returns:
            DataFrame with OHLCV data
        """
        obb = self._get_obb()
        symbol = self._get_symbol(market_type)

        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        try:
            result = obb.crypto.price.historical(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                provider=self.provider.value,
            )
            return result.to_dataframe()
        except Exception as e:
            console.print(f"[red]Error fetching {market_type.value} data: {e}[/red]")
            return pd.DataFrame()

    def get_current_price(self, market_type: MarketType) -> Optional[CryptoSnapshot]:
        """
        Get current price snapshot for BTC or ETH.

        Args:
            market_type: BTC or ETH

        Returns:
            CryptoSnapshot with current market data
        """
        # Get recent 1-minute data and use the latest
        df = self.get_historical_prices(
            market_type,
            start_date=datetime.now(timezone.utc) - timedelta(hours=24),
            end_date=datetime.now(timezone.utc),
            interval="1m",
        )

        if df.empty:
            return None

        latest = df.iloc[-1]
        first = df.iloc[0]

        # Calculate 24h change
        open_price = Decimal(str(first["open"]))
        close_price = Decimal(str(latest["close"]))
        change = close_price - open_price
        change_pct = (change / open_price) * 100 if open_price else Decimal(0)

        return CryptoSnapshot(
            symbol=market_type.value,
            price=close_price,
            change_24h=change,
            change_percent_24h=change_pct,
            high_24h=Decimal(str(df["high"].max())),
            low_24h=Decimal(str(df["low"].min())),
            volume_24h=Decimal(str(df["volume"].sum())) if "volume" in df else None,
            timestamp=datetime.now(timezone.utc),
        )

    def calculate_momentum(
        self,
        market_type: MarketType,
        window_minutes: int = 15,
    ) -> Optional[tuple[Decimal, TradeDirection]]:
        """
        Calculate price momentum over a time window.

        Args:
            market_type: BTC or ETH
            window_minutes: Lookback window in minutes

        Returns:
            (percent_change, direction) or None if insufficient data
        """
        df = self.get_historical_prices(
            market_type,
            start_date=datetime.now(timezone.utc) - timedelta(minutes=window_minutes + 5),
            end_date=datetime.now(timezone.utc),
            interval="1m",
        )

        if len(df) < 2:
            return None

        # Use the last N rows based on window
        window_data = df.tail(window_minutes)
        if len(window_data) < 2:
            return None

        first_price = Decimal(str(window_data.iloc[0]["close"]))
        last_price = Decimal(str(window_data.iloc[-1]["close"]))

        if first_price == 0:
            return None

        change_pct = ((last_price - first_price) / first_price) * 100
        direction = TradeDirection.UP if change_pct > 0 else TradeDirection.DOWN

        return (abs(change_pct), direction)

    def get_momentum_signal(
        self,
        market_type: MarketType,
        threshold: Decimal = Decimal("0.15"),
        window_minutes: int = 15,
    ) -> Optional[MomentumSignal]:
        """
        Generate a momentum signal if threshold is exceeded.

        Args:
            market_type: BTC or ETH
            threshold: Minimum % change to trigger signal
            window_minutes: Lookback window

        Returns:
            MomentumSignal if momentum exceeds threshold, else None
        """
        result = self.calculate_momentum(market_type, window_minutes)

        if result is None:
            return None

        change_pct, direction = result

        if change_pct < threshold:
            return None

        snapshot = self.get_current_price(market_type)
        spot_price = snapshot.price if snapshot else Decimal(0)

        # Calculate signal strength (0-1)
        strength = min(change_pct / Decimal("0.5"), Decimal("1.0"))

        return MomentumSignal(
            asset=market_type.value,
            direction=direction,
            spot_price=spot_price,
            price_change_percent=change_pct,
            timestamp=datetime.now(timezone.utc),
            source=f"openbb:{self.provider.value}",
            strength=strength,
        )

    def backtest_momentum_strategy(
        self,
        market_type: MarketType,
        start_date: datetime,
        end_date: datetime,
        threshold: Decimal = Decimal("0.15"),
        window_minutes: int = 15,
    ) -> pd.DataFrame:
        """
        Backtest momentum strategy on historical data.

        Identifies all points where momentum exceeded threshold
        and tracks hypothetical trade outcomes.

        Args:
            market_type: BTC or ETH
            start_date: Backtest start date
            end_date: Backtest end date
            threshold: Momentum threshold
            window_minutes: Lookback window

        Returns:
            DataFrame with backtest results
        """
        df = self.get_historical_prices(
            market_type,
            start_date=start_date,
            end_date=end_date,
            interval="1m",
        )

        if df.empty:
            return pd.DataFrame()

        signals = []

        for i in range(window_minutes, len(df)):
            window = df.iloc[i - window_minutes:i]
            first_price = window.iloc[0]["close"]
            current_price = window.iloc[-1]["close"]

            change_pct = ((current_price - first_price) / first_price) * 100

            if abs(change_pct) >= float(threshold):
                direction = "up" if change_pct > 0 else "down"

                # Look 15 minutes ahead for outcome
                if i + window_minutes < len(df):
                    future_price = df.iloc[i + window_minutes]["close"]
                    future_change = ((future_price - current_price) / current_price) * 100

                    # Did price continue in same direction?
                    if direction == "up":
                        won = future_change > 0
                    else:
                        won = future_change < 0

                    signals.append({
                        "timestamp": df.index[i] if hasattr(df.index, '__iter__') else i,
                        "entry_price": current_price,
                        "direction": direction,
                        "momentum_pct": abs(change_pct),
                        "future_price": future_price,
                        "future_change_pct": future_change,
                        "won": won,
                    })

        return pd.DataFrame(signals)

    def print_market_summary(self):
        """Print a summary of current crypto market conditions."""
        table = Table(title="Crypto Market Summary (via OpenBB)")

        table.add_column("Asset", style="cyan")
        table.add_column("Price", style="green")
        table.add_column("24h Change", style="yellow")
        table.add_column("24h High", style="dim")
        table.add_column("24h Low", style="dim")
        table.add_column("Momentum (15m)", style="magenta")

        for market in [MarketType.BTC, MarketType.ETH]:
            snapshot = self.get_current_price(market)
            momentum = self.calculate_momentum(market, 15)

            if snapshot:
                change_color = "green" if snapshot.change_percent_24h >= 0 else "red"
                change_str = f"[{change_color}]{snapshot.change_percent_24h:+.2f}%[/{change_color}]"

                mom_str = "-"
                if momentum:
                    mom_pct, mom_dir = momentum
                    mom_color = "green" if mom_dir == TradeDirection.UP else "red"
                    arrow = "↑" if mom_dir == TradeDirection.UP else "↓"
                    mom_str = f"[{mom_color}]{arrow} {mom_pct:.2f}%[/{mom_color}]"

                table.add_row(
                    market.value,
                    f"${snapshot.price:,.2f}",
                    change_str,
                    f"${snapshot.high_24h:,.2f}",
                    f"${snapshot.low_24h:,.2f}",
                    mom_str,
                )
            else:
                table.add_row(market.value, "N/A", "N/A", "N/A", "N/A", "N/A")

        console.print(table)


class OpenBBMomentumMonitor:
    """
    Alternative momentum monitor using OpenBB polling.

    For use when WebSocket connections are unavailable or as a
    backup/validation source for the primary WebSocket feeds.
    """

    def __init__(
        self,
        provider: OpenBBProvider = OpenBBProvider.YFINANCE,
        threshold: Decimal = Decimal("0.15"),
        poll_interval: int = 10,
    ):
        self.data_provider = OpenBBDataProvider(provider)
        self.threshold = threshold
        self.poll_interval = poll_interval
        self._running = False
        self._callbacks = []

    def on_signal(self, callback):
        """Register a callback for momentum signals."""
        self._callbacks.append(callback)

    def _emit_signal(self, signal: MomentumSignal):
        """Emit signal to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                console.print(f"[red]Callback error: {e}[/red]")

    async def start(self):
        """Start polling for momentum signals."""
        self._running = True
        console.print(f"[cyan]Starting OpenBB momentum monitor (polling every {self.poll_interval}s)...[/cyan]")

        while self._running:
            for market in [MarketType.BTC, MarketType.ETH]:
                try:
                    signal = self.data_provider.get_momentum_signal(
                        market,
                        threshold=self.threshold,
                    )
                    if signal:
                        self._emit_signal(signal)
                except Exception as e:
                    console.print(f"[yellow]Error checking {market.value}: {e}[/yellow]")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop the monitor."""
        self._running = False
        console.print("[cyan]OpenBB momentum monitor stopped[/cyan]")
