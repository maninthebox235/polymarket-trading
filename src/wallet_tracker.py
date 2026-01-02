"""
Core wallet tracking module for Polymarket.

Tracks wallet activity, positions, and calculates performance metrics.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import httpx
from rich.console import Console

from .config import get_config
from .models import (
    MarketType,
    Position,
    TradeDirection,
    TradeOutcome,
    WalletStats,
)

console = Console()


class WalletTracker:
    """Track and analyze Polymarket wallet activity."""

    # Known BTC/ETH market patterns
    BTC_PATTERNS = ["bitcoin", "btc", "₿"]
    ETH_PATTERNS = ["ethereum", "eth", "ether"]

    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address.lower()
        self.config = get_config()
        self.positions: list[Position] = []
        self.stats: Optional[WalletStats] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Tracker not initialized. Use 'async with' context.")
        return self._client

    def _classify_market(self, market_question: str, market_slug: str) -> MarketType:
        """Classify market as BTC, ETH, or other."""
        text = f"{market_question} {market_slug}".lower()

        for pattern in self.BTC_PATTERNS:
            if pattern in text:
                return MarketType.BTC

        for pattern in self.ETH_PATTERNS:
            if pattern in text:
                return MarketType.ETH

        return MarketType.OTHER

    def _parse_direction(self, outcome: str) -> TradeDirection:
        """Parse trade direction from outcome string."""
        outcome_lower = outcome.lower()
        if any(word in outcome_lower for word in ["yes", "up", "above", "higher", "over"]):
            return TradeDirection.UP
        return TradeDirection.DOWN

    async def fetch_wallet_positions(self) -> list[dict]:
        """Fetch all positions for a wallet from Polymarket API."""
        url = f"{self.config.polymarket.gamma_url}/positions"
        params = {"user": self.wallet_address}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            console.print(f"[red]Error fetching positions: {e}[/red]")
            return []

    async def fetch_wallet_trades(self, limit: int = 1000) -> list[dict]:
        """Fetch trade history for a wallet."""
        url = f"{self.config.polymarket.api_url}/trades"
        params = {"maker": self.wallet_address, "limit": limit}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("trades", [])
        except httpx.HTTPError as e:
            console.print(f"[red]Error fetching trades: {e}[/red]")
            return []

    async def fetch_wallet_profile(self) -> Optional[dict]:
        """Fetch wallet profile info if available."""
        url = f"{self.config.polymarket.gamma_url}/users"
        params = {"address": self.wallet_address}

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            users = response.json()
            return users[0] if users else None
        except httpx.HTTPError:
            return None

    async def fetch_market_info(self, market_id: str) -> Optional[dict]:
        """Fetch market details."""
        url = f"{self.config.polymarket.gamma_url}/markets/{market_id}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return None

    async def analyze_wallet(self) -> WalletStats:
        """Perform full wallet analysis."""
        console.print(f"[cyan]Analyzing wallet: {self.wallet_address}[/cyan]")

        # Fetch data in parallel
        positions_data, trades_data, profile = await asyncio.gather(
            self.fetch_wallet_positions(),
            self.fetch_wallet_trades(),
            self.fetch_wallet_profile(),
        )

        username = profile.get("username") if profile else None

        # Process positions
        self.positions = []
        total_volume = Decimal("0")
        total_pnl = Decimal("0")
        winning = 0
        losing = 0
        pending = 0
        entry_prices = []
        timestamps = []

        for pos_data in positions_data:
            try:
                market_question = pos_data.get("market", {}).get("question", "")
                market_slug = pos_data.get("market", {}).get("slug", "")
                market_type = self._classify_market(market_question, market_slug)

                outcome_str = pos_data.get("outcome", "Yes")
                size = Decimal(str(pos_data.get("size", 0)))
                avg_price = Decimal(str(pos_data.get("avgPrice", 0.5)))

                if size == 0:
                    continue

                size_usd = size * avg_price
                total_volume += size_usd

                # Parse timestamp
                created = pos_data.get("createdAt")
                if created:
                    try:
                        ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    except ValueError:
                        ts = datetime.now(timezone.utc)
                else:
                    ts = datetime.now(timezone.utc)

                timestamps.append(ts)
                entry_prices.append(avg_price)

                # Determine outcome
                realized_pnl = Decimal(str(pos_data.get("realizedPnl", 0)))
                cur_value = Decimal(str(pos_data.get("curValue", 0)))

                if pos_data.get("settled", False):
                    pnl = realized_pnl
                    if pnl > 0:
                        outcome = TradeOutcome.WIN
                        winning += 1
                    else:
                        outcome = TradeOutcome.LOSS
                        losing += 1
                    total_pnl += pnl
                else:
                    outcome = TradeOutcome.PENDING
                    pending += 1
                    pnl = cur_value - size_usd  # Unrealized

                position = Position(
                    market_id=pos_data.get("market", {}).get("id", ""),
                    market_slug=market_slug,
                    market_type=market_type,
                    direction=self._parse_direction(outcome_str),
                    entry_price=avg_price,
                    size_usd=size_usd,
                    shares=size,
                    timestamp=ts,
                    outcome=outcome,
                    pnl=pnl,
                )
                self.positions.append(position)

            except (KeyError, ValueError) as e:
                console.print(f"[yellow]Skipping position: {e}[/yellow]")
                continue

        total_trades = winning + losing + pending

        self.stats = WalletStats(
            address=self.wallet_address,
            username=username,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            pending_trades=pending,
            total_volume=total_volume,
            total_pnl=total_pnl,
            average_position_size=total_volume / total_trades if total_trades > 0 else Decimal("0"),
            win_rate=Decimal(winning) / Decimal(winning + losing) * 100 if (winning + losing) > 0 else Decimal("0"),
            avg_entry_price=sum(entry_prices) / len(entry_prices) if entry_prices else Decimal("0"),
            first_trade=min(timestamps) if timestamps else None,
            last_trade=max(timestamps) if timestamps else None,
        )

        return self.stats

    def get_btc_eth_positions(self) -> list[Position]:
        """Filter positions to only BTC and ETH markets."""
        return [p for p in self.positions if p.market_type in (MarketType.BTC, MarketType.ETH)]

    def get_15min_window_positions(self) -> list[Position]:
        """
        Filter for 15-minute window positions (the gabagool22 strategy).
        These are typically short-term BTC/ETH price prediction markets.
        """
        btc_eth = self.get_btc_eth_positions()
        # 15-min windows usually have specific slug patterns
        window_patterns = ["15-minute", "15min", "15m", "minute"]
        return [
            p for p in btc_eth
            if any(pattern in p.market_slug.lower() for pattern in window_patterns)
        ]


async def track_wallet(address: str) -> WalletStats:
    """Convenience function to track a wallet."""
    async with WalletTracker(address) as tracker:
        return await tracker.analyze_wallet()


async def compare_wallets(addresses: list[str]) -> list[WalletStats]:
    """Track and compare multiple wallets."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for addr in addresses:
            tracker = WalletTracker(addr)
            tracker._client = client
            tasks.append(tracker.analyze_wallet())

        return await asyncio.gather(*tasks)
