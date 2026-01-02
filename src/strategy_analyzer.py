"""
Strategy analyzer for Polymarket wallets.

Analyzes trading patterns, position sizing, and strategy characteristics.
Specifically designed to identify patterns like gabagool22's momentum reading.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from rich.console import Console
from rich.table import Table

from .models import MarketType, Position, TradeOutcome, WalletStats

console = Console()


@dataclass
class StrategyProfile:
    """Profile of a wallet's trading strategy."""

    # Core metrics
    strategy_type: str  # "momentum", "arbitrage", "both_sides", "mixed"
    primary_markets: list[MarketType]
    avg_position_size: Decimal
    position_size_std: Decimal  # Consistency measure

    # Timing patterns
    avg_trades_per_day: Decimal
    trades_per_hour_distribution: dict[int, int]
    preferred_window: str  # "15min", "hourly", "daily", etc.

    # Performance
    win_rate: Decimal
    avg_win_size: Decimal
    avg_loss_size: Decimal
    profit_factor: Decimal  # gross_profit / gross_loss
    sharpe_estimate: Decimal

    # Risk profile
    max_position_size: Decimal
    max_concurrent_positions: int
    sizing_consistency: Decimal  # 0-1, higher = more consistent sizing

    # Edge indicators
    avg_entry_price: Decimal
    entry_price_vs_resolution: Decimal  # How far from 50¢
    edge_per_trade: Decimal


class StrategyAnalyzer:
    """
    Analyze a wallet's trading strategy from position history.

    Designed to identify patterns like:
    - gabagool22: Small consistent sizing, momentum reading, BTC/ETH 15-min windows
    - Account88888: Both-sides arbitrage
    - Large whale: Big positions, less frequent
    """

    def __init__(self, positions: list[Position], stats: WalletStats):
        self.positions = positions
        self.stats = stats

    def analyze(self) -> StrategyProfile:
        """Perform full strategy analysis."""
        if not self.positions:
            return self._empty_profile()

        # Calculate all metrics
        sizes = [p.size_usd for p in self.positions]
        avg_size = sum(sizes) / len(sizes)
        size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        size_std = size_variance ** Decimal("0.5")

        # Sizing consistency (lower std relative to mean = more consistent)
        sizing_consistency = Decimal("1") - min(size_std / avg_size, Decimal("1")) if avg_size > 0 else Decimal("0")

        # Market type distribution
        market_counts: dict[MarketType, int] = {}
        for p in self.positions:
            market_counts[p.market_type] = market_counts.get(p.market_type, 0) + 1

        primary_markets = sorted(market_counts.keys(), key=lambda m: market_counts[m], reverse=True)

        # Trading frequency
        if self.stats.first_trade and self.stats.last_trade:
            days_active = max((self.stats.last_trade - self.stats.first_trade).days, 1)
            trades_per_day = Decimal(len(self.positions)) / Decimal(days_active)
        else:
            trades_per_day = Decimal("0")

        # Hour distribution
        hour_dist: dict[int, int] = {h: 0 for h in range(24)}
        for p in self.positions:
            hour_dist[p.timestamp.hour] += 1

        # Win/loss analysis
        wins = [p for p in self.positions if p.outcome == TradeOutcome.WIN]
        losses = [p for p in self.positions if p.outcome == TradeOutcome.LOSS]

        avg_win = sum(p.pnl for p in wins if p.pnl) / len(wins) if wins else Decimal("0")
        avg_loss = abs(sum(p.pnl for p in losses if p.pnl) / len(losses)) if losses else Decimal("0")

        gross_profit = sum(p.pnl for p in wins if p.pnl and p.pnl > 0)
        gross_loss = abs(sum(p.pnl for p in losses if p.pnl and p.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")

        # Entry price analysis
        entry_prices = [p.entry_price for p in self.positions]
        avg_entry = sum(entry_prices) / len(entry_prices)
        entry_vs_resolution = abs(avg_entry - Decimal("0.5"))

        # Edge per trade
        resolved = [p for p in self.positions if p.outcome != TradeOutcome.PENDING]
        if resolved:
            total_pnl = sum(p.pnl for p in resolved if p.pnl)
            total_risked = sum(p.size_usd for p in resolved)
            edge_per_trade = (total_pnl / total_risked) * 100 if total_risked > 0 else Decimal("0")
        else:
            edge_per_trade = Decimal("0")

        # Determine strategy type
        strategy_type = self._classify_strategy(
            primary_markets=primary_markets,
            sizing_consistency=sizing_consistency,
            avg_entry=avg_entry,
            trades_per_day=trades_per_day,
        )

        # Determine preferred window
        preferred_window = self._detect_window_preference()

        # Estimate Sharpe-like ratio
        if resolved:
            returns = [float(p.pnl / p.size_usd) if p.pnl and p.size_usd > 0 else 0 for p in resolved]
            if returns and len(returns) > 1:
                mean_ret = sum(returns) / len(returns)
                ret_var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
                ret_std = ret_var ** 0.5
                sharpe = Decimal(str(mean_ret / ret_std)) if ret_std > 0 else Decimal("0")
            else:
                sharpe = Decimal("0")
        else:
            sharpe = Decimal("0")

        return StrategyProfile(
            strategy_type=strategy_type,
            primary_markets=primary_markets[:3],
            avg_position_size=avg_size,
            position_size_std=size_std,
            avg_trades_per_day=trades_per_day,
            trades_per_hour_distribution=hour_dist,
            preferred_window=preferred_window,
            win_rate=self.stats.win_rate,
            avg_win_size=avg_win,
            avg_loss_size=avg_loss,
            profit_factor=profit_factor,
            sharpe_estimate=sharpe,
            max_position_size=max(sizes),
            max_concurrent_positions=self._estimate_max_concurrent(),
            sizing_consistency=sizing_consistency,
            avg_entry_price=avg_entry,
            entry_price_vs_resolution=entry_vs_resolution,
            edge_per_trade=edge_per_trade,
        )

    def _classify_strategy(
        self,
        primary_markets: list[MarketType],
        sizing_consistency: Decimal,
        avg_entry: Decimal,
        trades_per_day: Decimal,
    ) -> str:
        """Classify the strategy type based on characteristics."""

        # Check for BTC/ETH focus (momentum reading pattern)
        is_crypto_focused = (
            MarketType.BTC in primary_markets[:2] or
            MarketType.ETH in primary_markets[:2]
        )

        # High frequency + consistent sizing + crypto = momentum reading
        if (
            is_crypto_focused and
            sizing_consistency > Decimal("0.6") and
            trades_per_day > Decimal("5")
        ):
            return "momentum_reading"

        # Entry around 50¢ suggests arbitrage or both-sides
        if abs(avg_entry - Decimal("0.5")) < Decimal("0.05"):
            return "arbitrage"

        # High volume, less consistent = whale accumulation
        if sizing_consistency < Decimal("0.4") and trades_per_day < Decimal("3"):
            return "whale_accumulation"

        # Default
        return "mixed"

    def _detect_window_preference(self) -> str:
        """Detect preferred market window (15min, hourly, daily)."""
        window_patterns = {
            "15min": ["15-minute", "15min", "15m"],
            "hourly": ["hourly", "1-hour", "1h"],
            "daily": ["daily", "24h", "day"],
        }

        counts = {window: 0 for window in window_patterns}

        for p in self.positions:
            slug = p.market_slug.lower()
            for window, patterns in window_patterns.items():
                if any(pat in slug for pat in patterns):
                    counts[window] += 1
                    break

        if max(counts.values()) > 0:
            return max(counts, key=counts.get)
        return "unknown"

    def _estimate_max_concurrent(self) -> int:
        """Estimate maximum concurrent positions."""
        if not self.positions:
            return 0

        # Sort by timestamp
        sorted_pos = sorted(self.positions, key=lambda p: p.timestamp)

        # Simplified: count positions within a 1-hour window
        max_concurrent = 1
        for i, pos in enumerate(sorted_pos):
            window_end = pos.timestamp + timedelta(hours=1)
            concurrent = sum(
                1 for p in sorted_pos[i:]
                if p.timestamp <= window_end
            )
            max_concurrent = max(max_concurrent, concurrent)

        return max_concurrent

    def _empty_profile(self) -> StrategyProfile:
        """Return empty profile when no positions."""
        return StrategyProfile(
            strategy_type="unknown",
            primary_markets=[],
            avg_position_size=Decimal("0"),
            position_size_std=Decimal("0"),
            avg_trades_per_day=Decimal("0"),
            trades_per_hour_distribution={},
            preferred_window="unknown",
            win_rate=Decimal("0"),
            avg_win_size=Decimal("0"),
            avg_loss_size=Decimal("0"),
            profit_factor=Decimal("0"),
            sharpe_estimate=Decimal("0"),
            max_position_size=Decimal("0"),
            max_concurrent_positions=0,
            sizing_consistency=Decimal("0"),
            avg_entry_price=Decimal("0"),
            entry_price_vs_resolution=Decimal("0"),
            edge_per_trade=Decimal("0"),
        )

    def print_analysis(self, profile: Optional[StrategyProfile] = None):
        """Print a formatted analysis report."""
        if profile is None:
            profile = self.analyze()

        console.print("\n[bold cyan]Strategy Analysis Report[/bold cyan]\n")

        # Summary table
        table = Table(title=f"Wallet: {self.stats.username or self.stats.address[:16]}...")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Strategy Type", profile.strategy_type.replace("_", " ").title())
        table.add_row("Primary Markets", ", ".join(m.value.upper() for m in profile.primary_markets))
        table.add_row("Preferred Window", profile.preferred_window)
        table.add_row("Avg Position Size", f"${profile.avg_position_size:,.2f}")
        table.add_row("Sizing Consistency", f"{profile.sizing_consistency:.1%}")
        table.add_row("Trades/Day", f"{profile.avg_trades_per_day:.1f}")
        table.add_row("Win Rate", f"{profile.win_rate:.1f}%")
        table.add_row("Profit Factor", f"{profile.profit_factor:.2f}")
        table.add_row("Avg Entry Price", f"{profile.avg_entry_price:.2f}¢")
        table.add_row("Edge/Trade", f"{profile.edge_per_trade:.2f}%")

        console.print(table)

        # Strategy description
        console.print(f"\n[bold]Strategy Profile:[/bold]")
        self._print_strategy_description(profile)

    def _print_strategy_description(self, profile: StrategyProfile):
        """Print human-readable strategy description."""
        if profile.strategy_type == "momentum_reading":
            console.print("""
[yellow]Momentum Reading Strategy Detected[/yellow]

This wallet exhibits the gabagool22 pattern:
- Focus on BTC/ETH short-term windows (likely 15-minute)
- Consistent position sizing (low variance)
- High trade frequency
- Entry prices suggesting directional bets on momentum

The strategy appears to exploit lag between spot price moves on
exchanges (Binance/Coinbase) and Polymarket odds adjustment.
When spot moves hard, PM odds take seconds to catch up.
""")
        elif profile.strategy_type == "arbitrage":
            console.print("""
[yellow]Arbitrage/Both-Sides Strategy Detected[/yellow]

This wallet shows characteristics of:
- Entries near 50¢ (neutral odds)
- Likely betting both sides to lock in spreads
- Lower directional risk, profit from odds discrepancies
""")
        elif profile.strategy_type == "whale_accumulation":
            console.print("""
[yellow]Whale Accumulation Pattern Detected[/yellow]

This wallet shows:
- Variable position sizes (building positions over time)
- Lower trade frequency
- Likely longer-term thesis bets
""")
        else:
            console.print("""
[yellow]Mixed Strategy[/yellow]

This wallet doesn't fit a clear pattern.
May use multiple strategies or opportunistic trading.
""")


def compare_to_gabagool22(profile: StrategyProfile) -> dict[str, float]:
    """
    Compare a strategy profile to the gabagool22 benchmark.

    Returns similarity scores for key characteristics.
    """
    # gabagool22 benchmark values
    gabagool_benchmark = {
        "avg_position_size": Decimal("1600"),  # $1,600 average
        "sizing_consistency": Decimal("0.85"),  # Very consistent
        "trades_per_day": Decimal("15"),  # High frequency
        "win_rate": Decimal("55"),  # Slightly above 50%
        "entry_price": Decimal("0.45"),  # Entry around 40-50¢
        "strategy_type": "momentum_reading",
    }

    scores = {}

    # Position size similarity (within 50% = good match)
    size_diff = abs(profile.avg_position_size - gabagool_benchmark["avg_position_size"])
    size_ratio = size_diff / gabagool_benchmark["avg_position_size"]
    scores["position_sizing"] = max(0, 1 - float(size_ratio))

    # Consistency similarity
    scores["consistency"] = float(profile.sizing_consistency / gabagool_benchmark["sizing_consistency"])

    # Frequency similarity
    freq_ratio = float(profile.avg_trades_per_day / gabagool_benchmark["trades_per_day"])
    scores["frequency"] = min(freq_ratio, 1.0)

    # Strategy type match
    scores["strategy_match"] = 1.0 if profile.strategy_type == gabagool_benchmark["strategy_type"] else 0.3

    # Overall similarity
    scores["overall"] = sum(scores.values()) / len(scores)

    return scores
