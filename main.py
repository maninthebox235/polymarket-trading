#!/usr/bin/env python3
"""
Polymarket Wallet Tracker CLI

Track whale wallets, analyze momentum strategies, and detect edge
in crypto prediction markets.

Inspired by the gabagool22 strategy:
- BTC/ETH 15-minute windows
- Small consistent position sizing (~$1,600)
- Momentum reading: spot moves before PM odds adjust
- Entry around 40-50¢, resolution pays $1
"""

import asyncio
import sys
from decimal import Decimal

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import get_config
from src.equity_curve import EquityCurve
from src.momentum_detector import MomentumDetector
from src.strategy_analyzer import StrategyAnalyzer, compare_to_gabagool22
from src.wallet_tracker import WalletTracker

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Polymarket Wallet Tracker - Analyze whale wallets and trading strategies."""
    pass


@cli.command()
@click.argument("address")
@click.option("--full", is_flag=True, help="Show full detailed analysis")
@click.option("--compare-gabagool", is_flag=True, help="Compare to gabagool22 benchmark")
def analyze(address: str, full: bool, compare_gabagool: bool):
    """
    Analyze a Polymarket wallet.

    ADDRESS can be a wallet address or username (e.g., gabagool22)
    """
    asyncio.run(_analyze_wallet(address, full, compare_gabagool))


async def _analyze_wallet(address: str, full: bool, compare_gabagool: bool):
    """Run wallet analysis."""
    console.print(Panel(
        f"[bold cyan]Analyzing wallet: {address}[/bold cyan]",
        title="Polymarket Wallet Tracker"
    ))

    async with WalletTracker(address) as tracker:
        stats = await tracker.analyze_wallet()

        if stats.total_trades == 0:
            console.print("[yellow]No trades found for this wallet.[/yellow]")
            return

        # Print basic stats
        _print_wallet_summary(stats)

        if full:
            # Strategy analysis
            analyzer = StrategyAnalyzer(tracker.positions, stats)
            profile = analyzer.analyze()
            analyzer.print_analysis(profile)

            # Equity curve
            curve = EquityCurve(tracker.positions)
            curve.print_curve_ascii()
            curve.print_stats()

            description = curve.describe_curve_quality()
            console.print(f"\n[bold]Curve Assessment:[/bold] {description}")

        if compare_gabagool:
            analyzer = StrategyAnalyzer(tracker.positions, stats)
            profile = analyzer.analyze()
            scores = compare_to_gabagool22(profile)

            console.print("\n[bold cyan]Comparison to gabagool22:[/bold cyan]")
            for metric, score in scores.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                console.print(f"  {metric:20} [{bar}] {score:.0%}")


def _print_wallet_summary(stats):
    """Print wallet summary table."""
    table = Table(title="Wallet Summary")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    username = stats.username or "Unknown"
    table.add_row("Username", username)
    table.add_row("Address", f"{stats.address[:8]}...{stats.address[-6:]}")
    table.add_row("Total Trades", str(stats.total_trades))
    table.add_row("Win Rate", f"{stats.win_rate:.1f}%")
    table.add_row("Total Volume", f"${stats.total_volume:,.2f}")
    table.add_row("Total P&L", f"${stats.total_pnl:,.2f}")
    table.add_row("Avg Position Size", f"${stats.average_position_size:,.2f}")
    table.add_row("ROI", f"{stats.roi_percent:.1f}%")

    if stats.first_trade and stats.last_trade:
        days = (stats.last_trade - stats.first_trade).days
        table.add_row("Active Period", f"{days} days")

    console.print(table)


@cli.command()
@click.argument("addresses", nargs=-1)
def compare(addresses: tuple[str, ...]):
    """Compare multiple wallets side by side."""
    if len(addresses) < 2:
        console.print("[red]Please provide at least 2 addresses to compare.[/red]")
        return

    asyncio.run(_compare_wallets(list(addresses)))


async def _compare_wallets(addresses: list[str]):
    """Compare multiple wallets."""
    console.print(Panel(
        f"[bold cyan]Comparing {len(addresses)} wallets[/bold cyan]",
        title="Wallet Comparison"
    ))

    results = []
    for addr in addresses:
        async with WalletTracker(addr) as tracker:
            stats = await tracker.analyze_wallet()
            analyzer = StrategyAnalyzer(tracker.positions, stats)
            profile = analyzer.analyze()
            results.append((stats, profile))

    # Comparison table
    table = Table(title="Wallet Comparison")
    table.add_column("Metric", style="cyan")
    for stats, _ in results:
        name = stats.username or stats.address[:10]
        table.add_column(name, style="green")

    metrics = [
        ("Total P&L", lambda s, p: f"${s.total_pnl:,.0f}"),
        ("Win Rate", lambda s, p: f"{s.win_rate:.1f}%"),
        ("Avg Size", lambda s, p: f"${s.average_position_size:,.0f}"),
        ("Trades/Day", lambda s, p: f"{p.avg_trades_per_day:.1f}"),
        ("Strategy", lambda s, p: p.strategy_type),
        ("Consistency", lambda s, p: f"{p.sizing_consistency:.0%}"),
    ]

    for name, getter in metrics:
        row = [name]
        for stats, profile in results:
            row.append(getter(stats, profile))
        table.add_row(*row)

    console.print(table)


@cli.command()
@click.option("--threshold", default=0.15, help="Momentum threshold percentage")
def monitor(threshold: float):
    """
    Monitor spot prices for momentum signals.

    Watches Binance and Coinbase for BTC/ETH moves that could
    create Polymarket edge opportunities.
    """
    console.print(Panel(
        "[bold cyan]Starting Momentum Monitor[/bold cyan]\n"
        f"Threshold: {threshold}% move in 15-second window\n"
        "Press Ctrl+C to stop",
        title="Momentum Detector"
    ))

    detector = MomentumDetector(threshold=Decimal(str(threshold)))

    def on_signal(signal):
        direction_emoji = "🟢" if signal.direction.value == "up" else "🔴"
        console.print(
            f"{direction_emoji} [bold]{signal.asset}[/bold] "
            f"{signal.direction.value.upper()} {signal.price_change_percent:.2f}% "
            f"@ ${signal.spot_price:,.2f} "
            f"[dim]({signal.source})[/dim] "
            f"Strength: {signal.strength:.0%}"
        )

    detector.on_signal(on_signal)

    try:
        asyncio.run(detector.start())
    except KeyboardInterrupt:
        detector.stop()
        console.print("\n[yellow]Monitor stopped.[/yellow]")


@cli.command()
@click.argument("address")
@click.option("--output", "-o", help="Output file path")
def export(address: str, output: str):
    """Export wallet positions to JSON."""
    import json
    from datetime import datetime

    asyncio.run(_export_wallet(address, output))


async def _export_wallet(address: str, output: str):
    """Export wallet data."""
    async with WalletTracker(address) as tracker:
        stats = await tracker.analyze_wallet()

        data = {
            "address": stats.address,
            "username": stats.username,
            "exported_at": datetime.now().isoformat(),
            "stats": {
                "total_trades": stats.total_trades,
                "win_rate": float(stats.win_rate),
                "total_volume": float(stats.total_volume),
                "total_pnl": float(stats.total_pnl),
                "average_position_size": float(stats.average_position_size),
            },
            "positions": [
                {
                    "market_slug": p.market_slug,
                    "market_type": p.market_type.value,
                    "direction": p.direction.value,
                    "entry_price": float(p.entry_price),
                    "size_usd": float(p.size_usd),
                    "outcome": p.outcome.value,
                    "pnl": float(p.pnl) if p.pnl else None,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in tracker.positions
            ]
        }

        import json
        if output:
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]Exported to {output}[/green]")
        else:
            console.print(json.dumps(data, indent=2))


@cli.command()
def gabagool():
    """
    Analyze gabagool22 - the quietest $457K winner on Polymarket.

    This wallet exemplifies the momentum reading strategy:
    - BTC/ETH 15-minute windows only
    - $1,600 average position size (tiny but consistent)
    - Entry around 40-50 cents
    - Exploits spot price lag vs PM odds
    """
    console.print(Panel("""
[bold cyan]gabagool22 Strategy Analysis[/bold cyan]

The quietest automated wallet on Polymarket made $457K and nobody noticed.

[bold]The Strategy: Momentum Reading[/bold]

When BTC or ETH starts moving hard on Binance or Coinbase,
Polymarket odds lag behind by a few seconds. The 15-minute
window still shows old prices while the real move already happened.

He watches spot. Sees a push in one direction. Opens Polymarket.
The odds have not adjusted yet. He enters the side that should
win before the market catches up.

Entry around 40-50¢. Resolution pays $1. Small edge, but real.

[bold]Position Sizing (The Real Edge)[/bold]

Other wallets size big: $10K, $20K, $40K per trade.
gabagool22: $1,600 average. Tiny.

More entries. More data. More chances for the edge to play out.
Less damage when something misses.

[bold]The Equity Curve[/bold]

A straight line climbing for two months.
The biggest win would barely get a like on Twitter.
And that's exactly why nobody talks about him.

We want the hero moment. The screenshot. The trade.
gabagool22 never had one. He just made $457K without it.
""", title="The gabagool22 Breakdown"))

    # Run actual analysis if API is available
    console.print("\n[dim]Run 'polymarket analyze gabagool22 --full' for live data[/dim]")


if __name__ == "__main__":
    cli()
