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
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import get_config
from src.equity_curve import EquityCurve
from src.momentum_detector import MomentumDetector
from src.openbb_provider import (
    OpenBBDataProvider,
    OpenBBMomentumMonitor,
    OpenBBProvider,
)
from src.models import MarketType
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


@cli.group()
def openbb():
    """OpenBB-powered market data and analysis commands."""
    pass


@openbb.command("market")
@click.option("--provider", default="yfinance", help="Data provider (yfinance, fmp, polygon, tiingo)")
def openbb_market(provider: str):
    """
    Show current crypto market summary.

    Displays BTC/ETH prices, 24h changes, and 15-minute momentum
    using OpenBB data providers.
    """
    try:
        provider_enum = OpenBBProvider(provider)
    except ValueError:
        console.print(f"[red]Unknown provider: {provider}. Use: yfinance, fmp, polygon, tiingo[/red]")
        return

    console.print(Panel(
        f"[bold cyan]Crypto Market Summary[/bold cyan]\n"
        f"Data Provider: {provider}",
        title="OpenBB Market Data"
    ))

    try:
        data_provider = OpenBBDataProvider(provider_enum)
        data_provider.print_market_summary()
    except ImportError:
        console.print("[red]OpenBB is not installed. Run: pip install openbb[/red]")
    except Exception as e:
        console.print(f"[red]Error fetching market data: {e}[/red]")


@openbb.command("history")
@click.argument("asset", type=click.Choice(["btc", "eth"], case_sensitive=False))
@click.option("--days", default=7, help="Number of days of history")
@click.option("--interval", default="1h", help="Data interval (1m, 5m, 15m, 1h, 1d)")
@click.option("--provider", default="yfinance", help="Data provider")
def openbb_history(asset: str, days: int, interval: str, provider: str):
    """
    Fetch historical price data for BTC or ETH.

    Displays price history with OHLCV data from OpenBB.
    """
    try:
        provider_enum = OpenBBProvider(provider)
    except ValueError:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return

    market_type = MarketType.BTC if asset.upper() == "BTC" else MarketType.ETH

    console.print(Panel(
        f"[bold cyan]Historical Data: {asset.upper()}[/bold cyan]\n"
        f"Period: {days} days | Interval: {interval} | Provider: {provider}",
        title="OpenBB Historical Data"
    ))

    try:
        data_provider = OpenBBDataProvider(provider_enum)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        df = data_provider.get_historical_prices(
            market_type,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        if df.empty:
            console.print("[yellow]No data available for the specified period.[/yellow]")
            return

        # Print summary stats
        table = Table(title=f"{asset.upper()} Price Statistics ({days} days)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Data Points", str(len(df)))
        table.add_row("Highest Price", f"${df['high'].max():,.2f}")
        table.add_row("Lowest Price", f"${df['low'].min():,.2f}")
        table.add_row("Latest Close", f"${df['close'].iloc[-1]:,.2f}")

        if "volume" in df.columns:
            table.add_row("Total Volume", f"${df['volume'].sum():,.0f}")

        # Price range
        price_range = df['high'].max() - df['low'].min()
        range_pct = (price_range / df['low'].min()) * 100
        table.add_row("Price Range", f"${price_range:,.2f} ({range_pct:.1f}%)")

        console.print(table)

        # Recent prices
        console.print("\n[bold]Recent Prices:[/bold]")
        recent = df.tail(10)
        recent_table = Table()
        recent_table.add_column("Time", style="dim")
        recent_table.add_column("Open", style="cyan")
        recent_table.add_column("High", style="green")
        recent_table.add_column("Low", style="red")
        recent_table.add_column("Close", style="yellow")

        for idx, row in recent.iterrows():
            time_str = str(idx)[-19:] if hasattr(idx, '__str__') else str(idx)
            recent_table.add_row(
                time_str,
                f"${row['open']:,.2f}",
                f"${row['high']:,.2f}",
                f"${row['low']:,.2f}",
                f"${row['close']:,.2f}",
            )

        console.print(recent_table)

    except ImportError:
        console.print("[red]OpenBB is not installed. Run: pip install openbb[/red]")
    except Exception as e:
        console.print(f"[red]Error fetching historical data: {e}[/red]")


@openbb.command("backtest")
@click.argument("asset", type=click.Choice(["btc", "eth"], case_sensitive=False))
@click.option("--days", default=30, help="Number of days to backtest")
@click.option("--threshold", default=0.15, help="Momentum threshold percentage")
@click.option("--provider", default="yfinance", help="Data provider")
def openbb_backtest(asset: str, days: int, threshold: float, provider: str):
    """
    Backtest the momentum strategy on historical data.

    Identifies historical momentum signals and calculates
    hypothetical win rates and outcomes.
    """
    try:
        provider_enum = OpenBBProvider(provider)
    except ValueError:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return

    market_type = MarketType.BTC if asset.upper() == "BTC" else MarketType.ETH

    console.print(Panel(
        f"[bold cyan]Momentum Strategy Backtest: {asset.upper()}[/bold cyan]\n"
        f"Period: {days} days | Threshold: {threshold}% | Provider: {provider}",
        title="OpenBB Backtest"
    ))

    try:
        data_provider = OpenBBDataProvider(provider_enum)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        console.print("[dim]Running backtest...[/dim]")

        results = data_provider.backtest_momentum_strategy(
            market_type,
            start_date=start_date,
            end_date=end_date,
            threshold=Decimal(str(threshold)),
        )

        if results.empty:
            console.print("[yellow]No momentum signals found in the period.[/yellow]")
            return

        # Calculate stats
        total_signals = len(results)
        wins = results["won"].sum()
        win_rate = (wins / total_signals) * 100

        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Signals", str(total_signals))
        table.add_row("Wins", str(wins))
        table.add_row("Losses", str(total_signals - wins))
        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Avg Momentum", f"{results['momentum_pct'].mean():.3f}%")
        table.add_row("Max Momentum", f"{results['momentum_pct'].max():.3f}%")

        # Direction breakdown
        up_signals = results[results["direction"] == "up"]
        down_signals = results[results["direction"] == "down"]

        if len(up_signals) > 0:
            up_wr = (up_signals["won"].sum() / len(up_signals)) * 100
            table.add_row("UP Signals Win Rate", f"{up_wr:.1f}% ({len(up_signals)} signals)")

        if len(down_signals) > 0:
            down_wr = (down_signals["won"].sum() / len(down_signals)) * 100
            table.add_row("DOWN Signals Win Rate", f"{down_wr:.1f}% ({len(down_signals)} signals)")

        console.print(table)

        # Show sample signals
        console.print("\n[bold]Sample Signals:[/bold]")
        sample = results.head(10)
        sample_table = Table()
        sample_table.add_column("Direction", style="cyan")
        sample_table.add_column("Momentum %", style="yellow")
        sample_table.add_column("Entry Price", style="dim")
        sample_table.add_column("Future Price", style="dim")
        sample_table.add_column("Result", style="green")

        for _, row in sample.iterrows():
            result_str = "[green]WIN[/green]" if row["won"] else "[red]LOSS[/red]"
            dir_emoji = "↑" if row["direction"] == "up" else "↓"
            sample_table.add_row(
                f"{dir_emoji} {row['direction'].upper()}",
                f"{row['momentum_pct']:.3f}%",
                f"${row['entry_price']:,.2f}",
                f"${row['future_price']:,.2f}",
                result_str,
            )

        console.print(sample_table)

        # Assessment
        console.print("\n[bold]Assessment:[/bold]")
        if win_rate >= 55:
            console.print(f"[green]Strong edge detected! {win_rate:.1f}% win rate suggests viable strategy.[/green]")
        elif win_rate >= 50:
            console.print(f"[yellow]Marginal edge. {win_rate:.1f}% win rate - consider higher threshold.[/yellow]")
        else:
            console.print(f"[red]No edge found. {win_rate:.1f}% win rate is below breakeven.[/red]")

    except ImportError:
        console.print("[red]OpenBB is not installed. Run: pip install openbb[/red]")
    except Exception as e:
        console.print(f"[red]Error running backtest: {e}[/red]")


@openbb.command("monitor")
@click.option("--threshold", default=0.15, help="Momentum threshold percentage")
@click.option("--interval", default=10, help="Polling interval in seconds")
@click.option("--provider", default="yfinance", help="Data provider")
def openbb_monitor(threshold: float, interval: int, provider: str):
    """
    Monitor crypto prices for momentum signals using OpenBB.

    Alternative to WebSocket-based monitoring, uses polling.
    Useful when exchange WebSockets are unavailable.
    """
    try:
        provider_enum = OpenBBProvider(provider)
    except ValueError:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return

    console.print(Panel(
        f"[bold cyan]OpenBB Momentum Monitor[/bold cyan]\n"
        f"Threshold: {threshold}% | Poll Interval: {interval}s | Provider: {provider}\n"
        "Press Ctrl+C to stop",
        title="OpenBB Monitor"
    ))

    try:
        monitor = OpenBBMomentumMonitor(
            provider=provider_enum,
            threshold=Decimal(str(threshold)),
            poll_interval=interval,
        )

        def on_signal(signal):
            direction_emoji = "🟢" if signal.direction.value == "up" else "🔴"
            console.print(
                f"{direction_emoji} [bold]{signal.asset}[/bold] "
                f"{signal.direction.value.upper()} {signal.price_change_percent:.2f}% "
                f"@ ${signal.spot_price:,.2f} "
                f"[dim]({signal.source})[/dim] "
                f"Strength: {signal.strength:.0%}"
            )

        monitor.on_signal(on_signal)
        asyncio.run(monitor.start())

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")
    except ImportError:
        console.print("[red]OpenBB is not installed. Run: pip install openbb[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    cli()
