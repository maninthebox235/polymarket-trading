"""
Equity curve tracking and visualization.

Builds cumulative P&L curves, calculates drawdowns, and
generates performance metrics for wallet analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from rich.console import Console
from rich.table import Table

from .models import EquityPoint, Position, TradeOutcome

console = Console()


@dataclass
class EquityCurveStats:
    """Statistics derived from the equity curve."""

    total_pnl: Decimal
    peak_pnl: Decimal
    max_drawdown: Decimal
    max_drawdown_percent: Decimal
    max_drawdown_duration_days: int
    current_drawdown: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    win_streak_max: int
    lose_streak_max: int
    recovery_factor: Decimal  # total_pnl / max_drawdown


class EquityCurve:
    """
    Build and analyze equity curves from position history.

    The equity curve is the cumulative P&L over time - a straight line
    climbing steadily (like gabagool22's) indicates consistent edge.
    """

    def __init__(self, positions: list[Position]):
        self.positions = sorted(
            [p for p in positions if p.outcome != TradeOutcome.PENDING],
            key=lambda p: p.resolution_time or p.timestamp
        )
        self.curve: list[EquityPoint] = []
        self._build_curve()

    def _build_curve(self):
        """Build the equity curve from positions."""
        if not self.positions:
            return

        cumulative_pnl = Decimal("0")
        peak = Decimal("0")

        for i, pos in enumerate(self.positions):
            pnl = pos.pnl or Decimal("0")
            cumulative_pnl += pnl
            peak = max(peak, cumulative_pnl)

            drawdown = peak - cumulative_pnl
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else Decimal("0")

            timestamp = pos.resolution_time or pos.timestamp

            point = EquityPoint(
                timestamp=timestamp,
                cumulative_pnl=cumulative_pnl,
                trade_count=i + 1,
                drawdown=drawdown,
                drawdown_percent=drawdown_pct,
            )
            self.curve.append(point)

    def get_stats(self) -> EquityCurveStats:
        """Calculate comprehensive statistics from the curve."""
        if not self.curve:
            return self._empty_stats()

        # Basic metrics
        total_pnl = self.curve[-1].cumulative_pnl
        peak_pnl = max(p.cumulative_pnl for p in self.curve)

        # Drawdown analysis
        max_dd = max(p.drawdown for p in self.curve)
        max_dd_pct = max(p.drawdown_percent for p in self.curve)
        current_dd = self.curve[-1].drawdown

        # Drawdown duration
        max_dd_duration = self._calculate_max_drawdown_duration()

        # Streaks
        win_streak, lose_streak = self._calculate_streaks()

        # Risk-adjusted returns
        returns = self._calculate_returns()
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        calmar = total_pnl / max_dd if max_dd > 0 else Decimal("999")

        recovery_factor = total_pnl / max_dd if max_dd > 0 else Decimal("999")

        return EquityCurveStats(
            total_pnl=total_pnl,
            peak_pnl=peak_pnl,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            max_drawdown_duration_days=max_dd_duration,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            win_streak_max=win_streak,
            lose_streak_max=lose_streak,
            recovery_factor=recovery_factor,
        )

    def _calculate_max_drawdown_duration(self) -> int:
        """Calculate longest time spent in drawdown."""
        if not self.curve:
            return 0

        max_duration = 0
        dd_start: Optional[datetime] = None

        for point in self.curve:
            if point.drawdown > 0:
                if dd_start is None:
                    dd_start = point.timestamp
            else:
                if dd_start is not None:
                    duration = (point.timestamp - dd_start).days
                    max_duration = max(max_duration, duration)
                    dd_start = None

        # Check if still in drawdown
        if dd_start is not None:
            duration = (self.curve[-1].timestamp - dd_start).days
            max_duration = max(max_duration, duration)

        return max_duration

    def _calculate_streaks(self) -> tuple[int, int]:
        """Calculate max win and lose streaks."""
        if not self.positions:
            return 0, 0

        max_win = 0
        max_lose = 0
        current_win = 0
        current_lose = 0

        for pos in self.positions:
            if pos.outcome == TradeOutcome.WIN:
                current_win += 1
                current_lose = 0
                max_win = max(max_win, current_win)
            elif pos.outcome == TradeOutcome.LOSS:
                current_lose += 1
                current_win = 0
                max_lose = max(max_lose, current_lose)

        return max_win, max_lose

    def _calculate_returns(self) -> list[float]:
        """Calculate per-trade returns."""
        returns = []
        for pos in self.positions:
            if pos.pnl and pos.size_usd > 0:
                ret = float(pos.pnl / pos.size_usd)
                returns.append(ret)
        return returns

    def _calculate_sharpe(self, returns: list[float]) -> Decimal:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return Decimal("0")

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = variance ** 0.5

        if std == 0:
            return Decimal("999") if mean_ret > 0 else Decimal("0")

        # Annualize assuming daily trades
        sharpe = (mean_ret / std) * (252 ** 0.5)
        return Decimal(str(round(sharpe, 2)))

    def _calculate_sortino(self, returns: list[float]) -> Decimal:
        """Calculate Sortino ratio (only penalizes downside vol)."""
        if len(returns) < 2:
            return Decimal("0")

        mean_ret = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return Decimal("999") if mean_ret > 0 else Decimal("0")

        downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = downside_var ** 0.5

        if downside_std == 0:
            return Decimal("999") if mean_ret > 0 else Decimal("0")

        sortino = (mean_ret / downside_std) * (252 ** 0.5)
        return Decimal(str(round(sortino, 2)))

    def _empty_stats(self) -> EquityCurveStats:
        """Return empty stats when no data."""
        return EquityCurveStats(
            total_pnl=Decimal("0"),
            peak_pnl=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_percent=Decimal("0"),
            max_drawdown_duration_days=0,
            current_drawdown=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            calmar_ratio=Decimal("0"),
            win_streak_max=0,
            lose_streak_max=0,
            recovery_factor=Decimal("0"),
        )

    def print_curve_ascii(self, width: int = 60, height: int = 15):
        """Print an ASCII representation of the equity curve."""
        if not self.curve:
            console.print("[yellow]No data for equity curve[/yellow]")
            return

        pnls = [float(p.cumulative_pnl) for p in self.curve]
        min_pnl = min(pnls)
        max_pnl = max(pnls)
        pnl_range = max_pnl - min_pnl or 1

        # Normalize to height
        normalized = [
            int((p - min_pnl) / pnl_range * (height - 1))
            for p in pnls
        ]

        # Sample to width if too many points
        if len(normalized) > width:
            step = len(normalized) / width
            sampled = [normalized[int(i * step)] for i in range(width)]
        else:
            sampled = normalized + [normalized[-1]] * (width - len(normalized))

        console.print("\n[bold cyan]Equity Curve[/bold cyan]")
        console.print(f"[dim]${max_pnl:,.0f}[/dim]")

        # Print rows from top to bottom
        for row in range(height - 1, -1, -1):
            line = ""
            for col in range(width):
                if sampled[col] >= row:
                    line += "█"
                else:
                    line += " "
            console.print(f"[green]{line}[/green]")

        console.print(f"[dim]${min_pnl:,.0f}[/dim]")
        console.print(f"[dim]{'─' * width}[/dim]")

    def print_stats(self, stats: Optional[EquityCurveStats] = None):
        """Print formatted statistics."""
        if stats is None:
            stats = self.get_stats()

        table = Table(title="Equity Curve Statistics")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total P&L", f"${stats.total_pnl:,.2f}")
        table.add_row("Peak P&L", f"${stats.peak_pnl:,.2f}")
        table.add_row("Max Drawdown", f"${stats.max_drawdown:,.2f}")
        table.add_row("Max Drawdown %", f"{stats.max_drawdown_percent:.1f}%")
        table.add_row("Max DD Duration", f"{stats.max_drawdown_duration_days} days")
        table.add_row("Current Drawdown", f"${stats.current_drawdown:,.2f}")
        table.add_row("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}")
        table.add_row("Sortino Ratio", f"{stats.sortino_ratio:.2f}")
        table.add_row("Calmar Ratio", f"{stats.calmar_ratio:.2f}")
        table.add_row("Max Win Streak", str(stats.win_streak_max))
        table.add_row("Max Lose Streak", str(stats.lose_streak_max))
        table.add_row("Recovery Factor", f"{stats.recovery_factor:.2f}")

        console.print(table)

    def describe_curve_quality(self) -> str:
        """
        Generate a qualitative description of the equity curve.

        A "straight line climbing" like gabagool22's indicates
        consistent edge and good risk management.
        """
        stats = self.get_stats()

        if not self.curve or len(self.curve) < 10:
            return "Insufficient data to assess curve quality."

        descriptions = []

        # Assess linearity (consistent growth)
        pnls = [float(p.cumulative_pnl) for p in self.curve]

        # Calculate R² to line
        n = len(pnls)
        x_mean = (n - 1) / 2
        y_mean = sum(pnls) / n

        ss_tot = sum((y - y_mean) ** 2 for y in pnls)
        slope = sum((i - x_mean) * (pnls[i] - y_mean) for i in range(n)) / sum((i - x_mean) ** 2 for i in range(n))
        intercept = y_mean - slope * x_mean
        ss_res = sum((pnls[i] - (slope * i + intercept)) ** 2 for i in range(n))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r_squared > 0.95:
            descriptions.append("Near-perfect linear growth. Exceptional consistency.")
        elif r_squared > 0.85:
            descriptions.append("Strong linear trend. Consistent edge being realized.")
        elif r_squared > 0.7:
            descriptions.append("Moderate consistency with some variance.")
        else:
            descriptions.append("High variance in returns. Choppy equity curve.")

        # Assess drawdowns
        if stats.max_drawdown_percent < 5:
            descriptions.append("Minimal drawdowns indicate tight risk control.")
        elif stats.max_drawdown_percent < 15:
            descriptions.append("Reasonable drawdowns within normal bounds.")
        else:
            descriptions.append("Significant drawdowns present. Higher risk profile.")

        # Assess recovery
        if stats.recovery_factor > 10:
            descriptions.append("Excellent recovery factor - profits far exceed max DD.")
        elif stats.recovery_factor > 3:
            descriptions.append("Good recovery factor - solid profit to risk ratio.")

        # Assess Sharpe
        if stats.sharpe_ratio > 3:
            descriptions.append("Outstanding risk-adjusted returns (Sharpe > 3).")
        elif stats.sharpe_ratio > 2:
            descriptions.append("Strong risk-adjusted returns (Sharpe > 2).")
        elif stats.sharpe_ratio > 1:
            descriptions.append("Acceptable risk-adjusted returns (Sharpe > 1).")

        return " ".join(descriptions)


def analyze_equity(positions: list[Position]) -> tuple[EquityCurve, EquityCurveStats]:
    """Convenience function to analyze equity from positions."""
    curve = EquityCurve(positions)
    stats = curve.get_stats()
    return curve, stats
