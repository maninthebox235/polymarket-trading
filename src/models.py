"""Data models for wallet tracking and strategy analysis."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MarketType(str, Enum):
    BTC = "btc"
    ETH = "eth"
    OTHER = "other"


class TradeDirection(str, Enum):
    UP = "up"
    DOWN = "down"


class TradeOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    PENDING = "pending"


class Position(BaseModel):
    """A single position/trade on Polymarket."""

    market_id: str
    market_slug: str
    market_type: MarketType
    direction: TradeDirection
    entry_price: Decimal = Field(ge=0, le=1)
    size_usd: Decimal
    shares: Decimal
    timestamp: datetime
    outcome: TradeOutcome = TradeOutcome.PENDING
    pnl: Optional[Decimal] = None
    resolution_time: Optional[datetime] = None

    @property
    def potential_payout(self) -> Decimal:
        """Maximum payout if position wins ($1 per share)."""
        return self.shares

    @property
    def roi(self) -> Optional[Decimal]:
        """Return on investment if resolved."""
        if self.pnl is None:
            return None
        return (self.pnl / self.size_usd) * 100


class WalletStats(BaseModel):
    """Aggregated statistics for a wallet."""

    address: str
    username: Optional[str] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    pending_trades: int = 0
    total_volume: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    average_position_size: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None

    @property
    def roi_percent(self) -> Decimal:
        """Overall ROI percentage."""
        if self.total_volume == 0:
            return Decimal("0")
        return (self.total_pnl / self.total_volume) * 100


class MomentumSignal(BaseModel):
    """A detected momentum signal from spot markets."""

    asset: str  # BTC or ETH
    direction: TradeDirection
    spot_price: Decimal
    price_change_percent: Decimal
    timestamp: datetime
    source: str  # binance, coinbase
    strength: Decimal = Field(ge=0, le=1)  # 0-1 confidence score

    @property
    def is_strong(self) -> bool:
        """Whether this is a high-confidence signal."""
        return self.strength >= Decimal("0.7")


class MarketLag(BaseModel):
    """Detected lag between spot price and Polymarket odds."""

    asset: str
    spot_direction: TradeDirection
    spot_change_percent: Decimal
    polymarket_odds: Decimal
    expected_odds: Decimal
    lag_seconds: float
    edge_estimate: Decimal
    timestamp: datetime


class EquityPoint(BaseModel):
    """A single point on the equity curve."""

    timestamp: datetime
    cumulative_pnl: Decimal
    trade_count: int
    drawdown: Decimal = Decimal("0")
    drawdown_percent: Decimal = Decimal("0")
