"""Configuration management for wallet tracker."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel


class PolymarketConfig(BaseModel):
    """Polymarket API configuration."""

    api_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws"
    gamma_url: str = "https://gamma-api.polymarket.com"
    api_key: Optional[str] = None
    secret: Optional[str] = None
    passphrase: Optional[str] = None


class SpotFeedConfig(BaseModel):
    """Spot price feed configuration."""

    binance_ws_url: str = "wss://stream.binance.com:9443/ws"
    coinbase_ws_url: str = "wss://ws-feed.exchange.coinbase.com"


class OpenBBConfig(BaseModel):
    """OpenBB Platform configuration."""

    # Default provider for crypto data (yfinance is free, no API key needed)
    default_provider: str = "yfinance"
    # API keys for premium providers (optional)
    fmp_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    tiingo_api_key: Optional[str] = None
    # Polling interval for OpenBB momentum monitor (seconds)
    poll_interval: int = 10
    # Enable OpenBB as data source
    enabled: bool = True


class TrackerConfig(BaseModel):
    """Wallet tracker configuration."""

    poll_interval_seconds: int = 30
    momentum_window_seconds: int = 15
    min_position_size: float = 100.0
    max_lag_seconds: float = 60.0


class Config(BaseModel):
    """Main configuration container."""

    polymarket: PolymarketConfig = PolymarketConfig()
    spot_feeds: SpotFeedConfig = SpotFeedConfig()
    openbb: OpenBBConfig = OpenBBConfig()
    tracker: TrackerConfig = TrackerConfig()
    data_dir: Path = Path("data")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        polymarket = PolymarketConfig(
            api_url=os.getenv("POLYMARKET_API_URL", "https://clob.polymarket.com"),
            ws_url=os.getenv("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws"),
            api_key=os.getenv("POLYMARKET_API_KEY"),
            secret=os.getenv("POLYMARKET_SECRET"),
            passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        )

        spot_feeds = SpotFeedConfig(
            binance_ws_url=os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws"),
            coinbase_ws_url=os.getenv("COINBASE_WS_URL", "wss://ws-feed.exchange.coinbase.com"),
        )

        openbb = OpenBBConfig(
            default_provider=os.getenv("OPENBB_PROVIDER", "yfinance"),
            fmp_api_key=os.getenv("FMP_API_KEY"),
            polygon_api_key=os.getenv("POLYGON_API_KEY"),
            tiingo_api_key=os.getenv("TIINGO_API_KEY"),
            poll_interval=int(os.getenv("OPENBB_POLL_INTERVAL", "10")),
            enabled=os.getenv("OPENBB_ENABLED", "true").lower() == "true",
        )

        tracker = TrackerConfig(
            poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "30")),
            momentum_window_seconds=int(os.getenv("MOMENTUM_WINDOW_SECONDS", "15")),
        )

        return cls(
            polymarket=polymarket,
            spot_feeds=spot_feeds,
            openbb=openbb,
            tracker=tracker,
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
