"""
FastAPI application for Polymarket Wallet Tracker UI.

Run with: uvicorn web.app:app --reload
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wallet_tracker import WalletTracker
from src.strategy_analyzer import StrategyAnalyzer, compare_to_gabagool22
from src.equity_curve import EquityCurve
from src.momentum_detector import MomentumDetector
from src.models import TradeOutcome

# App setup
app = FastAPI(
    title="Polymarket Wallet Tracker",
    description="Track whale wallets and analyze momentum strategies",
    version="0.1.0"
)

# Static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# Request/Response models
class WalletRequest(BaseModel):
    address: str


class WalletAnalysis(BaseModel):
    address: str
    username: Optional[str]
    total_trades: int
    winning_trades: int
    losing_trades: int
    pending_trades: int
    win_rate: float
    total_volume: float
    total_pnl: float
    average_position_size: float
    roi_percent: float
    first_trade: Optional[str]
    last_trade: Optional[str]
    strategy_type: str
    sizing_consistency: float
    preferred_window: str
    equity_curve: list[dict]
    curve_quality: str
    gabagool_similarity: Optional[dict]


class CompareRequest(BaseModel):
    addresses: list[str]


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze", response_model=WalletAnalysis)
async def analyze_wallet(req: WalletRequest):
    """Analyze a Polymarket wallet."""
    try:
        async with WalletTracker(req.address) as tracker:
            stats = await tracker.analyze_wallet()

            if stats.total_trades == 0:
                raise HTTPException(status_code=404, detail="No trades found for this wallet")

            # Strategy analysis
            analyzer = StrategyAnalyzer(tracker.positions, stats)
            profile = analyzer.analyze()

            # Equity curve
            curve = EquityCurve(tracker.positions)
            curve_stats = curve.get_stats()
            curve_quality = curve.describe_curve_quality()

            # Build equity curve data for chart
            equity_data = [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "pnl": float(p.cumulative_pnl),
                    "drawdown": float(p.drawdown),
                    "trade_count": p.trade_count,
                }
                for p in curve.curve
            ]

            # Compare to gabagool22
            similarity = compare_to_gabagool22(profile)

            return WalletAnalysis(
                address=stats.address,
                username=stats.username,
                total_trades=stats.total_trades,
                winning_trades=stats.winning_trades,
                losing_trades=stats.losing_trades,
                pending_trades=stats.pending_trades,
                win_rate=float(stats.win_rate),
                total_volume=float(stats.total_volume),
                total_pnl=float(stats.total_pnl),
                average_position_size=float(stats.average_position_size),
                roi_percent=float(stats.roi_percent),
                first_trade=stats.first_trade.isoformat() if stats.first_trade else None,
                last_trade=stats.last_trade.isoformat() if stats.last_trade else None,
                strategy_type=profile.strategy_type,
                sizing_consistency=float(profile.sizing_consistency),
                preferred_window=profile.preferred_window,
                equity_curve=equity_data,
                curve_quality=curve_quality,
                gabagool_similarity=similarity,
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare_wallets(req: CompareRequest):
    """Compare multiple wallets."""
    if len(req.addresses) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 addresses")

    results = []
    for addr in req.addresses[:5]:  # Limit to 5 wallets
        try:
            async with WalletTracker(addr) as tracker:
                stats = await tracker.analyze_wallet()
                analyzer = StrategyAnalyzer(tracker.positions, stats)
                profile = analyzer.analyze()

                results.append({
                    "address": stats.address,
                    "username": stats.username,
                    "total_pnl": float(stats.total_pnl),
                    "win_rate": float(stats.win_rate),
                    "total_trades": stats.total_trades,
                    "avg_position_size": float(stats.average_position_size),
                    "strategy_type": profile.strategy_type,
                    "sizing_consistency": float(profile.sizing_consistency),
                })
        except Exception as e:
            results.append({
                "address": addr,
                "error": str(e),
            })

    return {"wallets": results}


@app.get("/api/positions/{address}")
async def get_positions(address: str, market_type: Optional[str] = None):
    """Get positions for a wallet."""
    async with WalletTracker(address) as tracker:
        await tracker.analyze_wallet()

        positions = tracker.positions
        if market_type:
            positions = [p for p in positions if p.market_type.value == market_type]

        return {
            "address": address,
            "count": len(positions),
            "positions": [
                {
                    "market_slug": p.market_slug,
                    "market_type": p.market_type.value,
                    "direction": p.direction.value,
                    "entry_price": float(p.entry_price),
                    "size_usd": float(p.size_usd),
                    "shares": float(p.shares),
                    "outcome": p.outcome.value,
                    "pnl": float(p.pnl) if p.pnl else None,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in positions
            ]
        }


# WebSocket for live momentum monitoring
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()
momentum_detector: Optional[MomentumDetector] = None
detector_task: Optional[asyncio.Task] = None


@app.websocket("/ws/momentum")
async def momentum_websocket(websocket: WebSocket):
    """WebSocket endpoint for live momentum signals."""
    global momentum_detector, detector_task

    await manager.connect(websocket)

    # Start detector if not running
    if momentum_detector is None:
        momentum_detector = MomentumDetector(threshold=Decimal("0.15"))

        def on_signal(signal):
            asyncio.create_task(manager.broadcast({
                "type": "signal",
                "asset": signal.asset,
                "direction": signal.direction.value,
                "price_change": float(signal.price_change_percent),
                "spot_price": float(signal.spot_price),
                "source": signal.source,
                "strength": float(signal.strength),
                "timestamp": signal.timestamp.isoformat(),
            }))

        momentum_detector.on_signal(on_signal)
        detector_task = asyncio.create_task(momentum_detector.start())

    try:
        while True:
            # Send periodic price updates
            prices = momentum_detector.get_current_prices()
            await websocket.send_json({
                "type": "prices",
                "data": {k: float(v) if v else None for k, v in prices.items()},
                "timestamp": datetime.now().isoformat(),
            })
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

        # Stop detector if no more connections
        if len(manager.active_connections) == 0 and momentum_detector:
            momentum_detector.stop()
            if detector_task:
                detector_task.cancel()
            momentum_detector = None
            detector_task = None


@app.get("/api/gabagool")
async def gabagool_info():
    """Get information about the gabagool22 strategy."""
    return {
        "name": "gabagool22",
        "profile_url": "https://polymarket.com/@gabagool22",
        "total_pnl": 457000,
        "strategy": {
            "type": "momentum_reading",
            "description": "Exploits lag between spot price moves and Polymarket odds adjustment",
            "markets": ["BTC 15-minute windows", "ETH 15-minute windows"],
            "avg_position_size": 1600,
            "entry_range": "40-50 cents",
            "resolution_payout": "1 dollar per share",
        },
        "key_insights": [
            "Small consistent sizing ($1,600 avg) vs whale sizing ($10K-$40K)",
            "More entries = more chances for edge to play out",
            "Less damage when something misses",
            "Straight line equity curve for two months",
            "No hero moments, just consistent profits",
        ],
        "edge_source": "When BTC/ETH moves hard on Binance/Coinbase, Polymarket 15-min window odds lag by seconds",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
