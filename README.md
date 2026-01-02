# Polymarket Wallet Tracker

Track whale wallets, analyze momentum strategies, and detect edge in crypto prediction markets.

Inspired by the gabagool22 strategy - the quietest $457K winner on Polymarket.

## The gabagool22 Strategy

```
The quietest automated wallet on Polymarket made $457K and nobody noticed.
```

**Momentum Reading** - When BTC or ETH starts moving hard on Binance or Coinbase, Polymarket odds lag behind by a few seconds. The 15-minute window still shows old prices while the real move already happened.

- Watch spot price feeds for sudden moves
- Open Polymarket before odds adjust
- Entry around 40-50¢, resolution pays $1
- Small edge, but real

**Position Sizing** - While others trade $10K-$40K positions, gabagool22 averages $1,600. More entries, more data, more chances for the edge to play out.

## Installation

```bash
# Clone the repository
git clone https://github.com/maninthebox235/polymarket-trading.git
cd polymarket-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## Usage

### Web UI

Launch the web interface for a visual dashboard:

```bash
# Start the web server
uvicorn web.app:app --reload

# Open in browser
# http://localhost:8000
```

Features:
- **Analyze Wallet** - Enter any wallet address or username to see stats, equity curve, and strategy breakdown
- **Compare Wallets** - Side-by-side comparison of up to 5 wallets
- **Live Momentum** - Real-time BTC/ETH price feeds with momentum signal detection
- **gabagool22** - Deep dive into the $457K strategy

### CLI: Analyze a Wallet

```bash
# Basic analysis
python main.py analyze gabagool22

# Full analysis with strategy breakdown and equity curve
python main.py analyze gabagool22 --full

# Compare to gabagool22 benchmark
python main.py analyze <wallet-address> --compare-gabagool
```

### Compare Wallets

```bash
python main.py compare gabagool22 Account88888 whale_wallet_123
```

### Monitor Momentum

```bash
# Watch for spot price moves that create PM edge
python main.py monitor

# Adjust threshold (default 0.15%)
python main.py monitor --threshold 0.2
```

### Export Data

```bash
# Export wallet positions to JSON
python main.py export gabagool22 -o gabagool22_positions.json
```

### Learn About gabagool22

```bash
python main.py gabagool
```

## Project Structure

```
polymarket-trading/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models (Position, WalletStats, etc.)
│   ├── wallet_tracker.py  # Core wallet tracking
│   ├── momentum_detector.py # Spot price momentum detection
│   ├── strategy_analyzer.py # Strategy classification
│   └── equity_curve.py    # Performance tracking
└── web/
    ├── app.py             # FastAPI application
    ├── templates/
    │   └── index.html     # Main dashboard template
    └── static/
        ├── css/style.css  # Dark theme styles
        └── js/app.js      # Frontend JavaScript
```

## Key Features

### Wallet Tracking
- Fetch positions and trades from Polymarket API
- Calculate win rates, P&L, and position sizing metrics
- Classify markets (BTC, ETH, other)
- Track 15-minute window activity

### Momentum Detection
- Real-time Binance and Coinbase websocket feeds
- Detect significant price moves within 15-second windows
- Calculate signal strength and direction
- Identify potential PM lag opportunities

### Strategy Analysis
- Classify strategies: momentum_reading, arbitrage, whale_accumulation
- Compare wallets to gabagool22 benchmark
- Analyze position sizing consistency
- Detect preferred market windows

### Equity Curve Analysis
- Build cumulative P&L curves
- Calculate drawdowns and recovery metrics
- Compute Sharpe, Sortino, and Calmar ratios
- ASCII visualization of equity curves

## API Endpoints Used

- `https://gamma-api.polymarket.com` - Positions and market data
- `https://clob.polymarket.com` - Order book and trades
- `wss://stream.binance.com` - BTC/ETH spot prices
- `wss://ws-feed.exchange.coinbase.com` - BTC/ETH spot prices

## Why This Works

> Other wallets size big. $10K, $20K, $40K per trade. They need fewer wins to make money. But one bad streak and the curve dips hard.
>
> gabagool22 went the opposite way. $1,600 average. Tiny. But he never stops. While someone else loads one big position, this wallet already fired five trades across five different windows.
>
> More entries. More data. More chances for the edge to play out. Less damage when something misses.

The equity curve tells the story - a straight line climbing for two months. No hero moments. Just consistent edge realization.

## License

MIT
