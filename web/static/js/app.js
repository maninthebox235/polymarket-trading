/**
 * Polymarket Wallet Tracker - Frontend Application
 */

// State
const state = {
    currentView: 'analyze',
    equityChart: null,
    momentumWs: null,
    isMonitoring: false,
};

// DOM Elements
const elements = {
    navItems: document.querySelectorAll('.nav-item'),
    views: document.querySelectorAll('.view'),
    walletInput: document.getElementById('wallet-input'),
    analyzeBtn: document.getElementById('analyze-btn'),
    analysisLoading: document.getElementById('analysis-loading'),
    analysisError: document.getElementById('analysis-error'),
    analysisResults: document.getElementById('analysis-results'),
    compareBtn: document.getElementById('compare-btn'),
    compareInputs: document.querySelectorAll('.compare-input'),
    compareLoading: document.getElementById('compare-loading'),
    compareResults: document.getElementById('compare-results'),
    startMomentumBtn: document.getElementById('start-momentum'),
    stopMomentumBtn: document.getElementById('stop-momentum'),
    btcPrice: document.getElementById('btc-price'),
    ethPrice: document.getElementById('eth-price'),
    signalsList: document.getElementById('signals-list'),
    connectionStatus: document.getElementById('connection-status'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupAnalyze();
    setupCompare();
    setupMomentum();
});

// Navigation
function setupNavigation() {
    elements.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            switchView(view);
        });
    });
}

function switchView(viewName) {
    // Update nav
    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });

    // Update views
    elements.views.forEach(view => {
        view.classList.toggle('active', view.id === `${viewName}-view`);
        view.classList.toggle('hidden', view.id !== `${viewName}-view`);
    });

    state.currentView = viewName;
}

// Analyze Wallet
function setupAnalyze() {
    elements.analyzeBtn.addEventListener('click', analyzeWallet);
    elements.walletInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeWallet();
    });
}

async function analyzeWallet() {
    const address = elements.walletInput.value.trim();
    if (!address) return;

    // Show loading
    elements.analysisLoading.classList.remove('hidden');
    elements.analysisError.classList.add('hidden');
    elements.analysisResults.classList.add('hidden');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze wallet');
        }

        const data = await response.json();
        displayAnalysis(data);

    } catch (error) {
        elements.analysisError.textContent = error.message;
        elements.analysisError.classList.remove('hidden');
    } finally {
        elements.analysisLoading.classList.add('hidden');
    }
}

function displayAnalysis(data) {
    // Stats
    const pnlEl = document.getElementById('stat-pnl');
    pnlEl.textContent = formatCurrency(data.total_pnl);
    pnlEl.className = `stat-value ${data.total_pnl >= 0 ? 'positive' : 'negative'}`;

    document.getElementById('stat-winrate').textContent = `${data.win_rate.toFixed(1)}%`;
    document.getElementById('stat-trades').textContent = data.total_trades.toLocaleString();
    document.getElementById('stat-avgsize').textContent = formatCurrency(data.average_position_size);
    document.getElementById('stat-roi').textContent = `${data.roi_percent.toFixed(1)}%`;
    document.getElementById('stat-strategy').textContent = formatStrategy(data.strategy_type);

    // Breakdown
    document.getElementById('breakdown-wins').textContent = data.winning_trades;
    document.getElementById('breakdown-losses').textContent = data.losing_trades;
    document.getElementById('breakdown-pending').textContent = data.pending_trades;

    // Equity Curve
    renderEquityCurve(data.equity_curve);
    document.getElementById('curve-quality').textContent = data.curve_quality;

    // Similarity
    renderSimilarity(data.gabagool_similarity);

    elements.analysisResults.classList.remove('hidden');
}

function renderEquityCurve(curveData) {
    const ctx = document.getElementById('equity-chart').getContext('2d');

    if (state.equityChart) {
        state.equityChart.destroy();
    }

    const labels = curveData.map(p => new Date(p.timestamp).toLocaleDateString());
    const pnlData = curveData.map(p => p.pnl);

    state.equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Cumulative P&L',
                data: pnlData,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: (ctx) => `P&L: ${formatCurrency(ctx.parsed.y)}`,
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    ticks: {
                        color: '#606070',
                        maxTicksLimit: 8,
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                    },
                    ticks: {
                        color: '#606070',
                        callback: (value) => formatCurrency(value),
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false,
            }
        }
    });
}

function renderSimilarity(similarity) {
    const container = document.getElementById('similarity-bars');
    container.innerHTML = '';

    const metrics = [
        { key: 'position_sizing', label: 'Position Sizing' },
        { key: 'consistency', label: 'Consistency' },
        { key: 'frequency', label: 'Trade Frequency' },
        { key: 'strategy_match', label: 'Strategy Match' },
        { key: 'overall', label: 'Overall Similarity' },
    ];

    metrics.forEach(({ key, label }) => {
        const value = similarity[key] || 0;
        const percent = Math.min(value * 100, 100);

        const bar = document.createElement('div');
        bar.className = 'similarity-bar';
        bar.innerHTML = `
            <div class="label">
                <span>${label}</span>
                <span>${percent.toFixed(0)}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: ${percent}%"></div>
            </div>
        `;
        container.appendChild(bar);
    });
}

// Compare Wallets
function setupCompare() {
    elements.compareBtn.addEventListener('click', compareWallets);
}

async function compareWallets() {
    const addresses = Array.from(elements.compareInputs)
        .map(input => input.value.trim())
        .filter(Boolean);

    if (addresses.length < 2) {
        alert('Please enter at least 2 wallet addresses');
        return;
    }

    elements.compareLoading.classList.remove('hidden');
    elements.compareResults.classList.add('hidden');

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ addresses }),
        });

        const data = await response.json();
        displayComparison(data.wallets);

    } catch (error) {
        alert('Failed to compare wallets: ' + error.message);
    } finally {
        elements.compareLoading.classList.add('hidden');
    }
}

function displayComparison(wallets) {
    const validWallets = wallets.filter(w => !w.error);

    if (validWallets.length === 0) {
        alert('No valid wallets found');
        return;
    }

    // Build table headers
    const thead = document.querySelector('.compare-table thead tr');
    thead.innerHTML = '<th>Metric</th>';
    validWallets.forEach(w => {
        thead.innerHTML += `<th>${w.username || w.address.slice(0, 10)}...</th>`;
    });

    // Build table body
    const tbody = document.getElementById('compare-tbody');
    tbody.innerHTML = '';

    const metrics = [
        { key: 'total_pnl', label: 'Total P&L', format: formatCurrency },
        { key: 'win_rate', label: 'Win Rate', format: v => `${v.toFixed(1)}%` },
        { key: 'total_trades', label: 'Total Trades', format: v => v.toLocaleString() },
        { key: 'avg_position_size', label: 'Avg Position', format: formatCurrency },
        { key: 'strategy_type', label: 'Strategy', format: formatStrategy },
        { key: 'sizing_consistency', label: 'Consistency', format: v => `${(v * 100).toFixed(0)}%` },
    ];

    metrics.forEach(({ key, label, format }) => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${label}</td>`;

        validWallets.forEach(w => {
            const value = w[key];
            row.innerHTML += `<td>${format(value)}</td>`;
        });

        tbody.appendChild(row);
    });

    elements.compareResults.classList.remove('hidden');
}

// Momentum Monitor
function setupMomentum() {
    elements.startMomentumBtn.addEventListener('click', startMomentum);
    elements.stopMomentumBtn.addEventListener('click', stopMomentum);
}

function startMomentum() {
    if (state.momentumWs) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/momentum`;

    state.momentumWs = new WebSocket(wsUrl);

    state.momentumWs.onopen = () => {
        state.isMonitoring = true;
        elements.startMomentumBtn.classList.add('hidden');
        elements.stopMomentumBtn.classList.remove('hidden');
        elements.connectionStatus.textContent = 'Monitoring...';
        elements.signalsList.innerHTML = '<div class="no-signals">Waiting for momentum signals...</div>';
    };

    state.momentumWs.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'prices') {
            updatePrices(data.data);
        } else if (data.type === 'signal') {
            addSignal(data);
        }
    };

    state.momentumWs.onclose = () => {
        state.isMonitoring = false;
        state.momentumWs = null;
        elements.startMomentumBtn.classList.remove('hidden');
        elements.stopMomentumBtn.classList.add('hidden');
        elements.connectionStatus.textContent = 'Disconnected';
    };

    state.momentumWs.onerror = () => {
        console.error('WebSocket error');
        stopMomentum();
    };
}

function stopMomentum() {
    if (state.momentumWs) {
        state.momentumWs.close();
        state.momentumWs = null;
    }
    state.isMonitoring = false;
    elements.startMomentumBtn.classList.remove('hidden');
    elements.stopMomentumBtn.classList.add('hidden');
    elements.connectionStatus.textContent = 'Ready';
}

function updatePrices(prices) {
    if (prices.binance_btc) {
        elements.btcPrice.textContent = formatCurrency(prices.binance_btc);
    }
    if (prices.binance_eth) {
        elements.ethPrice.textContent = formatCurrency(prices.binance_eth);
    }
}

function addSignal(signal) {
    // Remove "no signals" message if present
    const noSignals = elements.signalsList.querySelector('.no-signals');
    if (noSignals) {
        noSignals.remove();
    }

    const signalEl = document.createElement('div');
    signalEl.className = `signal-item ${signal.direction}`;

    const isStrong = signal.strength >= 0.7;
    const time = new Date(signal.timestamp).toLocaleTimeString();

    signalEl.innerHTML = `
        <span class="signal-direction">${signal.direction === 'up' ? '🟢' : '🔴'}</span>
        <div class="signal-info">
            <span class="signal-asset">${signal.asset}</span>
            <span class="signal-change ${signal.direction === 'up' ? 'positive' : 'negative'}">
                ${signal.direction === 'up' ? '+' : '-'}${signal.price_change.toFixed(2)}%
            </span>
            <span class="signal-meta">${signal.source} • ${time}</span>
        </div>
        <span class="signal-strength ${isStrong ? 'strong' : 'weak'}">
            ${isStrong ? 'STRONG' : 'WEAK'}
        </span>
    `;

    // Add to top of list
    elements.signalsList.insertBefore(signalEl, elements.signalsList.firstChild);

    // Limit to 50 signals
    while (elements.signalsList.children.length > 50) {
        elements.signalsList.removeChild(elements.signalsList.lastChild);
    }
}

// Utilities
function formatCurrency(value) {
    if (value === null || value === undefined) return '--';
    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    if (absValue >= 1000000) {
        return `${sign}$${(absValue / 1000000).toFixed(2)}M`;
    } else if (absValue >= 1000) {
        return `${sign}$${(absValue / 1000).toFixed(1)}K`;
    } else {
        return `${sign}$${absValue.toFixed(2)}`;
    }
}

function formatStrategy(strategy) {
    if (!strategy) return '-';
    return strategy
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}
