// Chart functionality for the trading bot
class TradingCharts {
    constructor() {
        this.charts = {};
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Price Chart'
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        };
    }

    async loadChart(symbol, containerId = 'priceChart') {
        try {
            const chartContainer = document.getElementById(containerId);
            if (!chartContainer) {
                console.error(`Chart container ${containerId} not found`);
                return;
            }

            // Show loading
            this.showChartLoading(chartContainer);

            // Get chart data
            const response = await fetch(`/api/chart-data/${symbol}?interval=1h&limit=100`);
            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Get indicators
            const indicatorsResponse = await fetch(`/api/indicators/${symbol}`);
            const indicators = await indicatorsResponse.json();

            // Create chart
            this.createPriceChart(containerId, symbol, data.data, indicators);

            // Load trading signals
            this.loadTradingSignals(symbol, indicators);

        } catch (error) {
            console.error('Error loading chart:', error);
            this.showChartError(containerId, error.message);
        }
    }

    createPriceChart(containerId, symbol, data, indicators) {
        const ctx = document.getElementById(containerId).getContext('2d');
        
        // Destroy existing chart
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        // Prepare data
        const labels = data.map(d => new Date(d.timestamp));
        const prices = data.map(d => d.close);
        const volumes = data.map(d => d.volume);

        // Create datasets
        const datasets = [
            {
                label: 'Price',
                data: prices,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                yAxisID: 'y'
            },
            {
                label: 'Volume',
                data: volumes,
                type: 'bar',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                yAxisID: 'y1'
            }
        ];

        // Add indicators if available
        if (indicators.indicators) {
            // SMA 20
            if (indicators.indicators.sma_20) {
                datasets.push({
                    label: 'SMA 20',
                    data: new Array(prices.length).fill(indicators.indicators.sma_20),
                    borderColor: 'rgb(255, 159, 64)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
            }

            // SMA 50
            if (indicators.indicators.sma_50) {
                datasets.push({
                    label: 'SMA 50',
                    data: new Array(prices.length).fill(indicators.indicators.sma_50),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
            }

            // Bollinger Bands
            if (indicators.indicators.bb_high && indicators.indicators.bb_low) {
                datasets.push({
                    label: 'BB Upper',
                    data: new Array(prices.length).fill(indicators.indicators.bb_high),
                    borderColor: 'rgba(54, 162, 235, 0.8)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
                datasets.push({
                    label: 'BB Lower',
                    data: new Array(prices.length).fill(indicators.indicators.bb_low),
                    borderColor: 'rgba(54, 162, 235, 0.8)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
            }
        }

        // Create chart
        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                ...this.defaultOptions,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: `${symbol} Price Chart`
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Price'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Volume'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    createCandlestickChart(containerId, symbol, data) {
        // This would require a candlestick chart library like Chart.js candlestick plugin
        // For now, we'll use a simplified line chart
        this.createPriceChart(containerId, symbol, data, {});
    }

    createIndicatorChart(containerId, symbol, indicator, data) {
        const ctx = document.getElementById(containerId).getContext('2d');
        
        // Destroy existing chart
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        const labels = data.map(d => new Date(d.timestamp));
        let datasets = [];
        let yAxisConfig = {};

        switch (indicator) {
            case 'rsi':
                datasets = [{
                    label: 'RSI',
                    data: data.map(d => d.rsi),
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    tension: 0.1
                }];
                yAxisConfig = {
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                };
                break;
            
            case 'macd':
                datasets = [
                    {
                        label: 'MACD',
                        data: data.map(d => d.macd),
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'transparent',
                        tension: 0.1
                    },
                    {
                        label: 'Signal',
                        data: data.map(d => d.macd_signal),
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'transparent',
                        tension: 0.1
                    }
                ];
                break;
            
            default:
                datasets = [{
                    label: indicator.toUpperCase(),
                    data: data.map(d => d[indicator]),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }];
        }

        this.charts[containerId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                ...this.defaultOptions,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: `${symbol} ${indicator.toUpperCase()}`
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour'
                        }
                    },
                    y: yAxisConfig
                }
            }
        });
    }

    async loadTradingSignals(symbol, indicators) {
        const signalsContainer = document.getElementById('tradingSignals');
        if (!signalsContainer) return;

        try {
            // Get prediction
            const predictionResponse = await fetch(`/api/prediction/${symbol}`);
            const prediction = await predictionResponse.json();

            let signalsHtml = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Trading Signals</h6>
                    </div>
                    <div class="card-body">
            `;

            // Technical signals
            if (indicators.signals) {
                signalsHtml += `
                    <div class="mb-3">
                        <h6 class="text-muted">Technical Analysis</h6>
                        <div class="mb-2">
                            <span class="badge bg-${this.getSignalColor(indicators.signals.overall_signal)}">
                                ${indicators.signals.overall_signal}
                            </span>
                            <small class="text-muted ms-2">
                                Confidence: ${(indicators.signals.overall_confidence * 100).toFixed(1)}%
                            </small>
                        </div>
                `;

                // Individual signals
                const signalTypes = ['rsi_signal', 'macd_signal', 'bb_signal', 'ma_signal'];
                signalTypes.forEach(type => {
                    if (indicators.signals[type]) {
                        signalsHtml += `
                            <div class="d-flex justify-content-between mb-1">
                                <span>${type.replace('_signal', '').toUpperCase()}</span>
                                <span class="badge bg-${this.getSignalColor(indicators.signals[type])}">
                                    ${indicators.signals[type]}
                                </span>
                            </div>
                        `;
                    }
                });

                signalsHtml += '</div>';
            }

            // ML prediction
            if (prediction && !prediction.error) {
                signalsHtml += `
                    <div class="mb-3">
                        <h6 class="text-muted">ML Prediction</h6>
                        <div class="mb-2">
                            <span class="badge bg-${this.getSignalColor(prediction.direction)}">
                                ${prediction.direction}
                            </span>
                            <small class="text-muted ms-2">
                                Confidence: ${(prediction.confidence * 100).toFixed(1)}%
                            </small>
                        </div>
                        <div class="small text-muted">
                            Predicted Price: ${prediction.predicted_price.toFixed(4)}<br>
                            Change: ${prediction.predicted_change_percent.toFixed(2)}%
                        </div>
                    </div>
                `;
            }

            // Support/Resistance
            if (indicators.support_resistance) {
                const sr = indicators.support_resistance;
                signalsHtml += `
                    <div class="mb-3">
                        <h6 class="text-muted">Support/Resistance</h6>
                        <div class="small">
                            <div class="d-flex justify-content-between">
                                <span>Resistance:</span>
                                <span class="text-danger">${sr.nearest_resistance ? sr.nearest_resistance.toFixed(4) : 'N/A'}</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>Current:</span>
                                <span>${sr.current_price.toFixed(4)}</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>Support:</span>
                                <span class="text-success">${sr.nearest_support ? sr.nearest_support.toFixed(4) : 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            signalsHtml += '</div></div>';
            signalsContainer.innerHTML = signalsHtml;

        } catch (error) {
            console.error('Error loading trading signals:', error);
            signalsContainer.innerHTML = `
                <div class="alert alert-danger">
                    Error loading signals: ${error.message}
                </div>
            `;
        }
    }

    getSignalColor(signal) {
        switch (signal) {
            case 'BUY':
            case 'UP':
                return 'success';
            case 'SELL':
            case 'DOWN':
                return 'danger';
            case 'HOLD':
                return 'secondary';
            default:
                return 'secondary';
        }
    }

    showChartLoading(container) {
        container.innerHTML = `
            <div class="d-flex justify-content-center align-items-center" style="height: 400px;">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }

    showChartError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Error loading chart: ${message}
                </div>
            `;
        }
    }

    destroyChart(containerId) {
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
            delete this.charts[containerId];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(chartId => {
            this.destroyChart(chartId);
        });
    }
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradingCharts = new TradingCharts();
});

// Global function for loading charts
function loadChart(symbol, containerId = 'priceChart') {
    if (window.tradingCharts) {
        window.tradingCharts.loadChart(symbol, containerId);
    }
}

// Export for other modules
window.TradingCharts = TradingCharts;
