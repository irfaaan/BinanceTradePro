// Dashboard functionality
class TradingDashboard {
    constructor() {
        this.refreshInterval = 30000; // 30 seconds
        this.priceUpdateInterval = null;
        this.init();
    }

    init() {
        this.startPriceUpdates();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Navigation highlighting
        const currentPath = window.location.pathname;
        document.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });

        // Auto-refresh toggle
        document.addEventListener('keydown', (e) => {
            if (e.key === 'r' && e.ctrlKey) {
                e.preventDefault();
                this.refreshData();
            }
        });
    }

    startPriceUpdates() {
        this.updatePrices();
        this.priceUpdateInterval = setInterval(() => {
            this.updatePrices();
        }, this.refreshInterval);
    }

    stopPriceUpdates() {
        if (this.priceUpdateInterval) {
            clearInterval(this.priceUpdateInterval);
            this.priceUpdateInterval = null;
        }
    }

    async updatePrices() {
        const priceElements = document.querySelectorAll('[data-symbol]');
        
        for (const element of priceElements) {
            const symbol = element.getAttribute('data-symbol');
            try {
                const response = await fetch(`/api/market-data/${symbol}`);
                const data = await response.json();
                
                if (data.price) {
                    this.updatePriceElement(element, data);
                }
            } catch (error) {
                console.error(`Error updating price for ${symbol}:`, error);
            }
        }
    }

    updatePriceElement(element, data) {
        const priceElement = element.querySelector('.current-price');
        const changeElement = element.querySelector('.price-change');
        
        if (priceElement) {
            const newPrice = parseFloat(data.price);
            const oldPrice = parseFloat(priceElement.textContent) || 0;
            
            priceElement.textContent = newPrice.toFixed(4);
            
            // Add price change animation
            if (newPrice > oldPrice) {
                priceElement.classList.add('price-up');
                setTimeout(() => priceElement.classList.remove('price-up'), 1000);
            } else if (newPrice < oldPrice) {
                priceElement.classList.add('price-down');
                setTimeout(() => priceElement.classList.remove('price-down'), 1000);
            }
        }
        
        if (changeElement && data.change_24h !== undefined) {
            const change = parseFloat(data.change_24h);
            changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeElement.className = `price-change ${change >= 0 ? 'text-success' : 'text-danger'}`;
        }
    }

    async refreshData() {
        // Show loading indicator
        const loadingElements = document.querySelectorAll('.loading-placeholder');
        loadingElements.forEach(el => el.style.display = 'block');
        
        try {
            await this.updatePrices();
            await this.updatePortfolio();
            await this.updateTrades();
        } catch (error) {
            console.error('Error refreshing data:', error);
        } finally {
            // Hide loading indicator
            loadingElements.forEach(el => el.style.display = 'none');
        }
    }

    async updatePortfolio() {
        try {
            const response = await fetch('/api/portfolio');
            const data = await response.json();
            
            if (data.total_value !== undefined) {
                const totalValueElement = document.querySelector('.portfolio-total-value');
                if (totalValueElement) {
                    totalValueElement.textContent = `${data.total_value.toFixed(2)} USDT`;
                }
            }
            
            if (data.pnl !== undefined) {
                const pnlElement = document.querySelector('.portfolio-pnl');
                if (pnlElement) {
                    pnlElement.textContent = `${data.pnl.toFixed(2)} USDT (${data.pnl_percent.toFixed(2)}%)`;
                    pnlElement.className = `portfolio-pnl ${data.pnl >= 0 ? 'text-success' : 'text-danger'}`;
                }
            }
        } catch (error) {
            console.error('Error updating portfolio:', error);
        }
    }

    async updateTrades() {
        try {
            const response = await fetch('/api/trades?limit=5');
            const trades = await response.json();
            
            const tradesContainer = document.querySelector('.recent-trades-container');
            if (tradesContainer && trades.length > 0) {
                this.renderRecentTrades(tradesContainer, trades);
            }
        } catch (error) {
            console.error('Error updating trades:', error);
        }
    }

    renderRecentTrades(container, trades) {
        const tableBody = container.querySelector('tbody');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        trades.forEach(trade => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${trade.symbol}</td>
                <td>
                    <span class="badge bg-${trade.side === 'BUY' ? 'success' : 'danger'}">
                        ${trade.side}
                    </span>
                </td>
                <td>${parseFloat(trade.price).toFixed(4)}</td>
                <td>
                    <small>${new Date(trade.timestamp).toLocaleTimeString()}</small>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    formatCurrency(amount, currency = 'USDT') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency === 'USDT' ? 'USD' : currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 4
        }).format(amount);
    }

    formatPercentage(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value / 100);
    }

    async executeManualTrade(symbol, side, quantity, price) {
        try {
            const response = await fetch('/manual-trade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol: symbol,
                    side: side,
                    quantity: quantity,
                    price: price
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(`${side} order executed successfully for ${symbol}`, 'success');
                this.refreshData();
                return true;
            } else {
                this.showNotification(`Trade failed: ${data.error}`, 'danger');
                return false;
            }
        } catch (error) {
            this.showNotification(`Error executing trade: ${error.message}`, 'danger');
            return false;
        }
    }

    async getSymbolAnalysis(symbol) {
        try {
            const response = await fetch(`/api/indicators/${symbol}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            return data;
        } catch (error) {
            console.error(`Error getting analysis for ${symbol}:`, error);
            return null;
        }
    }

    async getSymbolPrediction(symbol) {
        try {
            const response = await fetch(`/api/prediction/${symbol}`);
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            return data;
        } catch (error) {
            console.error(`Error getting prediction for ${symbol}:`, error);
            return null;
        }
    }

    destroy() {
        this.stopPriceUpdates();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});

// Utility functions
function showLoadingSpinner(element) {
    const spinner = document.createElement('div');
    spinner.className = 'text-center loading-spinner';
    spinner.innerHTML = `
        <div class="spinner-border spinner-border-sm" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    element.appendChild(spinner);
}

function hideLoadingSpinner(element) {
    const spinner = element.querySelector('.loading-spinner');
    if (spinner) {
        spinner.remove();
    }
}

// Global functions for template usage
function viewChart(symbol) {
    if (window.dashboard) {
        // This will be handled by charts.js
        console.log(`Loading chart for ${symbol}`);
    }
}

function quickTrade(symbol, side) {
    if (window.dashboard) {
        // This will be handled by the trade modal
        console.log(`Quick trade: ${side} ${symbol}`);
    }
}

// Export for other modules
window.TradingDashboard = TradingDashboard;
