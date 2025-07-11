# Advanced Binance Trading Bot

## Overview

This is a sophisticated cryptocurrency trading bot built with Flask that provides **live trading capabilities** with real money, advanced technical analysis, machine learning predictions, and comprehensive risk management. The bot is designed for professional-grade trading with multi-pair support, advanced strategy engine, and real-time portfolio management.

## Recent Changes (July 2025)

- **✅ Upgraded to Live Trading**: Switched from paper trading to real money trading mode
- **✅ Added Advanced Strategy Engine**: Multi-signal approach combining technical analysis, ML, and market regime detection
- **✅ Implemented Live Trader Module**: Real order execution with proper risk management
- **✅ Created Advanced Analysis**: Comprehensive market analysis with 85%+ confidence threshold
- **✅ Added Proxy Support**: Network connectivity improvements for restricted regions
- **✅ Enhanced Risk Management**: Dynamic position sizing, correlation monitoring, and drawdown protection

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask web framework with SQLAlchemy ORM
- **Database**: SQLite (configured to support PostgreSQL via DATABASE_URL environment variable)
- **API Structure**: RESTful API with Blueprint-based routing
- **Real-time Data**: WebSocket integration for live market data
- **Caching**: Custom SQLite-based data caching system

### Frontend Architecture
- **Template Engine**: Jinja2 templating with Bootstrap dark theme
- **JavaScript**: Vanilla JS with Chart.js for data visualization
- **Styling**: Bootstrap 5 with custom CSS enhancements
- **Real-time Updates**: AJAX-based price updates and dashboard refreshes

### Machine Learning Architecture
- **Model**: Random Forest regression for price prediction
- **Features**: Technical indicators, price patterns, and time-based features
- **Training**: Automated retraining with configurable intervals
- **Prediction**: Real-time predictions with confidence scoring

## Key Components

### Trading Engine
- **Paper Trading**: Simulated trading with configurable starting balance
- **Multi-pair Support**: Simultaneous trading across multiple cryptocurrency pairs
- **Risk Management**: Stop-loss, take-profit, position sizing, and daily loss limits
- **Strategy Engine**: Hybrid approach combining technical analysis and ML predictions

### Data Management
- **Binance Integration**: Real-time market data via Binance API and WebSocket
- **Technical Analysis**: 15+ indicators including RSI, MACD, Bollinger Bands, etc.
- **Historical Data**: Backtesting engine with historical performance analysis
- **Caching Layer**: SQLite-based caching for improved performance

### Web Interface
- **Dashboard**: Real-time portfolio overview and trading metrics
- **Portfolio Management**: Asset allocation and performance tracking
- **Backtesting Interface**: Historical strategy testing with configurable parameters
- **Settings Management**: Configuration of trading parameters and risk settings

## Data Flow

1. **Market Data Ingestion**: Binance API/WebSocket → Data Cache → Database
2. **Technical Analysis**: Raw price data → Technical Indicators → Trading Signals
3. **ML Predictions**: Feature engineering → Model inference → Confidence scoring
4. **Trading Decisions**: Combined signals → Strategy evaluation → Order execution
5. **Portfolio Updates**: Trade execution → Portfolio rebalancing → Performance tracking
6. **Web Interface**: Database queries → Template rendering → Real-time updates

## External Dependencies

### Core Trading
- **python-binance**: Binance API client for market data and trading
- **ta**: Technical analysis library for indicators
- **scikit-learn**: Machine learning models and preprocessing
- **pandas/numpy**: Data manipulation and numerical computing

### Web Framework
- **Flask**: Web framework with SQLAlchemy ORM
- **Jinja2**: Template engine for dynamic HTML generation
- **Bootstrap 5**: Frontend CSS framework with dark theme
- **Chart.js**: Interactive charting library

### Configuration
- **python-dotenv**: Environment variable management
- **JSON config**: Strategy parameters and trading settings
- **SQLite**: Default database with PostgreSQL support

## Deployment Strategy

### Environment Setup
- **Python 3.8+**: Core runtime requirement
- **Environment Variables**: API keys and database configuration
- **Configuration Files**: JSON-based trading parameters
- **Database**: SQLite for development, PostgreSQL for production

### Scalability Considerations
- **Database Migration**: Ready for PostgreSQL with Drizzle ORM potential
- **Caching Strategy**: Built-in data caching for API rate limit management
- **Multi-threading**: Thread-safe operations for concurrent trading
- **Resource Management**: Configurable refresh intervals and data retention

### Security Features
- **API Key Management**: Environment variable storage
- **Paper Trading**: Default safe trading mode
- **Rate Limiting**: Built-in API rate limit protection
- **Error Handling**: Comprehensive logging and error recovery

### Performance Optimization
- **Data Caching**: SQLite-based cache for frequently accessed data
- **Lazy Loading**: On-demand data fetching for web interface
- **Batch Processing**: Efficient bulk operations for historical data
- **Memory Management**: Configurable data retention and cleanup