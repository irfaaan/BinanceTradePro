from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import json
import logging
from config import config
from api.binance_client import binance_client
from analysis.technical_indicators import technical_analyzer
from analysis.ml_predictor import ml_predictor
from trading.strategy import trading_strategy
from trading.paper_trader import paper_trader
from backtesting.backtest_engine import backtest_engine
from utils.data_cache import data_cache
from models import TradingPair, Trade, Portfolio, db
import pandas as pd

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'binance_connected': binance_client.is_connected()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@api_bp.route('/market-data/<symbol>')
def get_market_data(symbol):
    """Get real-time market data for a symbol"""
    try:
        # Check cache first
        cache_key = f"market_data_{symbol}"
        cached_data = data_cache.get_cached_data(cache_key)
        
        if cached_data:
            return jsonify(cached_data)
        
        # Get fresh data
        current_price = binance_client.get_ticker_price(symbol)
        ticker_24hr = binance_client.get_24hr_ticker(symbol)
        
        if not current_price or not ticker_24hr:
            return jsonify({'error': 'Failed to get market data'}), 500
        
        data = {
            'symbol': symbol,
            'price': current_price,
            'change_24h': float(ticker_24hr.get('priceChangePercent', 0)),
            'volume_24h': float(ticker_24hr.get('volume', 0)),
            'high_24h': float(ticker_24hr.get('highPrice', 0)),
            'low_24h': float(ticker_24hr.get('lowPrice', 0)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache for 30 seconds
        data_cache.cache_data(cache_key, data, expiry_minutes=0.5)
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/chart-data/<symbol>')
def get_chart_data(symbol):
    """Get chart data for a symbol"""
    try:
        interval = request.args.get('interval', '1h')
        limit = int(request.args.get('limit', 100))
        
        # Check cache
        cache_key = f"chart_data_{symbol}_{interval}_{limit}"
        cached_data = data_cache.get_cached_data(cache_key)
        
        if cached_data:
            return jsonify(cached_data)
        
        # Get historical data
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if not klines:
            return jsonify({'error': 'No chart data available'}), 404
        
        # Convert to chart format
        chart_data = []
        for kline in klines:
            chart_data.append({
                'timestamp': int(kline[0]),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        
        result = {
            'symbol': symbol,
            'interval': interval,
            'data': chart_data
        }
        
        # Cache for 5 minutes
        data_cache.cache_data(cache_key, result, expiry_minutes=5)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting chart data for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/indicators/<symbol>')
def get_indicators(symbol):
    """Get technical indicators for a symbol"""
    try:
        # Get historical data
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval='1h',
            limit=100
        )
        
        if not klines:
            return jsonify({'error': 'No data available'}), 404
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'ignore'
        ])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate indicators
        indicators = technical_analyzer.calculate_all_indicators(df)
        signals = technical_analyzer.get_trading_signals(df)
        support_resistance = technical_analyzer.get_support_resistance(df)
        
        # Get latest values
        latest_indicators = {}
        for key, series in indicators.items():
            if not series.empty:
                latest_indicators[key] = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else None
        
        result = {
            'symbol': symbol,
            'indicators': latest_indicators,
            'signals': signals,
            'support_resistance': support_resistance,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/prediction/<symbol>')
def get_prediction(symbol):
    """Get ML prediction for a symbol"""
    try:
        # Get historical data
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval='1h',
            limit=100
        )
        
        if not klines:
            return jsonify({'error': 'No data available'}), 404
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'ignore'
        ])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Get prediction
        prediction = ml_predictor.predict(df, symbol)
        
        if prediction:
            return jsonify(prediction)
        else:
            return jsonify({'error': 'No prediction available'}), 404
            
    except Exception as e:
        logger.error(f"Error getting prediction for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/portfolio')
def get_portfolio():
    """Get portfolio summary"""
    try:
        portfolio_summary = paper_trader.get_portfolio_summary()
        return jsonify(portfolio_summary)
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/trades')
def get_trades():
    """Get trade history"""
    try:
        symbol = request.args.get('symbol')
        limit = int(request.args.get('limit', 50))
        
        trades = paper_trader.get_trade_history(symbol=symbol, limit=limit)
        return jsonify(trades)
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtest"""
    try:
        data = request.json
        symbol = data.get('symbol')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        strategy_params = data.get('strategy_params', {})
        
        if not symbol or not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Run backtest
        result = backtest_engine.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params
        )
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Backtest failed'}), 500
            
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/backtest-multi', methods=['POST'])
def run_multi_backtest():
    """Run backtest for multiple symbols"""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        strategy_params = data.get('strategy_params', {})
        
        if not symbols or not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Run multi-symbol backtest
        results = backtest_engine.run_multi_symbol_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params
        )
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error running multi-symbol backtest: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/strategy-status')
def get_strategy_status():
    """Get current strategy status"""
    try:
        status = trading_strategy.get_portfolio_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting strategy status: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/cache-stats')
def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = data_cache.get_cache_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache"""
    try:
        cache_type = request.json.get('type', 'expired')
        
        if cache_type == 'all':
            data_cache.clear_all_cache()
        else:
            data_cache.clear_expired_cache()
        
        return jsonify({'success': True, 'message': f'Cache cleared: {cache_type}'})
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/trading-pairs')
def get_trading_pairs():
    """Get configured trading pairs"""
    try:
        pairs = config.trading_pairs
        
        # Get current prices
        prices = {}
        for pair in pairs:
            price = binance_client.get_ticker_price(pair)
            if price:
                prices[pair] = price
        
        return jsonify({
            'pairs': pairs,
            'prices': prices
        })
        
    except Exception as e:
        logger.error(f"Error getting trading pairs: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/add-trading-pair', methods=['POST'])
def add_trading_pair():
    """Add a new trading pair"""
    try:
        data = request.json
        symbol = data.get('symbol', '').upper()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        # Validate symbol
        if not binance_client.validate_trading_pair(symbol):
            return jsonify({'error': 'Invalid trading pair'}), 400
        
        # Add to config
        current_pairs = config.trading_pairs
        if symbol not in current_pairs:
            current_pairs.append(symbol)
            config.update('trading_pairs', current_pairs)
            
            return jsonify({'success': True, 'message': f'Added {symbol} to trading pairs'})
        else:
            return jsonify({'error': 'Trading pair already exists'}), 400
            
    except Exception as e:
        logger.error(f"Error adding trading pair: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/remove-trading-pair', methods=['POST'])
def remove_trading_pair():
    """Remove a trading pair"""
    try:
        data = request.json
        symbol = data.get('symbol', '').upper()
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        # Remove from config
        current_pairs = config.trading_pairs
        if symbol in current_pairs:
            current_pairs.remove(symbol)
            config.update('trading_pairs', current_pairs)
            
            return jsonify({'success': True, 'message': f'Removed {symbol} from trading pairs'})
        else:
            return jsonify({'error': 'Trading pair not found'}), 400
            
    except Exception as e:
        logger.error(f"Error removing trading pair: {e}")
        return jsonify({'error': str(e)}), 500
