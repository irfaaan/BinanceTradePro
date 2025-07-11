from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from datetime import datetime, timedelta
import json
import logging
from config import config
from api.binance_client import binance_client
from analysis.technical_indicators import technical_analyzer
from analysis.ml_predictor import ml_predictor
from trading.strategy import trading_strategy
from trading.paper_trader import paper_trader
from models import TradingPair, Trade, Portfolio, db

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get trading pairs
        trading_pairs = config.trading_pairs
        
        # Get portfolio summary
        portfolio_summary = paper_trader.get_portfolio_summary()
        
        # Get recent trades
        recent_trades = paper_trader.get_trade_history(limit=10)
        
        # Get performance metrics
        performance_metrics = paper_trader.calculate_performance_metrics()
        
        context = {
            'trading_pairs': trading_pairs,
            'portfolio_summary': portfolio_summary,
            'recent_trades': recent_trades,
            'performance_metrics': performance_metrics,
            'is_connected': binance_client.is_connected()
        }
        
        return render_template('dashboard.html', **context)
        
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        flash(f"Error loading dashboard: {str(e)}", 'error')
        return render_template('dashboard.html', error=str(e))

@dashboard_bp.route('/market-data/<symbol>')
def market_data(symbol):
    """Get market data for a specific symbol"""
    try:
        # Get current price
        current_price = binance_client.get_ticker_price(symbol)
        
        # Get 24hr ticker
        ticker_24hr = binance_client.get_24hr_ticker(symbol)
        
        # Get recent trades
        recent_trades = binance_client.get_recent_trades(symbol, limit=20)
        
        # Get order book
        order_book = binance_client.get_order_book(symbol, limit=20)
        
        data = {
            'symbol': symbol,
            'current_price': current_price,
            'ticker_24hr': ticker_24hr,
            'recent_trades': recent_trades,
            'order_book': order_book,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/analysis/<symbol>')
def analysis(symbol):
    """Get analysis for a specific symbol"""
    try:
        # Get historical data
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval='1h',
            limit=100
        )
        
        if not klines:
            return jsonify({'error': 'No historical data available'}), 404
        
        # Convert to DataFrame
        import pandas as pd
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
        
        # Calculate technical indicators
        indicators = technical_analyzer.calculate_all_indicators(df)
        
        # Get trading signals
        signals = technical_analyzer.get_trading_signals(df)
        
        # Get support/resistance levels
        support_resistance = technical_analyzer.get_support_resistance(df)
        
        # Get ML prediction
        prediction = ml_predictor.predict(df, symbol)
        
        # Prepare chart data
        chart_data = {
            'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'ohlc': df[['open', 'high', 'low', 'close']].values.tolist(),
            'volume': df['volume'].tolist(),
            'indicators': {
                'sma_20': indicators.get('sma_20', pd.Series()).fillna(0).tolist()[-50:] if 'sma_20' in indicators else [],
                'sma_50': indicators.get('sma_50', pd.Series()).fillna(0).tolist()[-50:] if 'sma_50' in indicators else [],
                'rsi': indicators.get('rsi', pd.Series()).fillna(0).tolist()[-50:] if 'rsi' in indicators else [],
                'bb_high': indicators.get('bb_high', pd.Series()).fillna(0).tolist()[-50:] if 'bb_high' in indicators else [],
                'bb_low': indicators.get('bb_low', pd.Series()).fillna(0).tolist()[-50:] if 'bb_low' in indicators else []
            }
        }
        
        data = {
            'symbol': symbol,
            'chart_data': chart_data,
            'signals': signals,
            'support_resistance': support_resistance,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting analysis for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/portfolio')
def portfolio():
    """Portfolio management page"""
    try:
        portfolio_summary = paper_trader.get_portfolio_summary()
        trade_history = paper_trader.get_trade_history(limit=50)
        performance_metrics = paper_trader.calculate_performance_metrics()
        strategy_status = trading_strategy.get_portfolio_status()
        
        context = {
            'portfolio_summary': portfolio_summary,
            'trade_history': trade_history,
            'performance_metrics': performance_metrics,
            'strategy_status': strategy_status
        }
        
        return render_template('portfolio.html', **context)
        
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        flash(f"Error loading portfolio: {str(e)}", 'error')
        return render_template('portfolio.html', error=str(e))

@dashboard_bp.route('/backtesting')
def backtesting():
    """Backtesting page"""
    try:
        trading_pairs = config.trading_pairs
        
        context = {
            'trading_pairs': trading_pairs,
            'default_strategy_params': {
                'min_confidence': 0.7,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size': 0.1
            }
        }
        
        return render_template('backtesting.html', **context)
        
    except Exception as e:
        logger.error(f"Error loading backtesting: {e}")
        flash(f"Error loading backtesting: {str(e)}", 'error')
        return render_template('backtesting.html', error=str(e))

@dashboard_bp.route('/settings')
def settings():
    """Settings page"""
    try:
        current_config = {
            'trading_pairs': config.trading_pairs,
            'risk_management': config.get('risk_management', {}),
            'trading_strategy': config.get('trading_strategy', {}),
            'data_settings': config.get('data_settings', {}),
            'paper_trading': config.get('paper_trading', {}),
            'ml_settings': config.get('ml_settings', {}),
            'api_settings': config.get('api_settings', {})
        }
        
        context = {
            'config': current_config,
            'is_connected': binance_client.is_connected()
        }
        
        return render_template('settings.html', **context)
        
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        flash(f"Error loading settings: {str(e)}", 'error')
        return render_template('settings.html', error=str(e))

@dashboard_bp.route('/update-settings', methods=['POST'])
def update_settings():
    """Update bot settings"""
    try:
        data = request.json
        
        # Update configuration
        for key, value in data.items():
            config.update(key, value)
        
        # Update strategy parameters if provided
        if 'trading_strategy' in data:
            trading_strategy.update_strategy_parameters(data['trading_strategy'])
        
        return jsonify({'success': True, 'message': 'Settings updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/manual-trade', methods=['POST'])
def manual_trade():
    """Execute manual trade"""
    try:
        data = request.json
        symbol = data.get('symbol')
        side = data.get('side', '').upper()
        quantity = float(data.get('quantity', 0))
        price = float(data.get('price', 0))
        
        if not symbol or side not in ['BUY', 'SELL'] or quantity <= 0 or price <= 0:
            return jsonify({'error': 'Invalid trade parameters'}), 400
        
        # Execute trade
        if side == 'BUY':
            trade_info = paper_trader.place_buy_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                strategy_signal='MANUAL'
            )
        else:
            trade_info = paper_trader.place_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                strategy_signal='MANUAL'
            )
        
        if trade_info:
            return jsonify({
                'success': True,
                'message': f'Manual {side} order executed',
                'trade_info': trade_info
            })
        else:
            return jsonify({'error': 'Failed to execute trade'}), 500
            
    except Exception as e:
        logger.error(f"Error executing manual trade: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/train-model/<symbol>', methods=['POST'])
def train_model(symbol):
    """Train ML model for a symbol"""
    try:
        # Get historical data
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval='1h',
            limit=1000
        )
        
        if not klines:
            return jsonify({'error': 'No historical data available'}), 404
        
        # Convert to DataFrame
        import pandas as pd
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
        
        # Train model
        success = ml_predictor.train_model(df, symbol)
        
        if success:
            performance = ml_predictor.get_model_performance(symbol)
            return jsonify({
                'success': True,
                'message': f'Model trained successfully for {symbol}',
                'performance': performance
            })
        else:
            return jsonify({'error': 'Failed to train model'}), 500
            
    except Exception as e:
        logger.error(f"Error training model for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500
