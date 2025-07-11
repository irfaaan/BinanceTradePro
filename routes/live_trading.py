from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from datetime import datetime, timedelta
import json
import logging
from config import config
from trading.live_trader import live_trader
from trading.advanced_strategy import advanced_strategy
from api.binance_client import binance_client
from models import Trade, Portfolio, db

logger = logging.getLogger(__name__)

live_trading_bp = Blueprint('live_trading', __name__)

@live_trading_bp.route('/live-trading')
def live_trading_dashboard():
    """Live trading dashboard"""
    try:
        # Get real account info
        account_info = live_trader.get_account_info()
        
        # Get real balances
        balances = live_trader.get_real_balances()
        
        # Get portfolio summary
        portfolio_summary = live_trader.get_portfolio_summary()
        
        # Get recent live trades
        recent_trades = live_trader.get_trade_history(limit=20)
        
        # Get active positions
        active_positions = advanced_strategy.active_positions
        
        context = {
            'account_info': account_info,
            'balances': balances,
            'portfolio_summary': portfolio_summary,
            'recent_trades': recent_trades,
            'active_positions': active_positions,
            'is_connected': binance_client.is_connected(),
            'trading_pairs': config.trading_pairs
        }
        
        return render_template('live_trading.html', **context)
        
    except Exception as e:
        logger.error(f"Error loading live trading dashboard: {e}")
        flash(f"Error loading live trading data: {str(e)}", 'error')
        return render_template('live_trading.html', error=str(e))

@live_trading_bp.route('/api/live-trade', methods=['POST'])
def execute_live_trade():
    """Execute a live trade"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        action = data.get('action')  # 'BUY' or 'SELL'
        quantity = float(data.get('quantity', 0))
        order_type = data.get('order_type', 'MARKET')
        price = float(data.get('price', 0)) if data.get('price') else None
        
        if not symbol or not action:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Execute trade
        if action == 'BUY':
            result = live_trader.place_buy_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                strategy_signal="Manual Trade"
            )
        else:  # SELL
            result = live_trader.place_sell_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                strategy_signal="Manual Trade"
            )
        
        if result:
            return jsonify({
                'success': True,
                'order_id': result.get('orderId'),
                'message': f'{action} order placed successfully'
            })
        else:
            return jsonify({'error': 'Failed to place order'}), 500
            
    except Exception as e:
        logger.error(f"Error executing live trade: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/advanced-analysis/<symbol>')
def get_advanced_analysis(symbol):
    """Get advanced analysis for a symbol"""
    try:
        analysis = advanced_strategy.analyze_market_comprehensive(symbol)
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error getting advanced analysis for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/execute-advanced-trade', methods=['POST'])
def execute_advanced_trade():
    """Execute trade using advanced strategy"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        # Get comprehensive analysis
        analysis = advanced_strategy.analyze_market_comprehensive(symbol)
        
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 500
        
        # Execute trade based on analysis
        result = advanced_strategy.execute_advanced_trade(symbol, analysis)
        
        if result:
            return jsonify({
                'success': True,
                'trade_result': result,
                'analysis': analysis,
                'message': 'Advanced trade executed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'analysis': analysis,
                'message': 'No trade executed based on current analysis'
            })
            
    except Exception as e:
        logger.error(f"Error executing advanced trade: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/cancel-order', methods=['POST'])
def cancel_order():
    """Cancel an open order"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        order_id = data.get('order_id')
        
        if not symbol or not order_id:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        result = live_trader.cancel_order(symbol, order_id)
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Order cancelled successfully'
            })
        else:
            return jsonify({'error': 'Failed to cancel order'}), 500
            
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/open-orders/<symbol>')
def get_open_orders(symbol):
    """Get open orders for a symbol"""
    try:
        orders = live_trader.get_open_orders(symbol)
        return jsonify(orders)
        
    except Exception as e:
        logger.error(f"Error getting open orders for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/set-stop-loss', methods=['POST'])
def set_stop_loss():
    """Set stop loss order"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        quantity = float(data.get('quantity'))
        stop_price = float(data.get('stop_price'))
        
        result = live_trader.set_stop_loss(symbol, quantity, stop_price)
        
        if result:
            return jsonify({
                'success': True,
                'order_id': result.get('orderId'),
                'message': 'Stop loss set successfully'
            })
        else:
            return jsonify({'error': 'Failed to set stop loss'}), 500
            
    except Exception as e:
        logger.error(f"Error setting stop loss: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/set-take-profit', methods=['POST'])
def set_take_profit():
    """Set take profit order"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        quantity = float(data.get('quantity'))
        target_price = float(data.get('target_price'))
        
        result = live_trader.set_take_profit(symbol, quantity, target_price)
        
        if result:
            return jsonify({
                'success': True,
                'order_id': result.get('orderId'),
                'message': 'Take profit set successfully'
            })
        else:
            return jsonify({'error': 'Failed to set take profit'}), 500
            
    except Exception as e:
        logger.error(f"Error setting take profit: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/portfolio-performance')
def get_portfolio_performance():
    """Get portfolio performance metrics"""
    try:
        # Get real portfolio data
        portfolio_summary = live_trader.get_portfolio_summary()
        
        # Get trade history for performance calculation
        trades = live_trader.get_trade_history(limit=100)
        
        # Calculate performance metrics
        performance = {
            'total_value': portfolio_summary.get('total_value', 0),
            'total_trades': len(trades),
            'profitable_trades': len([t for t in trades if t.get('profit', 0) > 0]),
            'loss_trades': len([t for t in trades if t.get('profit', 0) < 0]),
            'win_rate': 0,
            'average_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        if performance['total_trades'] > 0:
            performance['win_rate'] = (performance['profitable_trades'] / performance['total_trades']) * 100
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        return jsonify({'error': str(e)}), 500

@live_trading_bp.route('/api/risk-metrics')
def get_risk_metrics():
    """Get current risk metrics"""
    try:
        # Calculate current portfolio risk
        risk_metrics = {
            'portfolio_heat': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'correlation_risk': 0.0,
            'position_concentration': 0.0,
            'leverage_ratio': 1.0,
            'risk_level': 'LOW'
        }
        
        # Get real balances for risk calculation
        balances = live_trader.get_real_balances()
        
        if balances:
            # Calculate position concentration
            total_value = sum(bal['total'] for bal in balances.values())
            if total_value > 0:
                max_position = max(bal['total'] for bal in balances.values())
                risk_metrics['position_concentration'] = max_position / total_value
        
        # Determine risk level
        if risk_metrics['position_concentration'] > 0.5:
            risk_metrics['risk_level'] = 'HIGH'
        elif risk_metrics['position_concentration'] > 0.3:
            risk_metrics['risk_level'] = 'MEDIUM'
        else:
            risk_metrics['risk_level'] = 'LOW'
        
        return jsonify(risk_metrics)
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return jsonify({'error': str(e)}), 500