import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from app import db
from models import Trade, Portfolio
from config import config
from api.binance_client import binance_client
import time

logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self):
        self.base_currency = config.get('paper_trading.base_currency', 'USDT')
        self.active_trades = {}
        self.risk_management = config.get('risk_management', {})
        self.last_trade_time = {}
        self.daily_loss_tracker = {}
        
    def get_account_info(self) -> Optional[Dict]:
        """Get real account information from Binance"""
        try:
            return binance_client.get_account_info()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_real_balances(self) -> Dict[str, float]:
        """Get real account balances from Binance"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return {}
            
            balances = {}
            for balance in account_info['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return balances
        except Exception as e:
            logger.error(f"Error getting real balances: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> float:
        """Calculate position size based on risk management and real account balance"""
        try:
            balances = self.get_real_balances()
            base_balance = balances.get(self.base_currency, {}).get('free', 0)
            
            if base_balance == 0:
                logger.warning(f"No {self.base_currency} balance available")
                return 0
            
            max_position_size = self.risk_management.get('max_position_size', 0.05)
            
            # Adjust position size based on confidence
            confidence_multiplier = min(confidence, 1.0)
            
            # Calculate position size
            position_value = base_balance * max_position_size * confidence_multiplier
            position_size = position_value / price
            
            # Apply minimum trade size from exchange
            min_trade_size = self.get_min_trade_size(symbol)
            if position_size < min_trade_size:
                logger.warning(f"Position size {position_size} below minimum {min_trade_size} for {symbol}")
                return 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_min_trade_size(self, symbol: str) -> float:
        """Get minimum trade size for a symbol"""
        try:
            symbol_info = binance_client.get_symbol_info(symbol)
            if symbol_info:
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'LOT_SIZE':
                        return float(filter_info['minQty'])
            return 0.0001  # Default minimum
        except Exception as e:
            logger.error(f"Error getting min trade size for {symbol}: {e}")
            return 0.0001
    
    def place_buy_order(self, symbol: str, quantity: float, price: float = None, 
                       order_type: str = 'MARKET', strategy_signal: str = None, 
                       prediction_confidence: float = None) -> Optional[Dict]:
        """Place a real buy order on Binance"""
        try:
            if not self.check_risk_limits(symbol, quantity, price or 0):
                return None
            
            # Round quantity to appropriate decimal places
            quantity = self.round_quantity(symbol, quantity)
            
            if order_type == 'MARKET':
                order = binance_client.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
            else:  # LIMIT
                if not price:
                    logger.error("Price required for limit order")
                    return None
                
                price = self.round_price(symbol, price)
                order = binance_client.client.order_limit_buy(
                    symbol=symbol,
                    quantity=quantity,
                    price=str(price)
                )
            
            # Log the trade
            self.log_trade(order, strategy_signal, prediction_confidence)
            
            logger.info(f"Buy order placed: {symbol} - {quantity} at {price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return None
    
    def place_sell_order(self, symbol: str, quantity: float, price: float = None,
                        order_type: str = 'MARKET', strategy_signal: str = None,
                        prediction_confidence: float = None) -> Optional[Dict]:
        """Place a real sell order on Binance"""
        try:
            if not self.check_available_balance(symbol, quantity):
                return None
            
            # Round quantity to appropriate decimal places
            quantity = self.round_quantity(symbol, quantity)
            
            if order_type == 'MARKET':
                order = binance_client.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
            else:  # LIMIT
                if not price:
                    logger.error("Price required for limit order")
                    return None
                
                price = self.round_price(symbol, price)
                order = binance_client.client.order_limit_sell(
                    symbol=symbol,
                    quantity=quantity,
                    price=str(price)
                )
            
            # Log the trade
            self.log_trade(order, strategy_signal, prediction_confidence)
            
            logger.info(f"Sell order placed: {symbol} - {quantity} at {price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing sell order for {symbol}: {e}")
            return None
    
    def check_risk_limits(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if trade meets risk management criteria"""
        try:
            # Check daily loss limit
            today = datetime.now().date()
            daily_loss = self.daily_loss_tracker.get(today, 0)
            max_daily_loss = self.risk_management.get('max_daily_loss', 0.1)
            
            balances = self.get_real_balances()
            total_value = sum(bal['total'] for bal in balances.values())
            
            if daily_loss > (total_value * max_daily_loss):
                logger.warning(f"Daily loss limit exceeded: {daily_loss}")
                return False
            
            # Check maximum open trades
            max_trades = self.risk_management.get('max_open_trades', 5)
            if len(self.active_trades) >= max_trades:
                logger.warning(f"Maximum open trades limit reached: {max_trades}")
                return False
            
            # Check position size limit
            position_value = quantity * price
            max_position_size = self.risk_management.get('max_position_size', 0.05)
            
            if position_value > (total_value * max_position_size):
                logger.warning(f"Position size too large: {position_value}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def check_available_balance(self, symbol: str, quantity: float) -> bool:
        """Check if sufficient balance is available for sell order"""
        try:
            base_asset = symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '')
            balances = self.get_real_balances()
            
            available = balances.get(base_asset, {}).get('free', 0)
            
            if available < quantity:
                logger.warning(f"Insufficient balance for {symbol}: {available} < {quantity}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking available balance: {e}")
            return False
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to appropriate decimal places based on symbol rules"""
        try:
            symbol_info = binance_client.get_symbol_info(symbol)
            if symbol_info:
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'LOT_SIZE':
                        step_size = float(filter_info['stepSize'])
                        precision = abs(Decimal(str(step_size)).as_tuple().exponent)
                        return round(quantity, precision)
            return round(quantity, 6)  # Default precision
        except Exception as e:
            logger.error(f"Error rounding quantity for {symbol}: {e}")
            return round(quantity, 6)
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to appropriate decimal places based on symbol rules"""
        try:
            symbol_info = binance_client.get_symbol_info(symbol)
            if symbol_info:
                for filter_info in symbol_info['filters']:
                    if filter_info['filterType'] == 'PRICE_FILTER':
                        tick_size = float(filter_info['tickSize'])
                        precision = abs(Decimal(str(tick_size)).as_tuple().exponent)
                        return round(price, precision)
            return round(price, 8)  # Default precision
        except Exception as e:
            logger.error(f"Error rounding price for {symbol}: {e}")
            return round(price, 8)
    
    def log_trade(self, order: Dict, strategy_signal: str = None, 
                  prediction_confidence: float = None):
        """Log trade to database"""
        try:
            trade = Trade(
                symbol=order['symbol'],
                side=order['side'],
                quantity=float(order['executedQty']),
                price=float(order['price']) if 'price' in order else float(order['fills'][0]['price']),
                total_value=float(order['cummulativeQuoteQty']),
                strategy_signal=strategy_signal,
                prediction_confidence=prediction_confidence,
                is_paper_trade=False,
                status=order['status'],
                created_at=datetime.utcnow()
            )
            db.session.add(trade)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders from Binance"""
        try:
            if symbol:
                return binance_client.client.get_open_orders(symbol=symbol)
            else:
                return binance_client.client.get_open_orders()
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def cancel_order(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Cancel an order"""
        try:
            return binance_client.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return None
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history from database"""
        try:
            query = Trade.query.filter_by(is_paper_trade=False)
            if symbol:
                query = query.filter_by(symbol=symbol)
            
            trades = query.order_by(Trade.created_at.desc()).limit(limit).all()
            
            return [{
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'created_at': trade.created_at.isoformat(),
                'strategy_signal': trade.strategy_signal,
                'prediction_confidence': trade.prediction_confidence
            } for trade in trades]
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with real balances"""
        try:
            balances = self.get_real_balances()
            
            if not balances:
                return {'error': 'Unable to fetch account balances'}
            
            # Calculate total value in USDT
            total_value = 0.0
            for asset, balance in balances.items():
                if asset == 'USDT':
                    total_value += balance['total']
                else:
                    # Get price in USDT
                    symbol = f"{asset}USDT"
                    price = binance_client.get_ticker_price(symbol)
                    if price:
                        total_value += balance['total'] * price
            
            return {
                'total_value': total_value,
                'assets': balances,
                'base_currency': self.base_currency,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
    
    def set_stop_loss(self, symbol: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Set stop loss order"""
        try:
            return binance_client.client.create_oco_order(
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                price=str(stop_price * 1.01),  # Limit price slightly above stop
                stopPrice=str(stop_price)
            )
        except Exception as e:
            logger.error(f"Error setting stop loss for {symbol}: {e}")
            return None
    
    def set_take_profit(self, symbol: str, quantity: float, target_price: float) -> Optional[Dict]:
        """Set take profit order"""
        try:
            return binance_client.client.order_limit_sell(
                symbol=symbol,
                quantity=quantity,
                price=str(target_price)
            )
        except Exception as e:
            logger.error(f"Error setting take profit for {symbol}: {e}")
            return None

# Global instance
live_trader = LiveTrader()