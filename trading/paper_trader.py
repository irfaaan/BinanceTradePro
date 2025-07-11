import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from app import db
from models import Trade, Portfolio
from config import config
import uuid

logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self):
        self.initial_balance = config.get('paper_trading.initial_balance', 10000.0)
        self.base_currency = config.get('paper_trading.base_currency', 'USDT')
        self.active_trades = {}
        # Don't initialize portfolio in __init__ to avoid app context issues
    
    def initialize_portfolio(self):
        """Initialize paper trading portfolio"""
        try:
            # Check if portfolio exists
            base_portfolio = Portfolio.query.filter_by(asset=self.base_currency).first()
            if not base_portfolio:
                # Create initial portfolio
                base_portfolio = Portfolio(
                    asset=self.base_currency,
                    balance=self.initial_balance,
                    locked_balance=0.0,
                    avg_price=1.0,
                    updated_at=datetime.utcnow()
                )
                db.session.add(base_portfolio)
                db.session.commit()
                logger.info(f"Initialized paper trading portfolio with {self.initial_balance} {self.base_currency}")
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
    
    def ensure_portfolio_initialized(self):
        """Ensure portfolio is initialized (call this before using portfolio methods)"""
        try:
            base_portfolio = Portfolio.query.filter_by(asset=self.base_currency).first()
            if not base_portfolio:
                self.initialize_portfolio()
        except Exception as e:
            logger.error(f"Error ensuring portfolio initialization: {e}")
    
    def get_portfolio_balance(self, asset: str) -> float:
        """Get available balance for an asset"""
        try:
            self.ensure_portfolio_initialized()
            portfolio = Portfolio.query.filter_by(asset=asset).first()
            return portfolio.balance if portfolio else 0.0
        except Exception as e:
            logger.error(f"Error getting balance for {asset}: {e}")
            return 0.0
    
    def get_total_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value in base currency"""
        try:
            total_value = 0.0
            portfolios = Portfolio.query.all()
            
            for portfolio in portfolios:
                if portfolio.asset == self.base_currency:
                    total_value += portfolio.balance
                else:
                    # Convert to base currency
                    symbol = f"{portfolio.asset}{self.base_currency}"
                    if symbol in prices:
                        total_value += portfolio.balance * prices[symbol]
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float) -> float:
        """Calculate position size based on risk management"""
        try:
            base_balance = self.get_portfolio_balance(self.base_currency)
            max_position_size = config.get('risk_management.max_position_size', 0.05)
            
            # Adjust position size based on confidence
            adjusted_max_size = max_position_size * confidence
            
            # Calculate position value
            position_value = base_balance * adjusted_max_size
            
            # Calculate quantity
            quantity = position_value / price
            
            # Ensure we have enough balance
            if position_value > base_balance:
                quantity = base_balance / price
            
            return max(quantity, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def place_buy_order(self, symbol: str, quantity: float, price: float, 
                       strategy_signal: str = None, prediction_confidence: float = None) -> Optional[Dict]:
        """Place a buy order"""
        try:
            # Extract base and quote assets
            base_asset = symbol.replace(self.base_currency, '')
            quote_asset = self.base_currency
            
            total_cost = quantity * price
            
            # Check if we have enough balance
            available_balance = self.get_portfolio_balance(quote_asset)
            if total_cost > available_balance:
                logger.warning(f"Insufficient balance for buy order: {symbol}")
                return None
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=price,
                total_value=total_cost,
                strategy_signal=strategy_signal,
                prediction_confidence=prediction_confidence,
                is_paper_trade=True,
                status='FILLED',
                created_at=datetime.utcnow()
            )
            
            db.session.add(trade)
            
            # Update portfolio balances
            self._update_portfolio_on_buy(base_asset, quote_asset, quantity, total_cost, price)
            
            db.session.commit()
            
            trade_info = {
                'id': trade.id,
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': price,
                'total_value': total_cost,
                'timestamp': trade.created_at
            }
            
            logger.info(f"Buy order executed: {symbol} {quantity} @ {price}")
            return trade_info
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            db.session.rollback()
            return None
    
    def place_sell_order(self, symbol: str, quantity: float, price: float,
                        strategy_signal: str = None, prediction_confidence: float = None) -> Optional[Dict]:
        """Place a sell order"""
        try:
            # Extract base and quote assets
            base_asset = symbol.replace(self.base_currency, '')
            quote_asset = self.base_currency
            
            # Check if we have enough balance
            available_balance = self.get_portfolio_balance(base_asset)
            if quantity > available_balance:
                logger.warning(f"Insufficient balance for sell order: {symbol}")
                return None
            
            total_value = quantity * price
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                price=price,
                total_value=total_value,
                strategy_signal=strategy_signal,
                prediction_confidence=prediction_confidence,
                is_paper_trade=True,
                status='FILLED',
                created_at=datetime.utcnow()
            )
            
            db.session.add(trade)
            
            # Update portfolio balances
            self._update_portfolio_on_sell(base_asset, quote_asset, quantity, total_value, price)
            
            db.session.commit()
            
            trade_info = {
                'id': trade.id,
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'price': price,
                'total_value': total_value,
                'timestamp': trade.created_at
            }
            
            logger.info(f"Sell order executed: {symbol} {quantity} @ {price}")
            return trade_info
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            db.session.rollback()
            return None
    
    def _update_portfolio_on_buy(self, base_asset: str, quote_asset: str, 
                                quantity: float, total_cost: float, price: float):
        """Update portfolio after buy order"""
        # Update base asset (increase)
        base_portfolio = Portfolio.query.filter_by(asset=base_asset).first()
        if base_portfolio:
            # Update average price
            old_balance = base_portfolio.balance
            old_avg_price = base_portfolio.avg_price
            new_balance = old_balance + quantity
            new_avg_price = ((old_balance * old_avg_price) + (quantity * price)) / new_balance
            
            base_portfolio.balance = new_balance
            base_portfolio.avg_price = new_avg_price
            base_portfolio.updated_at = datetime.utcnow()
        else:
            base_portfolio = Portfolio(
                asset=base_asset,
                balance=quantity,
                locked_balance=0.0,
                avg_price=price,
                updated_at=datetime.utcnow()
            )
            db.session.add(base_portfolio)
        
        # Update quote asset (decrease)
        quote_portfolio = Portfolio.query.filter_by(asset=quote_asset).first()
        if quote_portfolio:
            quote_portfolio.balance -= total_cost
            quote_portfolio.updated_at = datetime.utcnow()
    
    def _update_portfolio_on_sell(self, base_asset: str, quote_asset: str,
                                 quantity: float, total_value: float, price: float):
        """Update portfolio after sell order"""
        # Update base asset (decrease)
        base_portfolio = Portfolio.query.filter_by(asset=base_asset).first()
        if base_portfolio:
            base_portfolio.balance -= quantity
            base_portfolio.updated_at = datetime.utcnow()
        
        # Update quote asset (increase)
        quote_portfolio = Portfolio.query.filter_by(asset=quote_asset).first()
        if quote_portfolio:
            quote_portfolio.balance += total_value
            quote_portfolio.updated_at = datetime.utcnow()
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        try:
            query = Trade.query.filter_by(is_paper_trade=True)
            if symbol:
                query = query.filter_by(symbol=symbol)
            
            trades = query.order_by(Trade.created_at.desc()).limit(limit).all()
            
            return [{
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'strategy_signal': trade.strategy_signal,
                'prediction_confidence': trade.prediction_confidence,
                'timestamp': trade.created_at
            } for trade in trades]
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            trades = Trade.query.filter_by(is_paper_trade=True).all()
            
            if not trades:
                return {}
            
            # Calculate basic metrics
            total_trades = len(trades)
            buy_trades = [t for t in trades if t.side == 'BUY']
            sell_trades = [t for t in trades if t.side == 'SELL']
            
            total_volume = sum(t.total_value for t in trades)
            
            # Calculate win/loss (simplified)
            wins = 0
            losses = 0
            total_pnl = 0
            
            # Match buy and sell trades (simplified)
            for sell_trade in sell_trades:
                matching_buys = [t for t in buy_trades if t.symbol == sell_trade.symbol]
                if matching_buys:
                    buy_trade = matching_buys[-1]  # Most recent buy
                    pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
            
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            
            metrics = {
                'total_trades': total_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_volume': total_volume,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            self.ensure_portfolio_initialized()
            
            portfolios = Portfolio.query.all()
            summary = {
                'assets': {},
                'total_value': 0.0,
                'initial_balance': self.initial_balance,
                'pnl': 0.0,
                'pnl_percent': 0.0
            }
            
            for portfolio in portfolios:
                summary['assets'][portfolio.asset] = {
                    'balance': portfolio.balance,
                    'locked_balance': portfolio.locked_balance,
                    'avg_price': portfolio.avg_price,
                    'updated_at': portfolio.updated_at.isoformat() if portfolio.updated_at else None
                }
            
            # Calculate total value (simplified - assumes base currency)
            base_balance = summary['assets'].get(self.base_currency, {}).get('balance', 0.0)
            summary['total_value'] = base_balance
            summary['pnl'] = base_balance - self.initial_balance
            summary['pnl_percent'] = (summary['pnl'] / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': 0,
                'base_currency': self.base_currency,
                'assets': {},
                'initial_balance': self.initial_balance,
                'pnl': 0,
                'pnl_percent': 0
            }

# Global paper trader instance
paper_trader = PaperTrader()
