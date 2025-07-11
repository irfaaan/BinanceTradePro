import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from analysis.technical_indicators import technical_analyzer
from analysis.ml_predictor import ml_predictor
from trading.paper_trader import paper_trader
from config import config

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self):
        self.min_confidence = config.get('trading_strategy.min_confidence', 0.90)
        self.max_open_trades = config.get('risk_management.max_open_trades', 5)
        self.stop_loss_percent = config.get('risk_management.stop_loss_percent', 0.01)
        self.take_profit_percent = config.get('risk_management.take_profit_percent', 0.02)
        self.open_positions = {}
        
    def analyze_market(self, symbol: str, df) -> Dict:
        """Analyze market conditions for a symbol"""
        try:
            if df.empty:
                return {}
            
            # Get technical indicators and signals
            signals = technical_analyzer.get_trading_signals(df)
            support_resistance = technical_analyzer.get_support_resistance(df)
            
            # Get ML prediction
            prediction = ml_predictor.predict(df, symbol)
            
            # Combine analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'technical_signals': signals,
                'support_resistance': support_resistance,
                'ml_prediction': prediction,
                'current_price': df['close'].iloc[-1]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return {}
    
    def generate_trading_decision(self, analysis: Dict) -> Optional[Dict]:
        """Generate trading decision based on analysis"""
        try:
            if not analysis:
                return None
            
            symbol = analysis['symbol']
            current_price = analysis['current_price']
            technical_signals = analysis.get('technical_signals', {})
            ml_prediction = analysis.get('ml_prediction', {})
            
            # Check if we already have a position
            if symbol in self.open_positions:
                return self._check_exit_conditions(symbol, analysis)
            
            # Check entry conditions
            decision = self._check_entry_conditions(symbol, analysis)
            
            if decision and decision['action'] in ['BUY', 'SELL']:
                # Calculate position size
                confidence = decision.get('confidence', 0.0)
                quantity = paper_trader.calculate_position_size(symbol, current_price, confidence)
                
                if quantity > 0:
                    decision['quantity'] = quantity
                    decision['price'] = current_price
                    return decision
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating trading decision: {e}")
            return None
    
    def _check_entry_conditions(self, symbol: str, analysis: Dict) -> Optional[Dict]:
        """Check conditions for entering a new position"""
        try:
            technical_signals = analysis.get('technical_signals', {})
            ml_prediction = analysis.get('ml_prediction', {})
            support_resistance = analysis.get('support_resistance', {})
            
            # Get overall technical signal
            tech_signal = technical_signals.get('overall_signal', 'HOLD')
            tech_confidence = technical_signals.get('overall_confidence', 0.0)
            
            # Get ML prediction
            ml_direction = ml_prediction.get('direction', 'HOLD') if ml_prediction else 'HOLD'
            ml_confidence = ml_prediction.get('confidence', 0.0) if ml_prediction else 0.0
            
            # Combine signals
            combined_confidence = (tech_confidence + ml_confidence) / 2
            
            # Check minimum confidence threshold
            if combined_confidence < self.min_confidence:
                return None
            
            # Check if technical and ML signals agree
            if tech_signal == 'BUY' and ml_direction == 'UP':
                return {
                    'action': 'BUY',
                    'confidence': combined_confidence,
                    'reason': f'Technical: {tech_signal} ({tech_confidence:.2f}), ML: {ml_direction} ({ml_confidence:.2f})',
                    'technical_signals': technical_signals,
                    'ml_prediction': ml_prediction
                }
            elif tech_signal == 'SELL' and ml_direction == 'DOWN':
                return {
                    'action': 'SELL',
                    'confidence': combined_confidence,
                    'reason': f'Technical: {tech_signal} ({tech_confidence:.2f}), ML: {ml_direction} ({ml_confidence:.2f})',
                    'technical_signals': technical_signals,
                    'ml_prediction': ml_prediction
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {e}")
            return None
    
    def _check_exit_conditions(self, symbol: str, analysis: Dict) -> Optional[Dict]:
        """Check conditions for exiting an existing position"""
        try:
            if symbol not in self.open_positions:
                return None
            
            position = self.open_positions[symbol]
            current_price = analysis['current_price']
            entry_price = position['entry_price']
            side = position['side']
            
            # Calculate current P&L
            if side == 'BUY':
                pnl_percent = (current_price - entry_price) / entry_price
            else:  # SELL
                pnl_percent = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if pnl_percent <= -self.stop_loss_percent:
                return {
                    'action': 'SELL' if side == 'BUY' else 'BUY',
                    'confidence': 1.0,
                    'reason': f'Stop loss triggered: {pnl_percent:.2%}',
                    'exit_type': 'STOP_LOSS'
                }
            
            # Check take profit
            if pnl_percent >= self.take_profit_percent:
                return {
                    'action': 'SELL' if side == 'BUY' else 'BUY',
                    'confidence': 1.0,
                    'reason': f'Take profit triggered: {pnl_percent:.2%}',
                    'exit_type': 'TAKE_PROFIT'
                }
            
            # Check if signals have reversed
            technical_signals = analysis.get('technical_signals', {})
            tech_signal = technical_signals.get('overall_signal', 'HOLD')
            tech_confidence = technical_signals.get('overall_confidence', 0.0)
            
            if tech_confidence > 0.8:  # Strong signal
                if side == 'BUY' and tech_signal == 'SELL':
                    return {
                        'action': 'SELL',
                        'confidence': tech_confidence,
                        'reason': f'Signal reversal: {tech_signal}',
                        'exit_type': 'SIGNAL_REVERSAL'
                    }
                elif side == 'SELL' and tech_signal == 'BUY':
                    return {
                        'action': 'BUY',
                        'confidence': tech_confidence,
                        'reason': f'Signal reversal: {tech_signal}',
                        'exit_type': 'SIGNAL_REVERSAL'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def execute_trade(self, decision: Dict) -> bool:
        """Execute a trading decision"""
        try:
            symbol = decision['symbol']
            action = decision['action']
            quantity = decision['quantity']
            price = decision['price']
            confidence = decision['confidence']
            reason = decision['reason']
            
            # Check maximum open trades limit
            if len(self.open_positions) >= self.max_open_trades and symbol not in self.open_positions:
                logger.warning(f"Maximum open trades limit reached: {self.max_open_trades}")
                return False
            
            if action == 'BUY':
                trade_info = paper_trader.place_buy_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    strategy_signal=reason,
                    prediction_confidence=confidence
                )
                
                if trade_info:
                    # Add to open positions
                    self.open_positions[symbol] = {
                        'trade_id': trade_info['id'],
                        'side': 'BUY',
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_time': datetime.now(),
                        'confidence': confidence,
                        'reason': reason
                    }
                    return True
                    
            elif action == 'SELL':
                # Check if we have a position to sell
                if symbol in self.open_positions:
                    position = self.open_positions[symbol]
                    quantity = position['quantity']
                
                trade_info = paper_trader.place_sell_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    strategy_signal=reason,
                    prediction_confidence=confidence
                )
                
                if trade_info:
                    # Remove from open positions if closing
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            portfolio_summary = paper_trader.get_portfolio_summary()
            performance_metrics = paper_trader.calculate_performance_metrics()
            
            status = {
                'portfolio_summary': portfolio_summary,
                'performance_metrics': performance_metrics,
                'open_positions': self.open_positions,
                'total_open_positions': len(self.open_positions),
                'max_open_trades': self.max_open_trades
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    def update_strategy_parameters(self, params: Dict):
        """Update strategy parameters"""
        try:
            if 'min_confidence' in params:
                self.min_confidence = params['min_confidence']
            if 'max_open_trades' in params:
                self.max_open_trades = params['max_open_trades']
            if 'stop_loss_percent' in params:
                self.stop_loss_percent = params['stop_loss_percent']
            if 'take_profit_percent' in params:
                self.take_profit_percent = params['take_profit_percent']
            
            logger.info("Strategy parameters updated")
            
        except Exception as e:
            logger.error(f"Error updating strategy parameters: {e}")

# Global trading strategy instance
trading_strategy = TradingStrategy()
