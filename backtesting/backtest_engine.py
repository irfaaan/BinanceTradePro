import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from analysis.technical_indicators import technical_analyzer
from analysis.ml_predictor import ml_predictor
from api.binance_client import binance_client
from config import config

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self):
        self.initial_balance = config.get('paper_trading.initial_balance', 10000.0)
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage
        
    def prepare_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1h') -> pd.DataFrame:
        """Prepare historical data for backtesting"""
        try:
            # Get historical data
            klines = binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date,
                limit=1000
            )
            
            if not klines:
                logger.error(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
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
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only necessary columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Prepared {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, strategy_params: Dict = None) -> Dict:
        """Run backtest for a specific symbol"""
        try:
            # Prepare data
            df = self.prepare_data(symbol, start_date, end_date)
            if df.empty:
                return {}
            
            # Initialize backtest state
            portfolio = {
                'cash': self.initial_balance,
                'positions': {},
                'total_value': self.initial_balance
            }
            
            trades = []
            portfolio_history = []
            
            # Strategy parameters
            if strategy_params is None:
                strategy_params = {
                    'min_confidence': 0.7,
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'position_size': 0.1  # 10% of portfolio
                }
            
            # Calculate indicators for the entire dataset
            indicators = technical_analyzer.calculate_all_indicators(df)
            
            # Run backtest
            for i in range(50, len(df)):  # Start from 50 to have enough data for indicators
                current_time = df.index[i]
                current_price = df['close'].iloc[i]
                
                # Get current market data slice
                current_df = df.iloc[:i+1]
                
                # Generate trading signals
                signals = technical_analyzer.get_trading_signals(current_df)
                
                # Check for entry signals
                if symbol not in portfolio['positions']:
                    entry_signal = self._check_entry_signal(signals, strategy_params)
                    if entry_signal:
                        trade = self._execute_entry(
                            symbol, current_price, current_time, 
                            entry_signal, portfolio, strategy_params
                        )
                        if trade:
                            trades.append(trade)
                
                # Check for exit signals
                else:
                    exit_signal = self._check_exit_signal(
                        signals, portfolio['positions'][symbol], 
                        current_price, strategy_params
                    )
                    if exit_signal:
                        trade = self._execute_exit(
                            symbol, current_price, current_time, 
                            exit_signal, portfolio
                        )
                        if trade:
                            trades.append(trade)
                
                # Update portfolio value
                portfolio['total_value'] = self._calculate_portfolio_value(portfolio, {symbol: current_price})
                
                # Record portfolio history
                portfolio_history.append({
                    'timestamp': current_time,
                    'total_value': portfolio['total_value'],
                    'cash': portfolio['cash'],
                    'positions': portfolio['positions'].copy()
                })
            
            # Calculate performance metrics
            performance = self._calculate_performance(trades, portfolio_history)
            
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': self.initial_balance,
                'final_balance': portfolio['total_value'],
                'trades': trades,
                'portfolio_history': portfolio_history,
                'performance': performance,
                'strategy_params': strategy_params
            }
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            return {}
    
    def _check_entry_signal(self, signals: Dict, strategy_params: Dict) -> Optional[str]:
        """Check for entry signals"""
        try:
            overall_signal = signals.get('overall_signal', 'HOLD')
            overall_confidence = signals.get('overall_confidence', 0.0)
            
            min_confidence = strategy_params.get('min_confidence', 0.7)
            
            if overall_signal == 'BUY' and overall_confidence >= min_confidence:
                return 'BUY'
            elif overall_signal == 'SELL' and overall_confidence >= min_confidence:
                return 'SELL'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking entry signal: {e}")
            return None
    
    def _check_exit_signal(self, signals: Dict, position: Dict, current_price: float, strategy_params: Dict) -> Optional[str]:
        """Check for exit signals"""
        try:
            entry_price = position['entry_price']
            side = position['side']
            
            # Calculate P&L
            if side == 'BUY':
                pnl_percent = (current_price - entry_price) / entry_price
            else:  # SELL (short position)
                pnl_percent = (entry_price - current_price) / entry_price
            
            # Check stop loss
            stop_loss = strategy_params.get('stop_loss', 0.02)
            if pnl_percent <= -stop_loss:
                return 'STOP_LOSS'
            
            # Check take profit
            take_profit = strategy_params.get('take_profit', 0.04)
            if pnl_percent >= take_profit:
                return 'TAKE_PROFIT'
            
            # Check signal reversal
            overall_signal = signals.get('overall_signal', 'HOLD')
            overall_confidence = signals.get('overall_confidence', 0.0)
            
            if overall_confidence > 0.8:  # Strong signal
                if side == 'BUY' and overall_signal == 'SELL':
                    return 'SIGNAL_REVERSAL'
                elif side == 'SELL' and overall_signal == 'BUY':
                    return 'SIGNAL_REVERSAL'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return None
    
    def _execute_entry(self, symbol: str, price: float, timestamp: datetime, 
                      signal: str, portfolio: Dict, strategy_params: Dict) -> Optional[Dict]:
        """Execute entry trade"""
        try:
            position_size = strategy_params.get('position_size', 0.1)
            
            # Calculate position value
            position_value = portfolio['total_value'] * position_size
            
            # Apply slippage
            executed_price = price * (1 + self.slippage if signal == 'BUY' else 1 - self.slippage)
            
            # Calculate quantity
            quantity = position_value / executed_price
            
            # Calculate commission
            commission = position_value * self.commission
            
            # Check if we have enough cash
            total_cost = position_value + commission
            if total_cost > portfolio['cash']:
                return None
            
            # Update portfolio
            portfolio['cash'] -= total_cost
            portfolio['positions'][symbol] = {
                'side': signal,
                'quantity': quantity,
                'entry_price': executed_price,
                'entry_time': timestamp,
                'commission_paid': commission
            }
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'side': signal,
                'quantity': quantity,
                'price': executed_price,
                'timestamp': timestamp,
                'trade_type': 'ENTRY',
                'commission': commission,
                'total_value': position_value
            }
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing entry: {e}")
            return None
    
    def _execute_exit(self, symbol: str, price: float, timestamp: datetime, 
                     signal: str, portfolio: Dict) -> Optional[Dict]:
        """Execute exit trade"""
        try:
            position = portfolio['positions'][symbol]
            
            # Apply slippage
            executed_price = price * (1 - self.slippage if position['side'] == 'BUY' else 1 + self.slippage)
            
            # Calculate trade value
            trade_value = position['quantity'] * executed_price
            commission = trade_value * self.commission
            
            # Update portfolio
            portfolio['cash'] += trade_value - commission
            
            # Calculate P&L
            entry_value = position['quantity'] * position['entry_price']
            total_commission = position['commission_paid'] + commission
            
            if position['side'] == 'BUY':
                pnl = trade_value - entry_value - total_commission
            else:  # SELL (short position)
                pnl = entry_value - trade_value - total_commission
            
            # Remove position
            del portfolio['positions'][symbol]
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'side': 'SELL' if position['side'] == 'BUY' else 'BUY',
                'quantity': position['quantity'],
                'price': executed_price,
                'timestamp': timestamp,
                'trade_type': 'EXIT',
                'exit_reason': signal,
                'commission': commission,
                'total_value': trade_value,
                'pnl': pnl,
                'entry_price': position['entry_price'],
                'entry_time': position['entry_time']
            }
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing exit: {e}")
            return None
    
    def _calculate_portfolio_value(self, portfolio: Dict, prices: Dict) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = portfolio['cash']
            
            for symbol, position in portfolio['positions'].items():
                if symbol in prices:
                    position_value = position['quantity'] * prices[symbol]
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def _calculate_performance(self, trades: List[Dict], portfolio_history: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        try:
            if not trades or not portfolio_history:
                return {}
            
            # Basic metrics
            total_trades = len([t for t in trades if t['trade_type'] == 'EXIT'])
            winning_trades = len([t for t in trades if t['trade_type'] == 'EXIT' and t.get('pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            
            # Win rate
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Total P&L
            total_pnl = sum(t.get('pnl', 0) for t in trades if t['trade_type'] == 'EXIT')
            
            # Portfolio value evolution
            initial_value = self.initial_balance
            final_value = portfolio_history[-1]['total_value']
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            # Calculate returns for each period
            returns = []
            for i in range(1, len(portfolio_history)):
                prev_value = portfolio_history[i-1]['total_value']
                curr_value = portfolio_history[i]['total_value']
                if prev_value > 0:
                    period_return = (curr_value - prev_value) / prev_value
                    returns.append(period_return)
            
            # Risk metrics
            if returns:
                returns_array = np.array(returns)
                volatility = np.std(returns_array) * np.sqrt(365 * 24)  # Assuming hourly data
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(365 * 24) if np.std(returns_array) > 0 else 0
                
                # Maximum drawdown
                portfolio_values = [h['total_value'] for h in portfolio_history]
                peak = portfolio_values[0]
                max_drawdown = 0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                max_drawdown_percent = max_drawdown * 100
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown_percent = 0
            
            # Average trade metrics
            exit_trades = [t for t in trades if t['trade_type'] == 'EXIT']
            avg_win = np.mean([t['pnl'] for t in exit_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in exit_trades if t['pnl'] < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
            
            performance = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'initial_balance': initial_value,
                'final_balance': final_value,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown_percent,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    def run_multi_symbol_backtest(self, symbols: List[str], start_date: str, end_date: str, strategy_params: Dict = None) -> Dict:
        """Run backtest for multiple symbols"""
        try:
            results = {}
            
            for symbol in symbols:
                logger.info(f"Running backtest for {symbol}")
                result = self.run_backtest(symbol, start_date, end_date, strategy_params)
                if result:
                    results[symbol] = result
            
            # Calculate combined performance
            if results:
                combined_performance = self._calculate_combined_performance(results)
                results['combined'] = combined_performance
            
            return results
            
        except Exception as e:
            logger.error(f"Error running multi-symbol backtest: {e}")
            return {}
    
    def _calculate_combined_performance(self, results: Dict) -> Dict:
        """Calculate combined performance across all symbols"""
        try:
            total_initial = sum(r['initial_balance'] for r in results.values())
            total_final = sum(r['final_balance'] for r in results.values())
            total_trades = sum(r['performance'].get('total_trades', 0) for r in results.values())
            total_winning = sum(r['performance'].get('winning_trades', 0) for r in results.values())
            
            combined_return = ((total_final - total_initial) / total_initial) * 100
            combined_win_rate = (total_winning / total_trades) * 100 if total_trades > 0 else 0
            
            return {
                'total_initial_balance': total_initial,
                'total_final_balance': total_final,
                'total_return': combined_return,
                'total_trades': total_trades,
                'total_winning_trades': total_winning,
                'combined_win_rate': combined_win_rate,
                'symbols_tested': list(results.keys())
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined performance: {e}")
            return {}

# Global backtest engine instance
backtest_engine = BacktestEngine()
