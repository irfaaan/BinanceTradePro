import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config import config
from analysis.technical_indicators import technical_analyzer
from analysis.ml_predictor import ml_predictor
from trading.live_trader import live_trader
from trading.paper_trader import paper_trader
from api.binance_client import binance_client
from models import Trade, Portfolio, MarketData
from utils.data_cache import data_cache

logger = logging.getLogger(__name__)

class AdvancedTradingStrategy:
    def __init__(self):
        self.config = config
        self.is_live_trading = not self.config.get('api_settings.use_testnet', True)
        self.trader = live_trader if self.is_live_trading else paper_trader
        
        # Advanced strategy parameters
        self.strategy_params = {
            'trend_strength_threshold': 0.7,
            'volume_surge_threshold': 1.5,
            'volatility_threshold': 0.02,
            'ml_confidence_threshold': 0.85,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'macd_signal_strength': 0.5,
            'bollinger_squeeze_threshold': 0.1,
            'momentum_lookback': 14,
            'news_sentiment_weight': 0.3,
            'market_regime_weight': 0.4
        }
        
        # Risk management
        self.risk_params = {
            'max_correlation_exposure': 0.3,
            'portfolio_heat': 0.0,
            'max_drawdown_threshold': 0.15,
            'volatility_position_sizing': True,
            'dynamic_stop_loss': True,
            'trailing_stop_enabled': True
        }
        
        # Market regime detection
        self.market_regimes = {
            'trending_bull': {'weight': 1.2, 'strategy': 'momentum'},
            'trending_bear': {'weight': 0.8, 'strategy': 'mean_reversion'},
            'ranging': {'weight': 1.0, 'strategy': 'range_trading'},
            'high_volatility': {'weight': 0.6, 'strategy': 'conservative'},
            'low_volatility': {'weight': 1.1, 'strategy': 'breakout'}
        }
        
        self.active_positions = {}
        self.last_analysis_time = {}
        
    def analyze_market_comprehensive(self, symbol: str) -> Dict:
        """Comprehensive market analysis combining multiple approaches"""
        try:
            # Get historical data
            df = self.get_market_data(symbol)
            if df is None or len(df) < 100:
                return {'error': 'Insufficient data for analysis'}
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'technical_analysis': {},
                'ml_prediction': {},
                'market_regime': {},
                'risk_assessment': {},
                'trading_signal': {},
                'confidence_score': 0.0
            }
            
            # Technical Analysis
            analysis['technical_analysis'] = self.perform_technical_analysis(df)
            
            # ML Prediction
            analysis['ml_prediction'] = self.get_ml_prediction(df, symbol)
            
            # Market Regime Detection
            analysis['market_regime'] = self.detect_market_regime(df)
            
            # Risk Assessment
            analysis['risk_assessment'] = self.assess_risk(symbol, df)
            
            # Generate Trading Signal
            analysis['trading_signal'] = self.generate_trading_signal(analysis)
            
            # Calculate Overall Confidence
            analysis['confidence_score'] = self.calculate_confidence_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive market analysis for {symbol}: {e}")
            return {'error': str(e)}
    
    def perform_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Advanced technical analysis with multiple indicators"""
        try:
            indicators = technical_analyzer.calculate_all_indicators(df)
            signals = technical_analyzer.get_trading_signals(df)
            
            # Additional advanced indicators
            advanced_indicators = {
                'trend_strength': self.calculate_trend_strength(df),
                'volume_profile': self.analyze_volume_profile(df),
                'volatility_regime': self.detect_volatility_regime(df),
                'momentum_divergence': self.detect_momentum_divergence(df),
                'support_resistance': self.identify_dynamic_sr_levels(df),
                'pattern_recognition': self.detect_chart_patterns(df)
            }
            
            return {
                'indicators': indicators,
                'signals': signals,
                'advanced': advanced_indicators
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def get_ml_prediction(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Get ML prediction with ensemble methods"""
        try:
            prediction = ml_predictor.predict(df, symbol)
            
            if prediction:
                # Enhance prediction with additional ML models
                enhanced_prediction = {
                    'base_prediction': prediction,
                    'ensemble_confidence': self.calculate_ensemble_confidence(df, symbol),
                    'volatility_prediction': self.predict_volatility(df),
                    'trend_continuation_probability': self.predict_trend_continuation(df),
                    'reversal_probability': self.predict_reversal_probability(df)
                }
                
                return enhanced_prediction
            
            return {'error': 'No ML prediction available'}
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return {'error': str(e)}
    
    def detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        try:
            # Calculate regime indicators
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            
            # Trend strength
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            trend_strength = (sma_20 - sma_50) / sma_50
            
            # Current values
            current_vol = volatility.iloc[-1]
            current_trend = trend_strength.iloc[-1]
            
            # Regime classification
            if current_vol > volatility.quantile(0.8):
                regime = 'high_volatility'
            elif current_vol < volatility.quantile(0.2):
                regime = 'low_volatility'
            elif current_trend > 0.05:
                regime = 'trending_bull'
            elif current_trend < -0.05:
                regime = 'trending_bear'
            else:
                regime = 'ranging'
            
            return {
                'regime': regime,
                'trend_strength': current_trend,
                'volatility_level': current_vol,
                'confidence': self.calculate_regime_confidence(df, regime)
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    def assess_risk(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Comprehensive risk assessment"""
        try:
            # Portfolio correlation
            correlation_risk = self.calculate_portfolio_correlation(symbol)
            
            # Volatility risk
            volatility_risk = self.calculate_volatility_risk(df)
            
            # Liquidity risk
            liquidity_risk = self.calculate_liquidity_risk(df)
            
            # Market impact risk
            market_impact_risk = self.calculate_market_impact_risk(symbol)
            
            # Overall risk score
            risk_score = (correlation_risk + volatility_risk + liquidity_risk + market_impact_risk) / 4
            
            return {
                'overall_risk_score': risk_score,
                'correlation_risk': correlation_risk,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'market_impact_risk': market_impact_risk,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            return {'overall_risk_score': 0.5, 'risk_level': 'MEDIUM'}
    
    def generate_trading_signal(self, analysis: Dict) -> Dict:
        """Generate trading signal based on comprehensive analysis"""
        try:
            technical = analysis.get('technical_analysis', {})
            ml_pred = analysis.get('ml_prediction', {})
            regime = analysis.get('market_regime', {})
            risk = analysis.get('risk_assessment', {})
            
            # Signal components
            technical_signal = self.extract_technical_signal(technical)
            ml_signal = self.extract_ml_signal(ml_pred)
            regime_signal = self.extract_regime_signal(regime)
            
            # Weight signals based on current market conditions
            weights = self.calculate_signal_weights(regime, risk)
            
            # Combined signal
            combined_signal = (
                technical_signal * weights['technical'] +
                ml_signal * weights['ml'] +
                regime_signal * weights['regime']
            )
            
            # Determine action
            if combined_signal > 0.6 and risk['overall_risk_score'] < 0.8:
                action = 'BUY'
            elif combined_signal < -0.6 and risk['overall_risk_score'] < 0.8:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            return {
                'action': action,
                'signal_strength': abs(combined_signal),
                'technical_signal': technical_signal,
                'ml_signal': ml_signal,
                'regime_signal': regime_signal,
                'weights': weights,
                'combined_signal': combined_signal
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {'action': 'HOLD', 'signal_strength': 0.0}
    
    def execute_advanced_trade(self, symbol: str, analysis: Dict) -> Optional[Dict]:
        """Execute trade with advanced position sizing and risk management"""
        try:
            signal = analysis['trading_signal']
            confidence = analysis['confidence_score']
            risk = analysis['risk_assessment']
            
            if signal['action'] == 'HOLD':
                return None
            
            # Calculate position size
            position_size = self.calculate_advanced_position_size(
                symbol, signal, confidence, risk
            )
            
            if position_size == 0:
                return None
            
            # Get current price
            current_price = binance_client.get_ticker_price(symbol)
            if not current_price:
                return None
            
            # Execute trade
            if signal['action'] == 'BUY':
                result = self.trader.place_buy_order(
                    symbol=symbol,
                    quantity=position_size,
                    price=current_price,
                    strategy_signal=f"Advanced Strategy (Confidence: {confidence:.2f})",
                    prediction_confidence=confidence
                )
            else:  # SELL
                result = self.trader.place_sell_order(
                    symbol=symbol,
                    quantity=position_size,
                    price=current_price,
                    strategy_signal=f"Advanced Strategy (Confidence: {confidence:.2f})",
                    prediction_confidence=confidence
                )
            
            if result:
                # Set stop loss and take profit
                self.set_advanced_risk_management(symbol, result, analysis)
                
                # Update position tracking
                self.update_position_tracking(symbol, result, analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing advanced trade for {symbol}: {e}")
            return None
    
    def calculate_advanced_position_size(self, symbol: str, signal: Dict, 
                                       confidence: float, risk: Dict) -> float:
        """Calculate position size using advanced risk management"""
        try:
            # Base position size
            base_size = self.trader.calculate_position_size(symbol, 0, confidence)
            
            # Risk adjustment
            risk_multiplier = 1.0 - risk['overall_risk_score']
            
            # Volatility adjustment
            volatility_adjustment = self.calculate_volatility_adjustment(symbol)
            
            # Correlation adjustment
            correlation_adjustment = self.calculate_correlation_adjustment(symbol)
            
            # Final position size
            final_size = base_size * risk_multiplier * volatility_adjustment * correlation_adjustment
            
            return max(0, final_size)
            
        except Exception as e:
            logger.error(f"Error calculating advanced position size: {e}")
            return 0
    
    def set_advanced_risk_management(self, symbol: str, trade_result: Dict, analysis: Dict):
        """Set advanced stop loss and take profit orders"""
        try:
            if not self.is_live_trading:
                return  # Skip for paper trading
            
            # Dynamic stop loss based on volatility
            volatility = analysis['technical_analysis']['advanced']['volatility_regime']
            stop_loss_pct = max(0.005, volatility * 2)  # Minimum 0.5% stop loss
            
            # Take profit based on expected return
            ml_prediction = analysis['ml_prediction']['base_prediction']
            expected_return = ml_prediction.get('expected_return', 0.02)
            take_profit_pct = min(0.1, expected_return * 1.5)  # Maximum 10% take profit
            
            current_price = float(trade_result['price'])
            
            if trade_result['side'] == 'BUY':
                stop_price = current_price * (1 - stop_loss_pct)
                target_price = current_price * (1 + take_profit_pct)
            else:
                stop_price = current_price * (1 + stop_loss_pct)
                target_price = current_price * (1 - take_profit_pct)
            
            # Set orders
            quantity = float(trade_result['executedQty'])
            
            # Stop loss
            live_trader.set_stop_loss(symbol, quantity, stop_price)
            
            # Take profit
            live_trader.set_take_profit(symbol, quantity, target_price)
            
        except Exception as e:
            logger.error(f"Error setting advanced risk management: {e}")
    
    def update_position_tracking(self, symbol: str, trade_result: Dict, analysis: Dict):
        """Update position tracking for portfolio management"""
        try:
            self.active_positions[symbol] = {
                'entry_time': datetime.utcnow(),
                'entry_price': float(trade_result['price']),
                'quantity': float(trade_result['executedQty']),
                'side': trade_result['side'],
                'analysis': analysis,
                'trade_id': trade_result.get('orderId')
            }
            
        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")
    
    def get_market_data(self, symbol: str, interval: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            # Try to get from cache first
            cache_key = f"market_data_{symbol}_{interval}_{limit}"
            cached_data = data_cache.get_cached_data(cache_key)
            
            if cached_data:
                return pd.DataFrame(cached_data)
            
            # Get from Binance
            klines = binance_client.get_historical_klines(symbol, interval, limit=limit)
            
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to proper types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Cache the data
            data_cache.cache_data(cache_key, df.to_dict('records'), expiry_minutes=15)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    # Helper methods for advanced analysis
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like calculation"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            dm_plus = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
            dm_minus = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
            
            # Smooth the values
            tr_smooth = tr.rolling(14).mean()
            dm_plus_smooth = dm_plus.rolling(14).mean()
            dm_minus_smooth = dm_minus.rolling(14).mean()
            
            # Calculate DI
            di_plus = (dm_plus_smooth / tr_smooth) * 100
            di_minus = (dm_minus_smooth / tr_smooth) * 100
            
            # Calculate ADX
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx = dx.rolling(14).mean()
            
            return adx.iloc[-1] / 100 if not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile for better entry/exit points"""
        try:
            # Simple volume analysis
            volume_ma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / volume_ma
            
            return {
                'volume_surge': volume_ratio.iloc[-1] > 1.5,
                'volume_ratio': volume_ratio.iloc[-1],
                'avg_volume': volume_ma.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {'volume_surge': False, 'volume_ratio': 1.0}
    
    def detect_volatility_regime(self, df: pd.DataFrame) -> float:
        """Detect volatility regime"""
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            
            return volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.02
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return 0.02
    
    def detect_momentum_divergence(self, df: pd.DataFrame) -> bool:
        """Detect momentum divergence"""
        try:
            # Simple momentum divergence detection
            price_highs = df['high'].rolling(10).max()
            price_lows = df['low'].rolling(10).min()
            
            # Calculate RSI for momentum
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Check for divergence (simplified)
            return (price_highs.iloc[-1] > price_highs.iloc[-5] and 
                   rsi.iloc[-1] < rsi.iloc[-5])
            
        except Exception as e:
            logger.error(f"Error detecting momentum divergence: {e}")
            return False
    
    def identify_dynamic_sr_levels(self, df: pd.DataFrame) -> Dict:
        """Identify dynamic support and resistance levels"""
        try:
            # Use pivot points for S/R
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate pivot points
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            
            return {
                'pivot': pivot.iloc[-1],
                'resistance_1': r1.iloc[-1],
                'support_1': s1.iloc[-1],
                'current_price': close.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error identifying S/R levels: {e}")
            return {}
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect basic chart patterns"""
        try:
            # Simple pattern detection
            close = df['close']
            sma_20 = close.rolling(20).mean()
            
            # Trend detection
            if close.iloc[-1] > sma_20.iloc[-1] and close.iloc[-5] < sma_20.iloc[-5]:
                pattern = 'breakout_bullish'
            elif close.iloc[-1] < sma_20.iloc[-1] and close.iloc[-5] > sma_20.iloc[-5]:
                pattern = 'breakout_bearish'
            else:
                pattern = 'sideways'
            
            return {
                'pattern': pattern,
                'confidence': 0.6  # Simplified confidence
            }
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
            return {'pattern': 'unknown', 'confidence': 0.0}
    
    def calculate_ensemble_confidence(self, df: pd.DataFrame, symbol: str) -> float:
        """Calculate ensemble confidence from multiple models"""
        try:
            # Simplified ensemble confidence
            base_confidence = 0.7
            
            # Adjust based on data quality
            if len(df) < 100:
                base_confidence *= 0.8
            
            # Adjust based on volatility
            volatility = df['close'].pct_change().std()
            if volatility > 0.05:
                base_confidence *= 0.9
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {e}")
            return 0.5
    
    def predict_volatility(self, df: pd.DataFrame) -> float:
        """Predict future volatility"""
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            
            # Simple volatility prediction (using historical average)
            return volatility.mean()
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return 0.02
    
    def predict_trend_continuation(self, df: pd.DataFrame) -> float:
        """Predict trend continuation probability"""
        try:
            # Simple trend continuation based on momentum
            close = df['close']
            momentum = close.pct_change(10)
            
            # Probability based on momentum strength
            if momentum.iloc[-1] > 0.1:
                return 0.8
            elif momentum.iloc[-1] < -0.1:
                return 0.2
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error predicting trend continuation: {e}")
            return 0.5
    
    def predict_reversal_probability(self, df: pd.DataFrame) -> float:
        """Predict reversal probability"""
        try:
            # Simple reversal probability based on RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 80:
                return 0.8  # High probability of bearish reversal
            elif current_rsi < 20:
                return 0.8  # High probability of bullish reversal
            else:
                return 0.2  # Low probability of reversal
                
        except Exception as e:
            logger.error(f"Error predicting reversal probability: {e}")
            return 0.2
    
    def calculate_portfolio_correlation(self, symbol: str) -> float:
        """Calculate portfolio correlation risk"""
        try:
            # Simplified correlation calculation
            return 0.3  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation: {e}")
            return 0.5
    
    def calculate_volatility_risk(self, df: pd.DataFrame) -> float:
        """Calculate volatility risk"""
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            
            # Normalize volatility to risk score
            vol_score = min(1.0, volatility.iloc[-1] / 0.1)
            
            return vol_score
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return 0.5
    
    def calculate_liquidity_risk(self, df: pd.DataFrame) -> float:
        """Calculate liquidity risk"""
        try:
            # Use volume as proxy for liquidity
            volume_ma = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-1]
            
            liquidity_score = 1.0 - min(1.0, current_volume / volume_ma.iloc[-1])
            
            return liquidity_score
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.3
    
    def calculate_market_impact_risk(self, symbol: str) -> float:
        """Calculate market impact risk"""
        try:
            # Simplified market impact calculation
            return 0.2  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating market impact risk: {e}")
            return 0.3
    
    def extract_technical_signal(self, technical: Dict) -> float:
        """Extract signal from technical analysis"""
        try:
            signals = technical.get('signals', {})
            
            signal_sum = 0
            signal_count = 0
            
            for signal_type, signal_value in signals.items():
                if isinstance(signal_value, (int, float)):
                    signal_sum += signal_value
                    signal_count += 1
            
            return signal_sum / signal_count if signal_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error extracting technical signal: {e}")
            return 0.0
    
    def extract_ml_signal(self, ml_pred: Dict) -> float:
        """Extract signal from ML prediction"""
        try:
            base_pred = ml_pred.get('base_prediction', {})
            direction = base_pred.get('direction', 'HOLD')
            
            if direction == 'UP':
                return 1.0
            elif direction == 'DOWN':
                return -1.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error extracting ML signal: {e}")
            return 0.0
    
    def extract_regime_signal(self, regime: Dict) -> float:
        """Extract signal from market regime"""
        try:
            regime_type = regime.get('regime', 'unknown')
            
            if regime_type == 'trending_bull':
                return 0.8
            elif regime_type == 'trending_bear':
                return -0.8
            elif regime_type == 'ranging':
                return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error extracting regime signal: {e}")
            return 0.0
    
    def calculate_signal_weights(self, regime: Dict, risk: Dict) -> Dict:
        """Calculate weights for different signal types"""
        try:
            risk_score = risk.get('overall_risk_score', 0.5)
            
            # Adjust weights based on risk
            if risk_score > 0.7:
                return {'technical': 0.5, 'ml': 0.3, 'regime': 0.2}
            elif risk_score > 0.4:
                return {'technical': 0.4, 'ml': 0.4, 'regime': 0.2}
            else:
                return {'technical': 0.3, 'ml': 0.5, 'regime': 0.2}
                
        except Exception as e:
            logger.error(f"Error calculating signal weights: {e}")
            return {'technical': 0.33, 'ml': 0.33, 'regime': 0.34}
    
    def calculate_confidence_score(self, analysis: Dict) -> float:
        """Calculate overall confidence score"""
        try:
            technical_confidence = 0.7  # Simplified
            ml_confidence = analysis.get('ml_prediction', {}).get('base_prediction', {}).get('confidence', 0.5)
            regime_confidence = analysis.get('market_regime', {}).get('confidence', 0.5)
            
            # Weight the confidences
            overall_confidence = (
                technical_confidence * 0.3 +
                ml_confidence * 0.5 +
                regime_confidence * 0.2
            )
            
            return min(1.0, overall_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def calculate_regime_confidence(self, df: pd.DataFrame, regime: str) -> float:
        """Calculate confidence in regime classification"""
        try:
            # Simplified regime confidence
            if regime in ['trending_bull', 'trending_bear']:
                return 0.8
            elif regime in ['ranging', 'high_volatility']:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility adjustment for position sizing"""
        try:
            # Simplified volatility adjustment
            return 0.9  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation adjustment for position sizing"""
        try:
            # Simplified correlation adjustment
            return 0.95  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0

# Global instance
advanced_strategy = AdvancedTradingStrategy()