import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators for the given dataframe"""
        if df.empty or len(df) < 50:
            return {}
        
        try:
            indicators = {}
            
            # Price and volume data
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            # Moving Averages
            indicators['sma_10'] = ta.trend.sma_indicator(close, window=10)
            indicators['sma_20'] = ta.trend.sma_indicator(close, window=20)
            indicators['sma_50'] = ta.trend.sma_indicator(close, window=50)
            indicators['ema_12'] = ta.trend.ema_indicator(close, window=12)
            indicators['ema_26'] = ta.trend.ema_indicator(close, window=26)
            
            # RSI
            indicators['rsi'] = ta.momentum.rsi(close, window=14)
            
            # MACD
            macd = ta.trend.MACD(close)
            indicators['macd'] = macd.macd()
            indicators['macd_signal'] = macd.macd_signal()
            indicators['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close)
            indicators['bb_high'] = bollinger.bollinger_hband()
            indicators['bb_low'] = bollinger.bollinger_lband()
            indicators['bb_mid'] = bollinger.bollinger_mavg()
            indicators['bb_width'] = bollinger.bollinger_wband()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            indicators['stoch_k'] = stoch.stoch()
            indicators['stoch_d'] = stoch.stoch_signal()
            
            # On-Balance Volume
            indicators['obv'] = ta.volume.on_balance_volume(close, volume)
            
            # Average True Range
            indicators['atr'] = ta.volatility.average_true_range(high, low, close)
            
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(high, low)
            indicators['ichimoku_a'] = ichimoku.ichimoku_a()
            indicators['ichimoku_b'] = ichimoku.ichimoku_b()
            indicators['ichimoku_base'] = ichimoku.ichimoku_base_line()
            indicators['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            
            # Williams %R
            indicators['williams_r'] = ta.momentum.williams_r(high, low, close)
            
            # Commodity Channel Index
            indicators['cci'] = ta.trend.cci(high, low, close)
            
            # Money Flow Index
            indicators['mfi'] = ta.volume.money_flow_index(high, low, close, volume)
            
            # Volume Weighted Average Price
            indicators['vwap'] = ta.volume.volume_weighted_average_price(high, low, close, volume)
            
            # Awesome Oscillator
            indicators['ao'] = ta.momentum.awesome_oscillator(high, low)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def get_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on technical indicators"""
        if df.empty or len(df) < 50:
            return {}
        
        try:
            signals = {}
            indicators = self.calculate_all_indicators(df)
            
            if not indicators:
                return {}
            
            # Get latest values
            latest_idx = -1
            
            # RSI Signals
            rsi_value = indicators['rsi'].iloc[latest_idx]
            if rsi_value < 30:
                signals['rsi_signal'] = 'BUY'
                signals['rsi_strength'] = (30 - rsi_value) / 30
            elif rsi_value > 70:
                signals['rsi_signal'] = 'SELL'
                signals['rsi_strength'] = (rsi_value - 70) / 30
            else:
                signals['rsi_signal'] = 'HOLD'
                signals['rsi_strength'] = 0
            
            # MACD Signals
            macd_value = indicators['macd'].iloc[latest_idx]
            macd_signal = indicators['macd_signal'].iloc[latest_idx]
            macd_hist = indicators['macd_histogram'].iloc[latest_idx]
            
            if macd_value > macd_signal and macd_hist > 0:
                signals['macd_signal'] = 'BUY'
                signals['macd_strength'] = min(abs(macd_hist) / abs(macd_value), 1.0)
            elif macd_value < macd_signal and macd_hist < 0:
                signals['macd_signal'] = 'SELL'
                signals['macd_strength'] = min(abs(macd_hist) / abs(macd_value), 1.0)
            else:
                signals['macd_signal'] = 'HOLD'
                signals['macd_strength'] = 0
            
            # Bollinger Bands Signals
            close_price = df['close'].iloc[latest_idx]
            bb_high = indicators['bb_high'].iloc[latest_idx]
            bb_low = indicators['bb_low'].iloc[latest_idx]
            bb_mid = indicators['bb_mid'].iloc[latest_idx]
            
            if close_price <= bb_low:
                signals['bb_signal'] = 'BUY'
                signals['bb_strength'] = (bb_mid - close_price) / (bb_mid - bb_low)
            elif close_price >= bb_high:
                signals['bb_signal'] = 'SELL'
                signals['bb_strength'] = (close_price - bb_mid) / (bb_high - bb_mid)
            else:
                signals['bb_signal'] = 'HOLD'
                signals['bb_strength'] = 0
            
            # Moving Average Signals
            sma_20 = indicators['sma_20'].iloc[latest_idx]
            sma_50 = indicators['sma_50'].iloc[latest_idx]
            
            if sma_20 > sma_50 and close_price > sma_20:
                signals['ma_signal'] = 'BUY'
                signals['ma_strength'] = (close_price - sma_20) / sma_20
            elif sma_20 < sma_50 and close_price < sma_20:
                signals['ma_signal'] = 'SELL'
                signals['ma_strength'] = (sma_20 - close_price) / sma_20
            else:
                signals['ma_signal'] = 'HOLD'
                signals['ma_strength'] = 0
            
            # Stochastic Signals
            stoch_k = indicators['stoch_k'].iloc[latest_idx]
            stoch_d = indicators['stoch_d'].iloc[latest_idx]
            
            if stoch_k < 20 and stoch_k > stoch_d:
                signals['stoch_signal'] = 'BUY'
                signals['stoch_strength'] = (20 - stoch_k) / 20
            elif stoch_k > 80 and stoch_k < stoch_d:
                signals['stoch_signal'] = 'SELL'
                signals['stoch_strength'] = (stoch_k - 80) / 20
            else:
                signals['stoch_signal'] = 'HOLD'
                signals['stoch_strength'] = 0
            
            # Overall Signal Consensus
            buy_signals = sum(1 for s in [signals.get('rsi_signal'), signals.get('macd_signal'), 
                                        signals.get('bb_signal'), signals.get('ma_signal'), 
                                        signals.get('stoch_signal')] if s == 'BUY')
            
            sell_signals = sum(1 for s in [signals.get('rsi_signal'), signals.get('macd_signal'), 
                                         signals.get('bb_signal'), signals.get('ma_signal'), 
                                         signals.get('stoch_signal')] if s == 'SELL')
            
            total_signals = buy_signals + sell_signals
            
            if buy_signals > sell_signals and buy_signals >= 3:
                signals['overall_signal'] = 'BUY'
                signals['overall_confidence'] = buy_signals / 5
            elif sell_signals > buy_signals and sell_signals >= 3:
                signals['overall_signal'] = 'SELL'
                signals['overall_confidence'] = sell_signals / 5
            else:
                signals['overall_signal'] = 'HOLD'
                signals['overall_confidence'] = 0
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {}
    
    def get_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        try:
            if len(df) < window * 2:
                return {}
            
            high_prices = df['high']
            low_prices = df['low']
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(high_prices) - window):
                # Check for resistance (local maxima)
                if high_prices.iloc[i] == high_prices.iloc[i-window:i+window+1].max():
                    resistance_levels.append(high_prices.iloc[i])
                
                # Check for support (local minima)
                if low_prices.iloc[i] == low_prices.iloc[i-window:i+window+1].min():
                    support_levels.append(low_prices.iloc[i])
            
            # Get the most significant levels
            current_price = df['close'].iloc[-1]
            
            # Filter and sort resistance levels
            resistance_above = sorted([r for r in resistance_levels if r > current_price])
            resistance_below = sorted([r for r in resistance_levels if r <= current_price], reverse=True)
            
            # Filter and sort support levels
            support_below = sorted([s for s in support_levels if s < current_price], reverse=True)
            support_above = sorted([s for s in support_levels if s >= current_price])
            
            return {
                'current_price': current_price,
                'nearest_resistance': resistance_above[0] if resistance_above else None,
                'nearest_support': support_below[0] if support_below else None,
                'all_resistance': resistance_above[:3],  # Top 3
                'all_support': support_below[:3]  # Top 3
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}

# Global technical analyzer instance
technical_analyzer = TechnicalAnalyzer()
