import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class DemoDataGenerator:
    """Generate realistic demo data for the trading bot when Binance API is not available"""
    
    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT']
        self.base_prices = {
            'BTCUSDT': 43000,
            'ETHUSDT': 2600,
            'BNBUSDT': 320,
            'ADAUSDT': 0.52,
            'SOLUSDT': 98,
            'XRPUSDT': 0.61
        }
        self.price_volatility = {
            'BTCUSDT': 0.02,
            'ETHUSDT': 0.025,
            'BNBUSDT': 0.03,
            'ADAUSDT': 0.04,
            'SOLUSDT': 0.035,
            'XRPUSDT': 0.05
        }
        self.last_prices = self.base_prices.copy()
        self.last_update = time.time()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price with realistic fluctuation"""
        if symbol not in self.base_prices:
            return None
        
        base_price = self.base_prices[symbol]
        volatility = self.price_volatility[symbol]
        
        # Add random walk
        change = random.uniform(-volatility, volatility)
        new_price = self.last_prices[symbol] * (1 + change)
        
        # Keep price within reasonable bounds
        max_price = base_price * 1.2
        min_price = base_price * 0.8
        new_price = max(min_price, min(max_price, new_price))
        
        self.last_prices[symbol] = new_price
        return round(new_price, 8)
    
    def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        if symbol not in self.base_prices:
            return None
        
        current_price = self.get_current_price(symbol)
        base_price = self.base_prices[symbol]
        
        # Generate realistic 24hr data
        high_price = current_price * random.uniform(1.01, 1.05)
        low_price = current_price * random.uniform(0.95, 0.99)
        open_price = current_price * random.uniform(0.98, 1.02)
        
        price_change = current_price - open_price
        price_change_percent = (price_change / open_price) * 100
        
        volume = random.uniform(10000, 100000)
        
        return {
            'symbol': symbol,
            'price': str(current_price),
            'priceChange': str(price_change),
            'priceChangePercent': str(price_change_percent),
            'highPrice': str(high_price),
            'lowPrice': str(low_price),
            'openPrice': str(open_price),
            'volume': str(volume),
            'count': random.randint(50000, 200000)
        }
    
    def get_historical_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[List]:
        """Generate historical kline data"""
        if symbol not in self.base_prices:
            return []
        
        klines = []
        current_price = self.base_prices[symbol]
        volatility = self.price_volatility[symbol]
        
        # Generate historical data
        for i in range(limit):
            timestamp = int((datetime.now() - timedelta(hours=limit-i)).timestamp() * 1000)
            
            # Generate OHLCV data
            open_price = current_price
            
            # Generate realistic price movement
            change = random.uniform(-volatility, volatility)
            close_price = open_price * (1 + change)
            
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.01)
            low_price = min(open_price, close_price) * random.uniform(0.99, 1.0)
            
            volume = random.uniform(100, 1000)
            
            kline = [
                timestamp,
                str(open_price),
                str(high_price),
                str(low_price),
                str(close_price),
                str(volume),
                timestamp + 3600000,  # close_time
                str(volume * current_price),  # quote_volume
                random.randint(100, 500),  # trades
                str(volume * 0.6),  # taker_buy_base
                str(volume * current_price * 0.6),  # taker_buy_quote
                '0'  # ignore
            ]
            
            klines.append(kline)
            current_price = close_price
        
        return klines
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Generate order book data"""
        if symbol not in self.base_prices:
            return None
        
        current_price = self.get_current_price(symbol)
        
        # Generate bids and asks
        bids = []
        asks = []
        
        for i in range(min(limit, 20)):
            # Bids (buy orders) - below current price
            bid_price = current_price * (1 - (i + 1) * 0.001)
            bid_quantity = random.uniform(0.1, 10.0)
            bids.append([str(bid_price), str(bid_quantity)])
            
            # Asks (sell orders) - above current price
            ask_price = current_price * (1 + (i + 1) * 0.001)
            ask_quantity = random.uniform(0.1, 10.0)
            asks.append([str(ask_price), str(ask_quantity)])
        
        return {
            'lastUpdateId': random.randint(1000000, 9999999),
            'bids': bids,
            'asks': asks
        }
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Generate recent trades data"""
        if symbol not in self.base_prices:
            return []
        
        current_price = self.get_current_price(symbol)
        trades = []
        
        for i in range(min(limit, 50)):
            trade_price = current_price * random.uniform(0.999, 1.001)
            trade_quantity = random.uniform(0.01, 5.0)
            
            trade = {
                'id': random.randint(100000, 999999),
                'price': str(trade_price),
                'qty': str(trade_quantity),
                'time': int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000),
                'isBuyerMaker': random.choice([True, False])
            }
            trades.append(trade)
        
        return trades
    
    def get_account_info(self) -> Dict[str, Any]:
        """Generate demo account info"""
        return {
            'makerCommission': 10,
            'takerCommission': 10,
            'buyerCommission': 0,
            'sellerCommission': 0,
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True,
            'updateTime': int(time.time() * 1000),
            'accountType': 'SPOT',
            'balances': [
                {
                    'asset': 'USDT',
                    'free': '10000.00000000',
                    'locked': '0.00000000'
                },
                {
                    'asset': 'BTC',
                    'free': '0.50000000',
                    'locked': '0.00000000'
                },
                {
                    'asset': 'ETH',
                    'free': '5.00000000',
                    'locked': '0.00000000'
                }
            ]
        }
    
    def generate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Generate demo technical indicators"""
        return {
            'rsi': random.uniform(30, 70),
            'macd': {
                'macd': random.uniform(-0.5, 0.5),
                'signal': random.uniform(-0.5, 0.5),
                'histogram': random.uniform(-0.2, 0.2)
            },
            'bollinger_bands': {
                'upper': self.get_current_price(symbol) * 1.02,
                'middle': self.get_current_price(symbol),
                'lower': self.get_current_price(symbol) * 0.98
            },
            'sma_20': self.get_current_price(symbol) * random.uniform(0.98, 1.02),
            'sma_50': self.get_current_price(symbol) * random.uniform(0.95, 1.05),
            'ema_12': self.get_current_price(symbol) * random.uniform(0.99, 1.01),
            'ema_26': self.get_current_price(symbol) * random.uniform(0.97, 1.03),
            'volume_sma': random.uniform(1000, 5000),
            'atr': self.get_current_price(symbol) * random.uniform(0.01, 0.03)
        }
    
    def generate_ml_prediction(self, symbol: str) -> Dict[str, Any]:
        """Generate demo ML prediction"""
        confidence = random.uniform(0.6, 0.95)
        direction = random.choice(['UP', 'DOWN'])
        current_price = self.get_current_price(symbol)
        
        if direction == 'UP':
            predicted_price = current_price * random.uniform(1.01, 1.05)
        else:
            predicted_price = current_price * random.uniform(0.95, 0.99)
        
        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'expected_return': (predicted_price - current_price) / current_price,
            'model_accuracy': random.uniform(0.7, 0.9),
            'features_used': 45,
            'prediction_timeframe': '1h'
        }
    
    def generate_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate demo trading signals"""
        indicators = self.generate_technical_indicators(symbol)
        ml_pred = self.generate_ml_prediction(symbol)
        
        # Generate combined signal
        signal_strength = random.uniform(0.3, 0.9)
        action = random.choice(['BUY', 'SELL', 'HOLD'])
        
        return {
            'symbol': symbol,
            'action': action,
            'signal_strength': signal_strength,
            'confidence': random.uniform(0.6, 0.9),
            'technical_score': random.uniform(0.4, 0.8),
            'ml_score': ml_pred['confidence'],
            'volume_score': random.uniform(0.5, 0.9),
            'trend_score': random.uniform(0.4, 0.8),
            'risk_score': random.uniform(0.2, 0.6),
            'timestamp': datetime.now().isoformat()
        }

# Global demo data generator
demo_data = DemoDataGenerator()