import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
from datetime import datetime, timedelta
from config import config

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.api_key = config.binance_api_key
        self.secret_key = config.binance_secret_key
        self.testnet = config.use_testnet
        
        if not self.api_key or not self.secret_key:
            logger.warning("Binance API credentials not found. Using demo mode.")
            self.client = None
        else:
            try:
                self.client = Client(
                    self.api_key, 
                    self.secret_key, 
                    testnet=self.testnet
                )
                logger.info(f"Binance client initialized (testnet: {self.testnet})")
            except Exception as e:
                logger.error(f"Failed to initialize Binance client: {e}")
                self.client = None
    
    def is_connected(self) -> bool:
        """Check if client is properly connected"""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.client:
            return None
        try:
            return self.client.get_account()
        except BinanceAPIException as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current ticker price for a symbol"""
        if not self.client:
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def get_24hr_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker statistics"""
        if not self.client:
            return None
        try:
            return self.client.get_ticker(symbol=symbol)
        except BinanceAPIException as e:
            logger.error(f"Failed to get 24hr ticker for {symbol}: {e}")
            return None
    
    def get_historical_klines(self, symbol: str, interval: str, start_str: str = None, end_str: str = None, limit: int = 1000) -> List[List]:
        """Get historical kline/candlestick data"""
        if not self.client:
            return []
        try:
            return self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
        except BinanceAPIException as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol"""
        if not self.client:
            return []
        try:
            return self.client.get_recent_trades(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get order book for a symbol"""
        if not self.client:
            return None
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None
    
    def create_test_order(self, symbol: str, side: str, type: str, quantity: float, price: float = None) -> Optional[Dict]:
        """Create a test order (paper trading)"""
        if not self.client:
            return None
        try:
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': type,
                'quantity': quantity,
            }
            if price:
                order_params['price'] = price
                order_params['timeInForce'] = TIME_IN_FORCE_GTC
            
            return self.client.create_test_order(**order_params)
        except BinanceAPIException as e:
            logger.error(f"Failed to create test order for {symbol}: {e}")
            return None
    
    def get_exchange_info(self) -> Optional[Dict]:
        """Get exchange information"""
        if not self.client:
            return None
        try:
            return self.client.get_exchange_info()
        except BinanceAPIException as e:
            logger.error(f"Failed to get exchange info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get specific symbol information"""
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            return None
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
        return None
    
    def validate_trading_pair(self, symbol: str) -> bool:
        """Validate if a trading pair is available"""
        symbol_info = self.get_symbol_info(symbol)
        return symbol_info is not None and symbol_info['status'] == 'TRADING'

# Global client instance
binance_client = BinanceClient()
