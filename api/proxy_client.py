import os
import requests
import logging
from typing import Dict, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class ProxyBinanceClient:
    """Binance client with proxy support for restricted regions"""
    
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.secret_key = os.environ.get('BINANCE_SECRET_KEY')
        self.use_testnet = False  # Live trading mode
        
        # Proxy configuration (you can set these as environment variables)
        self.proxy_config = {
            'http': os.environ.get('HTTP_PROXY'),
            'https': os.environ.get('HTTPS_PROXY')
        }
        
        # Alternative: Use public proxies (not recommended for production)
        if not self.proxy_config['https']:
            self.proxy_config = None
        
        self.client = None
        self.session = None
        
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Binance client with proxy support"""
        try:
            # Create custom session with proxy
            session = requests.Session()
            
            if self.proxy_config:
                session.proxies.update(self.proxy_config)
                logger.info("Using proxy configuration for Binance API")
            
            # Initialize Binance client
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.secret_key,
                testnet=self.use_testnet
            )
            
            # Test connection
            self.client.ping()
            logger.info("Successfully connected to Binance API")
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            self.client = None
            
            # Try alternative solutions
            self.try_alternative_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            self.client = None
    
    def try_alternative_connection(self):
        """Try alternative connection methods"""
        try:
            # Method 1: Try different base URLs
            alternative_urls = [
                'https://api.binance.com',
                'https://api1.binance.com',
                'https://api2.binance.com',
                'https://api3.binance.com'
            ]
            
            for url in alternative_urls:
                try:
                    logger.info(f"Trying alternative URL: {url}")
                    
                    # Create session with custom base URL
                    session = requests.Session()
                    if self.proxy_config:
                        session.proxies.update(self.proxy_config)
                    
                    # Test connection to alternative URL
                    response = session.get(f"{url}/api/v3/ping", timeout=10)
                    if response.status_code == 200:
                        logger.info(f"Successfully connected via {url}")
                        
                        # Initialize client with custom base URL
                        self.client = Client(
                            api_key=self.api_key,
                            api_secret=self.secret_key,
                            testnet=self.use_testnet,
                            requests_params={'session': session}
                        )
                        
                        # Override base URL
                        self.client.API_URL = url
                        
                        return
                        
                except Exception as e:
                    logger.debug(f"Failed to connect via {url}: {e}")
                    continue
            
            # Method 2: Use VPN recommendation
            logger.warning("Direct connection failed. Consider using a VPN service.")
            self.recommend_vpn_setup()
            
        except Exception as e:
            logger.error(f"All alternative connection methods failed: {e}")
    
    def recommend_vpn_setup(self):
        """Provide VPN setup recommendations"""
        vpn_message = """
        🔧 CONNECTION ISSUE DETECTED 🔧
        
        The Binance API is not accessible from your current location.
        
        RECOMMENDED SOLUTIONS:
        1. Use a VPN service (ExpressVPN, NordVPN, etc.)
        2. Connect to a server in a supported region
        3. Set proxy environment variables:
           - HTTP_PROXY=http://proxy:port
           - HTTPS_PROXY=https://proxy:port
        
        Once connected, restart the application.
        """
        logger.warning(vpn_message)
    
    def is_connected(self) -> bool:
        """Check if client is connected"""
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
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current ticker price"""
        if not self.client:
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def get_24hr_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker statistics"""
        if not self.client:
            return None
        try:
            return self.client.get_ticker(symbol=symbol)
        except Exception as e:
            logger.error(f"Failed to get 24hr ticker for {symbol}: {e}")
            return None
    
    def get_historical_klines(self, symbol: str, interval: str, start_str: str = None, 
                             end_str: str = None, limit: int = 1000):
        """Get historical kline data"""
        if not self.client:
            return []
        try:
            return self.client.get_historical_klines(
                symbol, interval, start_str, end_str, limit
            )
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    def get_recent_trades(self, symbol: str, limit: int = 100):
        """Get recent trades"""
        if not self.client:
            return []
        try:
            return self.client.get_recent_trades(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []
    
    def get_order_book(self, symbol: str, limit: int = 100):
        """Get order book"""
        if not self.client:
            return None
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.client:
            return None
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
            return None
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def validate_trading_pair(self, symbol: str) -> bool:
        """Validate if trading pair is available"""
        symbol_info = self.get_symbol_info(symbol)
        return symbol_info is not None and symbol_info['status'] == 'TRADING'

# Global instance
proxy_binance_client = ProxyBinanceClient()