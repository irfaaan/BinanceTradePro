import asyncio
import json
import logging
from typing import Dict, List, Callable, Optional
from binance import AsyncClient, BinanceSocketManager
from config import config

logger = logging.getLogger(__name__)

class WebSocketHandler:
    def __init__(self):
        self.client = None
        self.bsm = None
        self.connections = {}
        self.callbacks = {}
        self.is_running = False
        
    async def initialize(self):
        """Initialize the WebSocket client"""
        try:
            api_key = config.binance_api_key
            secret_key = config.binance_secret_key
            
            if not api_key or not secret_key:
                logger.warning("WebSocket will run in demo mode - no API credentials")
                return False
                
            self.client = await AsyncClient.create(
                api_key=api_key,
                api_secret=secret_key,
                testnet=config.use_testnet
            )
            self.bsm = BinanceSocketManager(self.client)
            logger.info("WebSocket client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket client: {e}")
            return False
    
    async def start_kline_socket(self, symbol: str, interval: str, callback: Callable):
        """Start kline/candlestick WebSocket stream"""
        if not self.bsm:
            return False
            
        try:
            socket_key = f"{symbol}_{interval}_kline"
            if socket_key in self.connections:
                return True
                
            ts = self.bsm.kline_socket(symbol=symbol, interval=interval)
            self.connections[socket_key] = ts
            self.callbacks[socket_key] = callback
            
            async with ts as tscm:
                while True:
                    msg = await tscm.recv()
                    if callback:
                        await callback(msg)
                        
        except Exception as e:
            logger.error(f"Error in kline socket for {symbol}: {e}")
            return False
    
    async def start_ticker_socket(self, symbol: str, callback: Callable):
        """Start ticker WebSocket stream"""
        if not self.bsm:
            return False
            
        try:
            socket_key = f"{symbol}_ticker"
            if socket_key in self.connections:
                return True
                
            ts = self.bsm.symbol_ticker_socket(symbol=symbol)
            self.connections[socket_key] = ts
            self.callbacks[socket_key] = callback
            
            async with ts as tscm:
                while True:
                    msg = await tscm.recv()
                    if callback:
                        await callback(msg)
                        
        except Exception as e:
            logger.error(f"Error in ticker socket for {symbol}: {e}")
            return False
    
    async def start_depth_socket(self, symbol: str, callback: Callable):
        """Start order book depth WebSocket stream"""
        if not self.bsm:
            return False
            
        try:
            socket_key = f"{symbol}_depth"
            if socket_key in self.connections:
                return True
                
            ts = self.bsm.depth_socket(symbol=symbol)
            self.connections[socket_key] = ts
            self.callbacks[socket_key] = callback
            
            async with ts as tscm:
                while True:
                    msg = await tscm.recv()
                    if callback:
                        await callback(msg)
                        
        except Exception as e:
            logger.error(f"Error in depth socket for {symbol}: {e}")
            return False
    
    async def start_multiplex_socket(self, streams: List[str], callback: Callable):
        """Start multiplex WebSocket stream for multiple symbols"""
        if not self.bsm:
            return False
            
        try:
            socket_key = "multiplex"
            if socket_key in self.connections:
                return True
                
            ts = self.bsm.multiplex_socket(streams)
            self.connections[socket_key] = ts
            self.callbacks[socket_key] = callback
            
            async with ts as tscm:
                while True:
                    msg = await tscm.recv()
                    if callback:
                        await callback(msg)
                        
        except Exception as e:
            logger.error(f"Error in multiplex socket: {e}")
            return False
    
    async def close_connection(self, socket_key: str):
        """Close a specific WebSocket connection"""
        if socket_key in self.connections:
            try:
                await self.connections[socket_key].close()
                del self.connections[socket_key]
                if socket_key in self.callbacks:
                    del self.callbacks[socket_key]
                logger.info(f"Closed WebSocket connection: {socket_key}")
            except Exception as e:
                logger.error(f"Error closing connection {socket_key}: {e}")
    
    async def close_all_connections(self):
        """Close all WebSocket connections"""
        for socket_key in list(self.connections.keys()):
            await self.close_connection(socket_key)
        
        if self.client:
            await self.client.close_connection()
    
    def is_connected(self, socket_key: str) -> bool:
        """Check if a specific WebSocket connection is active"""
        return socket_key in self.connections
    
    def get_active_connections(self) -> List[str]:
        """Get list of active WebSocket connections"""
        return list(self.connections.keys())

# Global WebSocket handler instance
websocket_handler = WebSocketHandler()
