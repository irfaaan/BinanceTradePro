import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class DataCache:
    def __init__(self, db_path: str = 'data_cache.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize cache database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Market data cache
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        data TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME NOT NULL,
                        UNIQUE(symbol, interval, timestamp)
                    )
                ''')
                
                # Technical indicators cache
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS indicators_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        indicators TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME NOT NULL,
                        UNIQUE(symbol, timestamp)
                    )
                ''')
                
                # Predictions cache
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        prediction TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME NOT NULL,
                        UNIQUE(symbol, timestamp)
                    )
                ''')
                
                # General cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS general_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT NOT NULL UNIQUE,
                        data TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.info("Cache database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
    
    def cache_market_data(self, symbol: str, interval: str, timestamp: datetime, 
                         data: Dict, expiry_minutes: int = 60):
        """Cache market data"""
        try:
            with self.lock:
                expires_at = datetime.now() + timedelta(minutes=expiry_minutes)
                data_json = json.dumps(data)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data_cache 
                        (symbol, interval, timestamp, data, expires_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, interval, timestamp, data_json, expires_at))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error caching market data: {e}")
    
    def get_cached_market_data(self, symbol: str, interval: str, timestamp: datetime) -> Optional[Dict]:
        """Get cached market data"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT data FROM market_data_cache 
                        WHERE symbol = ? AND interval = ? AND timestamp = ? 
                        AND expires_at > CURRENT_TIMESTAMP
                    ''', (symbol, interval, timestamp))
                    
                    result = cursor.fetchone()
                    if result:
                        return json.loads(result[0])
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting cached market data: {e}")
            return None
    
    def cache_indicators(self, symbol: str, timestamp: datetime, indicators: Dict, expiry_minutes: int = 30):
        """Cache technical indicators"""
        try:
            with self.lock:
                expires_at = datetime.now() + timedelta(minutes=expiry_minutes)
                indicators_json = json.dumps(indicators, default=str)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO indicators_cache 
                        (symbol, timestamp, indicators, expires_at)
                        VALUES (?, ?, ?, ?)
                    ''', (symbol, timestamp, indicators_json, expires_at))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error caching indicators: {e}")
    
    def get_cached_indicators(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get cached indicators"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT indicators FROM indicators_cache 
                        WHERE symbol = ? AND timestamp = ? 
                        AND expires_at > CURRENT_TIMESTAMP
                    ''', (symbol, timestamp))
                    
                    result = cursor.fetchone()
                    if result:
                        return json.loads(result[0])
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting cached indicators: {e}")
            return None
    
    def cache_prediction(self, symbol: str, timestamp: datetime, prediction: Dict, expiry_minutes: int = 15):
        """Cache ML prediction"""
        try:
            with self.lock:
                expires_at = datetime.now() + timedelta(minutes=expiry_minutes)
                prediction_json = json.dumps(prediction, default=str)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO predictions_cache 
                        (symbol, timestamp, prediction, expires_at)
                        VALUES (?, ?, ?, ?)
                    ''', (symbol, timestamp, prediction_json, expires_at))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
    
    def get_cached_prediction(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get cached prediction"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT prediction FROM predictions_cache 
                        WHERE symbol = ? AND timestamp = ? 
                        AND expires_at > CURRENT_TIMESTAMP
                    ''', (symbol, timestamp))
                    
                    result = cursor.fetchone()
                    if result:
                        return json.loads(result[0])
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting cached prediction: {e}")
            return None
    
    def cache_data(self, cache_key: str, data: Any, expiry_minutes: int = 60):
        """Cache general data"""
        try:
            with self.lock:
                expires_at = datetime.now() + timedelta(minutes=expiry_minutes)
                data_json = json.dumps(data, default=str)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO general_cache 
                        (cache_key, data, expires_at)
                        VALUES (?, ?, ?)
                    ''', (cache_key, data_json, expires_at))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached general data"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT data FROM general_cache 
                        WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                    ''', (cache_key,))
                    
                    result = cursor.fetchone()
                    if result:
                        return json.loads(result[0])
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    def clear_expired_cache(self):
        """Clear expired cache entries"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Clear expired entries from all tables
                    tables = ['market_data_cache', 'indicators_cache', 'predictions_cache', 'general_cache']
                    for table in tables:
                        cursor.execute(f'DELETE FROM {table} WHERE expires_at <= CURRENT_TIMESTAMP')
                    
                    conn.commit()
                    logger.info("Cleared expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    stats = {}
                    
                    # Count entries in each table
                    tables = ['market_data_cache', 'indicators_cache', 'predictions_cache', 'general_cache']
                    for table in tables:
                        cursor.execute(f'SELECT COUNT(*) FROM {table}')
                        total = cursor.fetchone()[0]
                        
                        cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE expires_at > CURRENT_TIMESTAMP')
                        valid = cursor.fetchone()[0]
                        
                        stats[table] = {
                            'total_entries': total,
                            'valid_entries': valid,
                            'expired_entries': total - valid
                        }
                    
                    return stats
                    
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_all_cache(self):
        """Clear all cache data"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    tables = ['market_data_cache', 'indicators_cache', 'predictions_cache', 'general_cache']
                    for table in tables:
                        cursor.execute(f'DELETE FROM {table}')
                    
                    conn.commit()
                    logger.info("Cleared all cache data")
                    
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")

# Global cache instance
data_cache = DataCache()
