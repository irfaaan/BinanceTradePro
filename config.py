import os
import json
from typing import Dict, List, Any

class Config:
    def __init__(self):
        self.load_config()
        
    def load_config(self):
        """Load configuration from config.json and environment variables"""
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self.get_default_config()
            self.save_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "trading_pairs": [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"
            ],
            "risk_management": {
                "max_position_size": 0.05,  # 5% of portfolio per trade
                "stop_loss_percent": 0.01,  # 1% stop loss
                "take_profit_percent": 0.02,  # 2% take profit
                "max_daily_loss": 0.10,  # 10% max daily loss
                "max_open_trades": 5
            },
            "trading_strategy": {
                "min_confidence": 0.90,  # 90% minimum confidence for trades
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "sma_periods": [10, 20, 50],
                "ema_periods": [12, 26],
                "macd_params": [12, 26, 9],
                "bollinger_period": 20,
                "bollinger_std": 2
            },
            "data_settings": {
                "intervals": ["1m", "5m", "15m"],
                "history_days": 60,
                "cache_duration": 300  # 5 minutes
            },
            "paper_trading": {
                "initial_balance": 10000.0,  # $10,000 starting balance
                "base_currency": "USDT"
            },
            "ml_settings": {
                "model_type": "LSTM",
                "retrain_interval": 24,  # hours
                "feature_window": 60,  # 60 periods for features
                "prediction_horizon": 30  # minutes
            },
            "api_settings": {
                "use_testnet": True,
                "rate_limit_buffer": 0.8  # Use 80% of rate limits
            }
        }
    
    def save_config(self):
        """Save configuration to config.json"""
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config()
    
    @property
    def trading_pairs(self) -> List[str]:
        return self.config.get("trading_pairs", [])
    
    @property
    def binance_api_key(self) -> str:
        return os.getenv("BINANCE_API_KEY", "")
    
    @property
    def binance_secret_key(self) -> str:
        return os.getenv("BINANCE_SECRET_KEY", "")
    
    @property
    def use_testnet(self) -> bool:
        return self.config.get("api_settings", {}).get("use_testnet", True)

# Global config instance
config = Config()
