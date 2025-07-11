import logging
import sys
from datetime import datetime
from pathlib import Path
import os

class TradingBotLogger:
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        
        # Create handlers
        # File handler for all logs
        all_logs_handler = logging.FileHandler(
            self.log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
        )
        all_logs_handler.setLevel(logging.DEBUG)
        all_logs_handler.setFormatter(file_formatter)
        
        # File handler for errors only
        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(all_logs_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
        # Setup specific loggers
        self.setup_module_loggers()
    
    def setup_module_loggers(self):
        """Setup specific module loggers"""
        # Trading logger
        trading_logger = logging.getLogger('trading')
        trading_handler = logging.FileHandler(
            self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(logging.Formatter(
            '%(asctime)s - TRADING - %(levelname)s - %(message)s'
        ))
        trading_logger.addHandler(trading_handler)
        
        # API logger
        api_logger = logging.getLogger('api')
        api_handler = logging.FileHandler(
            self.log_dir / f"api_{datetime.now().strftime('%Y%m%d')}.log"
        )
        api_handler.setLevel(logging.INFO)
        api_handler.setFormatter(logging.Formatter(
            '%(asctime)s - API - %(levelname)s - %(message)s'
        ))
        api_logger.addHandler(api_handler)
        
        # Analysis logger
        analysis_logger = logging.getLogger('analysis')
        analysis_handler = logging.FileHandler(
            self.log_dir / f"analysis_{datetime.now().strftime('%Y%m%d')}.log"
        )
        analysis_handler.setLevel(logging.INFO)
        analysis_handler.setFormatter(logging.Formatter(
            '%(asctime)s - ANALYSIS - %(levelname)s - %(message)s'
        ))
        analysis_logger.addHandler(analysis_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        return logging.getLogger(name)
    
    def log_trade(self, trade_info: dict):
        """Log trade information"""
        logger = logging.getLogger('trading')
        logger.info(f"Trade executed: {trade_info}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        logger = logging.getLogger('error')
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_api_call(self, endpoint: str, params: dict = None, success: bool = True):
        """Log API call"""
        logger = logging.getLogger('api')
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"API call {status}: {endpoint} - params: {params}")
    
    def log_analysis(self, symbol: str, analysis_type: str, result: dict):
        """Log analysis result"""
        logger = logging.getLogger('analysis')
        logger.info(f"Analysis completed for {symbol} - {analysis_type}: {result}")
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        try:
            current_time = datetime.now()
            for log_file in self.log_dir.glob("*.log"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if (current_time - file_time).days > days_to_keep:
                    log_file.unlink()
                    print(f"Deleted old log file: {log_file}")
        except Exception as e:
            print(f"Error cleaning up logs: {e}")

# Initialize logger
bot_logger = TradingBotLogger()
