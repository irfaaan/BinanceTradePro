from app import db
from datetime import datetime
from sqlalchemy.dialects.sqlite import JSON

class TradingPair(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), unique=True, nullable=False)
    base_asset = db.Column(db.String(10), nullable=False)
    quote_asset = db.Column(db.String(10), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MarketData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    interval = db.Column(db.String(10), nullable=False)  # 1m, 5m, 15m, 1h, 1d
    
    __table_args__ = (db.UniqueConstraint('symbol', 'timestamp', 'interval'),)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(4), nullable=False)  # BUY or SELL
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    total_value = db.Column(db.Float, nullable=False)
    strategy_signal = db.Column(db.String(50))
    prediction_confidence = db.Column(db.Float)
    is_paper_trade = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(20), default='FILLED')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class TechnicalIndicators(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    indicators = db.Column(JSON)  # Store all indicators as JSON
    
    __table_args__ = (db.UniqueConstraint('symbol', 'timestamp'),)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset = db.Column(db.String(10), nullable=False)
    balance = db.Column(db.Float, default=0.0)
    locked_balance = db.Column(db.Float, default=0.0)
    avg_price = db.Column(db.Float, default=0.0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('asset'),)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    direction = db.Column(db.String(4), nullable=False)  # UP or DOWN
    timeframe = db.Column(db.String(10), nullable=False)  # 30m, 1h, 4h
    model_used = db.Column(db.String(50))
    actual_price = db.Column(db.Float)  # Filled later for accuracy calculation
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
