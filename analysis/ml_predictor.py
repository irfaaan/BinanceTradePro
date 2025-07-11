import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performance = {}
        self.prediction_horizon = 30  # minutes
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            if df.empty or len(df) < 60:
                return pd.DataFrame()
            
            features_df = df.copy()
            
            # Price features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['high_low_ratio'] = features_df['high'] / features_df['low']
            features_df['price_volatility'] = features_df['close'].rolling(window=20).std()
            
            # Volume features
            features_df['volume_change'] = features_df['volume'].pct_change()
            features_df['volume_ma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
            
            # Technical indicators (simplified)
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
            features_df['macd'] = self._calculate_macd(features_df['close'])
            features_df['bb_position'] = self._calculate_bb_position(features_df['close'])
            
            # Time-based features
            features_df['hour'] = pd.to_datetime(features_df.index).hour
            features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features_df[f'close_ma_{window}'] = features_df['close'].rolling(window=window).mean()
                features_df[f'close_std_{window}'] = features_df['close'].rolling(window=window).std()
                features_df[f'volume_ma_{window}'] = features_df['volume'].rolling(window=window).mean()
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
                features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            # Select feature columns
            self.feature_columns = [col for col in features_df.columns 
                                  if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            return features_df[self.feature_columns]
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd
    
    def _calculate_bb_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position
    
    def train_model(self, df: pd.DataFrame, symbol: str) -> bool:
        """Train ML model for a specific symbol"""
        try:
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for training {symbol}")
                return False
            
            # Prepare features
            features_df = self.prepare_features(df)
            if features_df.empty:
                return False
            
            # Create target variable (future price change)
            target = df['close'].shift(-30).pct_change()  # 30-minute future return
            
            # Align features and target
            common_index = features_df.index.intersection(target.index)
            X = features_df.loc[common_index]
            y = target.loc[common_index]
            
            # Remove NaN values
            valid_idx = ~(np.isnan(y) | np.isinf(y))
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 50:
                logger.warning(f"Insufficient valid data for training {symbol}")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy (percentage of correct direction predictions)
            direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.model_performance[symbol] = {
                'mse': mse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'trained_at': datetime.now(),
                'n_samples': len(X_train)
            }
            
            logger.info(f"Model trained for {symbol}: R2={r2:.4f}, Direction Accuracy={direction_accuracy:.4f}")
            
            # Save model to disk
            self._save_model(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Make prediction for a symbol"""
        try:
            if symbol not in self.models:
                logger.warning(f"No model available for {symbol}")
                return None
            
            # Prepare features
            features_df = self.prepare_features(df)
            if features_df.empty:
                return None
            
            # Get latest features
            latest_features = features_df.iloc[-1:].values
            
            # Scale features
            scaler = self.scalers[symbol]
            features_scaled = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models[symbol]
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence based on feature importance and prediction consistency
            confidence = self._calculate_confidence(model, features_scaled, prediction)
            
            # Determine direction
            direction = 'UP' if prediction > 0 else 'DOWN'
            
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + prediction)
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': prediction,
                'predicted_change_percent': prediction * 100,
                'direction': direction,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'model_performance': self.model_performance.get(symbol, {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def _calculate_confidence(self, model, features, prediction) -> float:
        """Calculate prediction confidence"""
        try:
            # Get feature importance
            feature_importance = model.feature_importances_
            
            # Calculate prediction strength based on feature importance
            weighted_features = np.abs(features[0] * feature_importance)
            strength = np.sum(weighted_features) / len(weighted_features)
            
            # Normalize to 0-1 range
            confidence = min(max(strength, 0), 1)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _save_model(self, symbol: str):
        """Save model to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            model_path = f'models/{symbol}_model.pkl'
            scaler_path = f'models/{symbol}_scaler.pkl'
            
            joblib.dump(self.models[symbol], model_path)
            joblib.dump(self.scalers[symbol], scaler_path)
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load model from disk"""
        try:
            model_path = f'models/{symbol}_model.pkl'
            scaler_path = f'models/{symbol}_scaler.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                logger.info(f"Model loaded for {symbol}")
                return True
            else:
                logger.warning(f"Model files not found for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return False
    
    def get_model_performance(self, symbol: str) -> Optional[Dict]:
        """Get model performance metrics"""
        return self.model_performance.get(symbol)
    
    def is_model_trained(self, symbol: str) -> bool:
        """Check if model is trained for symbol"""
        return symbol in self.models

# Global ML predictor instance
ml_predictor = MLPredictor()
