# core/features.py

import numpy as np
import pandas as pd
from scipy import stats
# from statsmodels.tsa.stattools import adfuller # Not used currently
from sklearn.preprocessing import StandardScaler, RobustScaler
# import talib # Removed talib
import pandas_ta  # Import pandas_ta
from collections import deque
import joblib # Keep joblib for scaler persistence
import os
import warnings # To suppress specific pandas_ta warnings if needed
import logging # Import logging
import talib
from joblib import dump, load
import pywt  # Wavelet transforms
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from sklearn.feature_selection import mutual_info_regression
import empyrical as ep  # Risk metrics

logger = logging.getLogger("genovo_traderv2") # Get logger

# Suppress specific PerformanceWarnings from pandas_ta if they become noisy
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class KalmanFilterFeatures:
    """Kalman filter for price and volume smoothing"""
    def __init__(self):
        self.price_kf = None
        self.volume_kf = None
        
    def initialize_filters(self, n_dim=1):
        self.price_kf = KalmanFilter(dim_x=2, dim_z=1)
        self.price_kf.x = np.zeros(2)
        self.price_kf.F = np.array([[1., 1.], [0., 1.]])
        self.price_kf.H = np.array([[1., 0.]])
        self.price_kf.P *= 1000.
        self.price_kf.R = 5
        self.price_kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
        
        self.volume_kf = KalmanFilter(dim_x=2, dim_z=1)
        self.volume_kf.x = np.zeros(2)
        self.volume_kf.F = np.array([[1., 1.], [0., 1.]])
        self.volume_kf.H = np.array([[1., 0.]])
        self.volume_kf.P *= 1000.
        self.volume_kf.R = 50
        self.volume_kf.Q = np.array([[1., 1.], [1., 1.]])
        
    def update(self, measurement, kf):
        kf.predict()
        kf.update(measurement)
        return kf.x[0], kf.x[1]  # Return state and velocity

class WaveletFeatures:
    """Wavelet-based feature extraction"""
    def __init__(self, wavelet='db1', level=3):
        self.wavelet = wavelet
        self.level = level
        
    def decompose(self, data):
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        return coeffs
        
    def get_features(self, data):
        coeffs = self.decompose(data)
        features = {}
        for i, coeff in enumerate(coeffs):
            if i == 0:
                features[f'wavelet_a{self.level}'] = coeff
            else:
                features[f'wavelet_d{self.level-i+1}'] = coeff
        return features

class MarketMicrostructureFeatures:
    """Advanced market microstructure features"""
    def __init__(self):
        self.prev_trades = None
        
    def calculate_vpin(self, volume, price_change, window=50):
        """Volume-synchronized Probability of Informed Trading"""
        signed_volume = volume * np.sign(price_change)
        buy_volume = np.where(signed_volume > 0, signed_volume, 0)
        sell_volume = np.where(signed_volume < 0, -signed_volume, 0)
        vpin = pd.Series(abs(buy_volume - sell_volume) / (buy_volume + sell_volume)).rolling(window).mean()
        return vpin
        
    def calculate_kyle_lambda(self, price_change, volume, window=50):
        """Kyle's Lambda (Price Impact)"""
        abs_price_change = abs(price_change)
        kyle_lambda = pd.Series(abs_price_change / volume).rolling(window).mean()
        return kyle_lambda
        
    def calculate_amihud(self, returns, volume, window=50):
        """Amihud Illiquidity Ratio"""
        illiq = pd.Series(abs(returns) / volume).rolling(window).mean()
        return illiq

class EnhancedFeatureExtractor:
    """Advanced feature extraction with sophisticated technical indicators"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.window_sizes = self.config.get('window_sizes', [5, 10, 20, 50, 100, 200, 500])
        self.is_fitted = False
        self.feature_names = []
        self.kalman = KalmanFilterFeatures()
        self.wavelet = WaveletFeatures()
        self.microstructure = MarketMicrostructureFeatures()
        self.selected_features = None
        
    def _add_basic_price_features(self, df):
        """Add basic price-based features"""
        # Initialize Kalman filters if needed
        if self.kalman.price_kf is None:
            self.kalman.initialize_filters()
            
        # Kalman filtered price and velocity
        price_states = []
        price_velocities = []
        for price in df['close'].values:
            state, velocity = self.kalman.update(price, self.kalman.price_kf)
            price_states.append(state)
            price_velocities.append(velocity)
        
        df['price_state'] = price_states
        df['price_velocity'] = price_velocities
        
        # Log returns and volatility
        for window in self.window_sizes:
            df[f'log_return_{window}'] = np.log(df['close'] / df['close'].shift(window))
            df[f'volatility_{window}'] = df['log_return_1'].rolling(window).std()
            
            # Realized volatility (high-frequency)
            df[f'realized_vol_{window}'] = np.sqrt(
                (np.log(df['high'] / df['low'])**2).rolling(window).mean() * 252
            )
            
            # Parkinson volatility estimator
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(df['high'] / df['low'])**2).rolling(window).mean() * 252
            )
        
        # Wavelet features
        close_wavelets = self.wavelet.get_features(df['close'].values)
        for name, values in close_wavelets.items():
            df[f'close_{name}'] = values[:len(df)]
            
        # Price ratios and patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        return df
        
    def _add_volume_features(self, df):
        """Add volume-based features"""
        # Kalman filtered volume
        volume_states = []
        volume_velocities = []
        for volume in df['tick_volume'].values:
            state, velocity = self.kalman.update(volume, self.kalman.volume_kf)
            volume_states.append(state)
            volume_velocities.append(velocity)
            
        df['volume_state'] = volume_states
        df['volume_velocity'] = volume_velocities
        
        for window in self.window_sizes:
            # Volume momentum and trends
            df[f'volume_sma_{window}'] = df['tick_volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['tick_volume'].rolling(window).std()
            df[f'volume_momentum_{window}'] = df['tick_volume'] / df[f'volume_sma_{window}']
            
            # Volume weighted metrics
            df[f'vwap_{window}'] = (df['close'] * df['tick_volume']).rolling(window).sum() / df['tick_volume'].rolling(window).sum()
            df[f'volume_price_trend_{window}'] = ((df['close'] - df['close'].shift(window)) * df['tick_volume']) / df['tick_volume'].rolling(window).sum()
            
            # OBV and accumulation/distribution
            df[f'obv_{window}'] = talib.OBV(df['close'].values, df['tick_volume'].values).rolling(window).mean()
            df[f'adl_{window}'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['tick_volume'].values).rolling(window).mean()
            
        # Market microstructure
        df['vpin'] = self.microstructure.calculate_vpin(df['tick_volume'], df['close'].diff())
        df['kyle_lambda'] = self.microstructure.calculate_kyle_lambda(df['close'].diff(), df['tick_volume'])
        df['amihud_ratio'] = self.microstructure.calculate_amihud(df['close'].pct_change(), df['tick_volume'])
        
        return df
        
    def _add_momentum_indicators(self, df):
        """Add momentum-based indicators"""
        # RSI with multiple timeframes
        for window in self.window_sizes:
            df[f'rsi_{window}'] = talib.RSI(df['close'].values, timeperiod=window)
            df[f'rsi_smooth_{window}'] = savgol_filter(
                df[f'rsi_{window}'].fillna(50), 
                min(window, 11) if window > 11 else 5, 
                3
            )
            
        # Enhanced MACD
        for (fast, slow) in [(12, 26), (5, 35), (8, 21)]:
            macd, signal, hist = talib.MACD(
                df['close'].values, 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=9
            )
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = signal
            df[f'macd_hist_{fast}_{slow}'] = hist
            
        # Advanced momentum indicators
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        df['mfi'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['tick_volume'].values)
        df['willr'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
        df['ultosc'] = talib.ULTOSC(df['high'].values, df['low'].values, df['close'].values)
        
        return df
        
    def _add_volatility_indicators(self, df):
        """Add volatility-based indicators"""
        # ATR and variants
        for window in self.window_sizes:
            df[f'atr_{window}'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=window)
            df[f'natr_{window}'] = talib.NATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=window)
            
            # Normalized ATR
            df[f'natr_percentile_{window}'] = (
                df[f'natr_{window}'].rolling(window).apply(
                    lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])
                )
            )
            
        # Enhanced Bollinger Bands
        for window in [20, 50, 100]:
            upper, middle, lower = talib.BBANDS(
                df['close'].values, 
                timeperiod=window,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            df[f'bb_upper_{window}'] = upper
            df[f'bb_middle_{window}'] = middle
            df[f'bb_lower_{window}'] = lower
            df[f'bb_width_{window}'] = (upper - lower) / middle
            
            # BB percentile
            df[f'bb_percentile_{window}'] = (df['close'] - lower) / (upper - lower)
            
        # Volatility regime detection
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['volatility_regime'] = pd.qcut(
            df['high_low_range'].rolling(100).std(),
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        ).astype(str)
        
        return df
        
    def _add_trend_indicators(self, df):
        """Add trend-based indicators"""
        # Multiple EMAs and variants
        for window in self.window_sizes:
            df[f'ema_{window}'] = talib.EMA(df['close'].values, timeperiod=window)
            df[f'tema_{window}'] = talib.TEMA(df['close'].values, timeperiod=window)
            df[f'dema_{window}'] = talib.DEMA(df['close'].values, timeperiod=window)
            
            # Trend strength
            df[f'trend_strength_{window}'] = abs(
                df[f'ema_{window}'] - df[f'ema_{window}'].shift(window)
            ) / df[f'atr_{window}']
            
        # Enhanced Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        # Cloud strength
        df['cloud_strength'] = df['senkou_span_a'] - df['senkou_span_b']
        
        # Trend detection
        df['supertrend'] = talib.HT_TRENDLINE(df['close'].values)
        df['trend_direction'] = np.sign(df['close'] - df['supertrend'])
        
        return df
        
    def _add_market_microstructure(self, df):
        """Add market microstructure features"""
        # Order flow imbalance
        df['trade_imbalance'] = df['tick_volume'] * np.sign(df['close'] - df['open'])
        df['cum_imbalance'] = df['trade_imbalance'].cumsum()
        
        # Liquidity measures
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['effective_spread'] = abs(df['close'] - df['vwap_20']) / df['vwap_20']
        
        # Order flow toxicity
        df['flow_toxicity'] = self.microstructure.calculate_vpin(
            df['tick_volume'],
            df['close'].diff(),
            window=50
        )
        
        # Market impact
        df['price_impact'] = self.microstructure.calculate_kyle_lambda(
            df['close'].diff(),
            df['tick_volume'],
            window=50
        )
        
        # Liquidity risk
        df['illiquidity'] = self.microstructure.calculate_amihud(
            df['close'].pct_change(),
            df['tick_volume'],
            window=50
        )
        
        return df
        
    def _add_regime_features(self, df):
        """Add market regime detection features"""
        # Trend strength and persistence
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        df['trend_strength'] = pd.qcut(
            df['adx'],
            q=5,
            labels=['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
        ).astype(str)
        
        # Volatility regime
        for window in [20, 50, 100]:
            vol = df['close'].pct_change().rolling(window).std() * np.sqrt(252)
            df[f'volatility_regime_{window}'] = pd.qcut(
                vol,
                q=5,
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            ).astype(str)
        
        # Volume regime
        for window in [20, 50, 100]:
            vol_z_score = (
                (df['tick_volume'] - df['tick_volume'].rolling(window).mean()) /
                df['tick_volume'].rolling(window).std()
            )
            df[f'volume_regime_{window}'] = pd.qcut(
                vol_z_score,
                q=5,
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            ).astype(str)
            
        # Market efficiency ratio
        for window in [20, 50, 100]:
            df[f'efficiency_ratio_{window}'] = abs(
                df['close'] - df['close'].shift(window)
            ) / (df['high'] - df['low']).rolling(window).sum()
            
        return df
        
    def _select_features(self, df, top_k=None):
        """Select most informative features using mutual information"""
        if top_k is None:
            return df
            
        # Calculate returns (target variable for feature selection)
        returns = df['close'].pct_change().shift(-1)  # Forward returns
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(
            df.select_dtypes(include=[np.number]).fillna(0),
            returns.fillna(0)
        )
        
        # Get top features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scores = pd.Series(mi_scores, index=numeric_cols)
        selected_features = scores.nlargest(top_k).index.tolist()
        
        # Add back non-numeric columns
        final_features = selected_features + df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        return df[final_features]
        
    def fit(self, df):
        """Fit the feature extractor"""
        features = self._extract_all_features(df)
        
        # Feature selection if configured
        if self.config.get('feature_selection', {}).get('method') == 'mutual_info':
            top_k = self.config.get('feature_selection', {}).get('top_k', None)
            if top_k:
                features = self._select_features(features, top_k)
                self.selected_features = features.columns.tolist()
        
        self.feature_names = features.columns.tolist()
        self.is_fitted = True
        return self
        
    def _extract_all_features(self, df):
        """Extract all features"""
        df = df.copy()
        
        # Add basic return
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        
        # Add all feature groups
        df = self._add_basic_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_trend_indicators(df)
        df = self._add_market_microstructure(df)
        df = self._add_regime_features(df)
        
        # Drop rows with NaN values from lookback windows
        df = df.dropna()
        
        return df
        
    def transform(self, df):
        """Transform the data by extracting features"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
            
        features = self._extract_all_features(df)
        
        # Apply feature selection if fitted with it
        if self.selected_features is not None:
            features = features[self.selected_features]
            
        return features
        
    def fit_transform(self, df):
        """Fit and transform the data"""
        self.fit(df)
        return self.transform(df)


class FeaturePipeline:
    """Complete feature engineering pipeline with advanced scaling"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_extractor = EnhancedFeatureExtractor(config)
        
        # Use robust quantile scaling if configured
        if self.config.get('normalization', {}).get('type') == 'robust_quantile':
            q_range = self.config.get('normalization', {}).get('quantile_range', [0.001, 0.999])
            self.scaler = RobustScaler(quantile_range=q_range)
        else:
            self.scaler = RobustScaler()
            
        self.is_fitted = False
        
    def fit(self, df):
        """Fit the pipeline"""
        features = self.feature_extractor.fit_transform(df)
        self.scaler.fit(features.select_dtypes(include=[np.number]))
        self.is_fitted = True
        return self
        
    def transform(self, df):
        """Transform using the pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        features = self.feature_extractor.transform(df)
        
        # Scale only numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        categorical_features = features.select_dtypes(exclude=[np.number])
        
        scaled_numeric = pd.DataFrame(
            self.scaler.transform(numeric_features),
            index=numeric_features.index,
            columns=numeric_features.columns
        )
        
        # Combine scaled numeric with categorical
        if not categorical_features.empty:
            scaled_features = pd.concat([scaled_numeric, categorical_features], axis=1)
        else:
            scaled_features = scaled_numeric
            
        return scaled_features
        
    def fit_transform(self, df):
        """Fit and transform the data"""
        self.fit(df)
        return self.transform(df)
        
    def save(self, path):
        """Save the pipeline to disk"""
        dump(self, path)
        
    @classmethod
    def load(cls, path):
        """Load the pipeline from disk"""
        return load(path)


def create_feature_pipeline(config=None):
    """Factory function to create feature pipeline"""
    return FeaturePipeline(config)
