# core/features.py

import numpy as np
import pandas as pd
from scipy import stats
# from statsmodels.tsa.stattools import adfuller # Not used currently
from sklearn.preprocessing import StandardScaler
# import talib # Removed talib
import pandas_ta  # Import pandas_ta
from collections import deque
import joblib # Keep joblib for scaler persistence
import os
import warnings # To suppress specific pandas_ta warnings if needed
import logging # Import logging

logger = logging.getLogger("genovo_traderv2") # Get logger

# Suppress specific PerformanceWarnings from pandas_ta if they become noisy
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class FeatureExtractor:
    """
    Comprehensive feature extraction pipeline using pandas-ta.
    Generates price, volume, technical, statistical, volatility,
    trend, seasonality features. Includes Donchian Channel.
    Includes robust scaler fitting and transformation.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.window_sizes = self.config.get('window_sizes', [5, 10, 20, 50, 100, 200]) # Keep 200 for other features
        # --- Donchian Config ---
        # Use lengths appropriate for your timeframe (M1).
        # 50 weeks = 50*5*24*60 = 360,000 M1 bars (too long!)
        # Let's use shorter, more typical lengths for M1, e.g., related to hours/days
        # Example: ~4 hours (240), ~8 hours (480), ~1 day (1440)
        self.donchian_lower_length = self.config.get('donchian_lower', 240) # e.g., 4-hour low
        self.donchian_upper_length = self.config.get('donchian_upper', 480) # e.g., 8-hour high
        # --- End Donchian Config ---

        self.feature_names = [] # Stores names of features the scaler is fitted on
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.tick_buffer = deque(maxlen=max(self.window_sizes + [self.donchian_lower_length, self.donchian_upper_length])) # Adjust buffer if needed
        self.order_book_buffer = deque(maxlen=200)
        self.feature_means = None
        self.feature_stds = None
        # print("FeatureExtractor initialized (using pandas-ta).")

    def fit(self, market_data):
        """ Fits the scaler based on historical market data. """
        if market_data is None or market_data.empty:
             self.is_fitted = False; return self
        features = self._extract_all_features(market_data)
        numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: self.is_fitted = False; return self
        features_numeric = features[numeric_cols]
        features_numeric = features_numeric.replace([np.inf, -np.inf], np.nan)
        features_filled = features_numeric.ffill().bfill()
        cols_with_nans = features_filled.columns[features_filled.isnull().any()].tolist()
        if cols_with_nans: features_cleaned = features_filled.drop(columns=cols_with_nans)
        else: features_cleaned = features_filled
        cols_to_drop = features_cleaned.columns[features_cleaned.var() == 0]
        if not cols_to_drop.empty: features_cleaned = features_cleaned.drop(columns=cols_to_drop)
        if features_cleaned.empty: self.is_fitted = False; return self
        try:
            self.scaler.fit(features_cleaned)
            self.is_fitted = True
            self.feature_names = features_cleaned.columns.tolist()
            self.feature_means = features_cleaned.mean()
            self.feature_stds = features_cleaned.std()
            logger.info(f"FeatureExtractor fitted with {len(self.feature_names)} numeric features.") # Log actual count
        except ValueError as e:
             logger.error(f"Error fitting scaler: {e}. Check for constant features or other issues.", exc_info=True)
             self.is_fitted = False; self.feature_names = []
        return self

    def transform(self, raw_features_data):
        """ Transforms pre-calculated raw features into a scaled feature matrix. """
        if not self.is_fitted:
            logger.error("Scaler not fitted. Cannot transform features.")
            return pd.DataFrame(columns=self.feature_names)
        if raw_features_data is None or raw_features_data.empty:
             logger.warning("Empty raw features data provided for transform.")
             return pd.DataFrame(columns=self.feature_names)

        features = raw_features_data.copy()
        cols_to_scale = []
        missing_cols = []
        extra_cols = list(features.columns)
        for col in self.feature_names:
            if col in features.columns:
                cols_to_scale.append(col)
                if col in extra_cols: extra_cols.remove(col)
            else: missing_cols.append(col)
        # if missing_cols: logger.warning(f"Features missing during transform (needed by scaler): {missing_cols}. They will be added and filled with 0.") # Reduce log noise
        # if extra_cols: logger.debug(f"Extra features found during transform (not used by scaler): {extra_cols}. They will be dropped.") # Reduce log noise
        if not cols_to_scale:
             logger.warning("No features available to scale matching the fitted scaler.")
             return pd.DataFrame(0, index=features.index, columns=self.feature_names)

        features_to_scale_df = features[cols_to_scale]
        features_to_scale_df = features_to_scale_df.replace([np.inf, -np.inf], np.nan)
        features_filled = features_to_scale_df.ffill().bfill()
        if features_filled.isnull().values.any():
             # logger.warning("NaNs still present after ffill/bfill before scaling. Filling remaining NaNs with 0.")
             features_filled.fillna(0, inplace=True) # Fill remaining with 0

        try:
             scaled_array = self.scaler.transform(features_filled)
             scaled_features = pd.DataFrame(scaled_array, index=features_filled.index, columns=cols_to_scale)
             if missing_cols:
                  for col in missing_cols: scaled_features[col] = 0.0
                  scaled_features = scaled_features[self.feature_names]
             return scaled_features
        except ValueError as e:
             logger.error(f"Error transforming features: {e}. Returning empty DataFrame.", exc_info=True)
             return pd.DataFrame(columns=self.feature_names)
        except Exception as e:
             logger.error(f"Unexpected error during feature transformation: {e}", exc_info=True)
             return pd.DataFrame(columns=self.feature_names)

    def save_scaler(self, path):
        """Saves the fitted scaler and feature names to a file using joblib."""
        if self.is_fitted:
            try:
                scaler_state = {'scaler': self.scaler, 'feature_names': self.feature_names, 'feature_means': self.feature_means, 'feature_stds': self.feature_stds}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                joblib.dump(scaler_state, path)
                logger.info(f"Scaler state saved successfully to {path}")
                return True
            except Exception as e:
                logger.error(f"Error saving scaler state to {path}: {e}", exc_info=True)
                return False
        else:
            logger.warning("Scaler not fitted. Cannot save state.")
            return False

    def load_scaler(self, path):
        """Loads the scaler and feature names from a file using joblib."""
        logger.debug(f"Attempting to load scaler state from: {path}")
        if os.path.exists(path):
            try:
                scaler_state = joblib.load(path)
                if 'scaler' not in scaler_state or 'feature_names' not in scaler_state:
                     logger.error(f"Invalid scaler state file format in {path}. Missing 'scaler' or 'feature_names'.")
                     self.is_fitted = False; self.feature_names = []; return
                self.scaler = scaler_state['scaler']
                self.feature_names = scaler_state['feature_names']
                self.feature_means = scaler_state.get('feature_means')
                self.feature_stds = scaler_state.get('feature_stds')
                self.is_fitted = True
                logger.info(f"Scaler state loaded successfully from {path} ({len(self.feature_names)} features).")
            except ModuleNotFoundError as e:
                 logger.error(f"Error loading scaler state from {path}: Module not found. Sklearn version mismatch? Error: {e}", exc_info=True)
                 self.is_fitted = False; self.feature_names = []
            except EOFError as e:
                 logger.error(f"Error loading scaler state from {path}: EOFError. File might be corrupted or incomplete. Error: {e}", exc_info=True)
                 self.is_fitted = False; self.feature_names = []
            except Exception as e:
                logger.error(f"Error loading scaler state from {path}: {type(e).__name__}: {e}", exc_info=True)
                self.is_fitted = False; self.feature_names = []
        else:
            logger.error(f"Scaler state file not found at {path}. Scaler remains unfitted.")
            self.is_fitted = False; self.feature_names = []


    def _extract_all_features(self, market_data):
        """Internal method to orchestrate feature extraction using pandas-ta."""
        df = market_data.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.lower()
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Market data must contain columns: {required_cols}")

        # Calculate features, catching potential errors for individual groups
        try: self._add_price_features_pta(df)
        except Exception as e: logger.warning(f"Warning during price features: {e}")
        try: self._add_volume_features_pta(df)
        except Exception as e: logger.warning(f"Warning during volume features: {e}")
        try: self._add_technical_indicators_pta(df)
        except Exception as e: logger.warning(f"Warning during technical indicators: {e}")
        try: self._add_statistical_features_pta(df)
        except Exception as e: logger.warning(f"Warning during statistical features: {e}")
        try: self._add_volatility_features_pta(df)
        except Exception as e: logger.warning(f"Warning during volatility features: {e}")
        try: self._add_trend_features_pta(df)
        except Exception as e: logger.warning(f"Warning during trend features: {e}")
        try: self._add_seasonality_features(df)
        except Exception as e: logger.warning(f"Warning during seasonality features: {e}")
        # --- Add Donchian Channel ---
        try: self._add_donchian_channel_pta(df)
        except Exception as e: logger.warning(f"Warning during Donchian Channel calculation: {e}")
        # --- End Add Donchian Channel ---


        original_cols = ['open', 'high', 'low', 'close', 'volume']
        features = df.drop(columns=original_cols, errors='ignore')
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    # --- Individual Feature Group Methods using pandas-ta ---
    # (Keep existing methods: _add_price_features_pta, _add_volume_features_pta, etc.)
    def _add_price_features_pta(self, df):
        df.ta.log_return(cumulative=False, append=True)
        df.ta.percent_return(cumulative=False, append=True)
        df.loc[:, 'high_low_ratio'] = (df['high'] / df['low']).replace([np.inf, -np.inf], np.nan)
        df.loc[:, 'close_open_ratio'] = (df['close'] / df['open']).replace([np.inf, -np.inf], np.nan)
        for window in self.window_sizes:
            df.ta.sma(length=window, append=True)
            df.ta.ema(length=window, append=True)
            sma_col = f'SMA_{window}'; ema_col = f'EMA_{window}'
            if sma_col in df.columns: df.loc[:, f'price_vs_sma_{window}'] = (df['close'] / df[sma_col]).replace([np.inf, -np.inf], np.nan)
            if ema_col in df.columns: df.loc[:, f'price_vs_ema_{window}'] = (df['close'] / df[ema_col]).replace([np.inf, -np.inf], np.nan)
            df.ta.mom(length=window, append=True)
        if 5 in self.window_sizes and 20 in self.window_sizes:
             sma5, sma20 = 'SMA_5', 'SMA_20'; ema5, ema20 = 'EMA_5', 'EMA_20'
             if sma5 in df.columns and sma20 in df.columns: df.loc[:, 'sma_5_20_diff'] = df[sma5] - df[sma20]
             if ema5 in df.columns and ema20 in df.columns: df.loc[:, 'ema_5_20_diff'] = df[ema5] - df[ema20]

    def _add_volume_features_pta(self, df):
        df.loc[:, 'log_volume'] = np.log1p(df['volume'])
        for window in self.window_sizes:
            df.ta.sma(close='volume', length=window, prefix='VOL', append=True)
            vol_sma_col = f'VOL_SMA_{window}'
            if vol_sma_col in df.columns: df.loc[:, f'volume_vs_sma_{window}'] = (df['volume'] / df[vol_sma_col]).replace([np.inf, -np.inf], np.nan)
        df.ta.obv(append=True)

    def _add_technical_indicators_pta(self, df):
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.willr(length=14, append=True)

    def _add_statistical_features_pta(self, df):
        log_returns_col = 'LOGRET_1'
        if log_returns_col not in df.columns: df.ta.log_return(cumulative=False, append=True)
        if log_returns_col not in df.columns: logger.error("Failed to calculate log returns. Skipping statistical features."); return
        df[log_returns_col] = pd.to_numeric(df[log_returns_col], errors='coerce').fillna(0)
        for window in self.window_sizes:
            if window <= 1: continue
            df.ta.stdev(close=df[log_returns_col], length=window, append=True, suffix=f"_{log_returns_col}")
            df.ta.skew(close=df[log_returns_col], length=window, append=True, suffix=f"_{log_returns_col}")
            df.ta.kurtosis(close=df[log_returns_col], length=window, append=True, suffix=f"_{log_returns_col}")
            rolling_mean = df[log_returns_col].rolling(window=window).mean()
            rolling_std = df[log_returns_col].rolling(window=window).std().replace(0, np.nan)
            df.loc[:, f'roll_zscore_{window}'] = ((df[log_returns_col] - rolling_mean) / rolling_std).fillna(0)

    def _add_volatility_features_pta(self, df):
        df.ta.atr(length=14, append=True)
        log_returns = df.get('LOGRET_1', pd.Series(dtype=float)).fillna(0)
        abs_log_returns = np.abs(log_returns)
        for window in self.window_sizes:
             if window <= 1: continue
             df.loc[:, f'abs_ret_ma_{window}'] = abs_log_returns.rolling(window=window).mean()
             df.loc[:, f'vol_of_vol_{window}'] = abs_log_returns.rolling(window=window).std().fillna(0)

    def _add_trend_features_pta(self, df):
        df.ta.aroon(length=14, append=True)

    def _add_seasonality_features(self, df):
        if isinstance(df.index, pd.DatetimeIndex):
            df.loc[:, 'hour'] = df.index.hour
            df.loc[:, 'day_of_week'] = df.index.dayofweek
            df.loc[:, 'hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df.loc[:, 'hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df.loc[:, 'day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df.loc[:, 'day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        else: logger.warning("DataFrame index is not DatetimeIndex, cannot add seasonality features.")

    # --- New Method for Donchian Channel ---
    def _add_donchian_channel_pta(self, df):
        """Adds Donchian Channel features using pandas-ta."""
        # Use lengths defined in __init__
        lower_len = self.donchian_lower_length
        upper_len = self.donchian_upper_length
        df.ta.donchian(lower_length=lower_len, upper_length=upper_len, append=True)
        # Resulting columns are typically named DCL_lower_upper, DCM_lower_upper, DCU_lower_upper
        # Example: DCL_240_480, DCM_240_480, DCU_240_480

        # Optional: Add feature for price position within the channel
        lower_col = f'DCL_{lower_len}_{upper_len}'
        upper_col = f'DCU_{lower_len}_{upper_len}'
        if lower_col in df.columns and upper_col in df.columns:
             channel_range = (df[upper_col] - df[lower_col]).replace(0, np.nan) # Avoid division by zero
             df.loc[:, f'donchian_pos_{lower_len}_{upper_len}'] = ((df['close'] - df[lower_col]) / channel_range).fillna(0.5) # Fill NaNs with mid-point (0.5)

    # --- Update methods remain the same ---
    def update_tick(self, tick_data): self.tick_buffer.append(tick_data)
    def update_order_book(self, order_book): self.order_book_buffer.append(order_book)


class FeaturePipeline:
    """
    Manages the overall feature engineering process using the pandas-ta based extractor.
    Handles scaler persistence.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_extractor = FeatureExtractor(self.config.get('feature_config', self.config)) # Pass config down
        self.selected_features = []

    def fit(self, market_data, target=None):
        """Fits the feature extractor's scaler."""
        self.feature_extractor.fit(market_data)
        self.selected_features = self.feature_extractor.feature_names
        return self

    def transform(self, raw_features_data):
        """Transforms pre-calculated raw features using the fitted feature extractor."""
        return self.feature_extractor.transform(raw_features_data)

    def save_scaler(self, path):
        """Saves the fitted scaler state via the FeatureExtractor."""
        return self.feature_extractor.save_scaler(path)

    def load_scaler(self, path):
        """Loads the scaler state via the FeatureExtractor."""
        self.feature_extractor.load_scaler(path)
        if self.feature_extractor.is_fitted:
             self.selected_features = self.feature_extractor.feature_names

    def update_tick(self, tick_data): self.feature_extractor.update_tick(tick_data)
    def update_order_book(self, order_book): self.feature_extractor.update_order_book(order_book)


# --- Factory Function ---
def create_feature_pipeline(config):
    """Factory function to create the FeaturePipeline."""
    feature_config = config.get('feature_config', {}) # Get feature_config section
    return FeaturePipeline(config=feature_config) # Pass it to the pipeline
