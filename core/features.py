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

# Suppress specific PerformanceWarnings from pandas_ta if they become noisy
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class FeatureExtractor:
    """
    Comprehensive feature extraction pipeline using pandas-ta.
    Generates price, volume, technical, statistical, volatility,
    trend, seasonality features.
    Includes robust scaler fitting and transformation.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.window_sizes = self.config.get('window_sizes', [5, 10, 20, 50, 100])
        self.feature_names = [] # Stores names of features the scaler is fitted on

        self.scaler = StandardScaler()
        self.is_fitted = False
        # Buffers not currently used in feature calculation but kept for potential future use
        self.tick_buffer = deque(maxlen=200)
        self.order_book_buffer = deque(maxlen=200)

        # Store mean/std for potential unscaling or analysis
        self.feature_means = None
        self.feature_stds = None
        print("FeatureExtractor initialized (using pandas-ta).")

    def fit(self, market_data):
        """
        Fits the scaler based on historical market data.
        Only fits on numeric features without NaN/inf values.
        """
        if market_data is None or market_data.empty:
             print("Warning: Empty market data provided for fitting. Scaler not fitted.")
             self.is_fitted = False
             return self

        print(f"Fitting scaler on initial data ({len(market_data)} rows)...")
        features = self._extract_all_features(market_data) # Extract raw features

        # --- Robust Cleaning Before Fit ---
        # 1. Identify numeric columns
        numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
             print("Warning: No numeric features generated during fit. Scaler not fitted.")
             self.is_fitted = False
             return self
        features_numeric = features[numeric_cols]

        # 2. Handle Infinite values
        features_numeric = features_numeric.replace([np.inf, -np.inf], np.nan)

        # 3. Handle NaNs (Drop rows with any NaNs in numeric columns for fitting)
        initial_len = len(features_numeric)
        features_cleaned = features_numeric.dropna()
        dropped_rows = initial_len - len(features_cleaned)
        if dropped_rows > 0:
             print(f"Warning: Dropped {dropped_rows} rows with NaNs before fitting scaler.")

        if features_cleaned.empty:
            print("Warning: No valid numeric features remaining after cleaning. Scaler not fitted.")
            self.is_fitted = False
            return self

        try:
            # Fit scaler only on the cleaned numeric data
            self.scaler.fit(features_cleaned)
            self.is_fitted = True
            # Store names of the columns the scaler was actually fitted on
            self.feature_names = features_cleaned.columns.tolist()
            # Store means/stds for reference
            self.feature_means = features_cleaned.mean()
            self.feature_stds = features_cleaned.std()
            print(f"FeatureExtractor fitted with {len(self.feature_names)} numeric features.")

        except ValueError as e:
             print(f"Error fitting scaler: {e}. Check for constant features or other issues.")
             self.is_fitted = False
             self.feature_names = []

        return self

    def transform(self, market_data):
        """
        Transforms new market data into a scaled feature matrix using the fitted scaler.
        Handles missing columns and potential NaN/inf values robustly.
        """
        if not self.is_fitted:
            print("Warning: Scaler not fitted. Returning unscaled features.")
            # Still extract features, but don't scale
            features = self._extract_all_features(market_data)
            # Basic cleaning of raw features before returning
            features = features.replace([np.inf, -np.inf], np.nan)
            # Forward fill NaNs in the unscaled output for model compatibility
            features = features.ffill().bfill()
            return features.select_dtypes(include=np.number) # Return only numeric

        if market_data is None or market_data.empty:
             print("Warning: Empty market data provided for transform.")
             # Return empty DataFrame with expected columns if possible
             return pd.DataFrame(columns=self.feature_names)

        # 1. Extract raw features
        features = self._extract_all_features(market_data)

        # 2. Select only the features the scaler was trained on
        cols_to_scale = [col for col in self.feature_names if col in features.columns]
        missing_cols = [col for col in self.feature_names if col not in features.columns]
        if missing_cols:
             print(f"Warning: Features missing during transform (were present during fit): {missing_cols}")

        if not cols_to_scale:
             print("Warning: No features available to scale matching the fitted scaler.")
             # Return empty DataFrame with expected columns
             return pd.DataFrame(columns=self.feature_names)

        features_to_scale_df = features[cols_to_scale]

        # 3. Clean features before scaling (handle inf/NaN)
        features_to_scale_df = features_to_scale_df.replace([np.inf, -np.inf], np.nan)
        # Fill NaNs *before* scaling (e.g., forward fill) - crucial!
        # Use ffill then bfill to handle NaNs at the beginning/end
        features_filled = features_to_scale_df.ffill().bfill()

        # Check if any NaNs remain after filling (shouldn't happen with ffill/bfill unless all are NaN)
        if features_filled.isnull().values.any():
             print("Warning: NaNs still present after ffill/bfill before scaling. Check input data.")
             # Decide on handling: drop rows, return empty, or try scaling anyway?
             # Returning empty might be safest if NaNs persist unexpectedly.
             return pd.DataFrame(columns=self.feature_names)

        try:
             # 4. Scale the cleaned and filled data
             scaled_array = self.scaler.transform(features_filled)
             scaled_features = pd.DataFrame(scaled_array, index=features_filled.index, columns=cols_to_scale)

             # 5. Reintroduce any missing fitted columns filled with zeros (or mean/median)
             # This ensures the output always has the columns the model expects
             if missing_cols:
                  for col in missing_cols:
                       # Fill with 0 after scaling (mean of scaled data)
                       scaled_features[col] = 0.0
                  # Reorder columns to match the original fitted order
                  scaled_features = scaled_features[self.feature_names]

             return scaled_features

        except ValueError as e:
             print(f"Error transforming features: {e}. Returning empty DataFrame.")
             return pd.DataFrame(columns=self.feature_names) # Return empty on error
        except Exception as e:
             print(f"Unexpected error during feature transformation: {e}")
             return pd.DataFrame(columns=self.feature_names) # Return empty on error

    def save_scaler(self, path):
        """Saves the fitted scaler and feature names to a file using joblib."""
        if self.is_fitted:
            try:
                scaler_state = {
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'feature_means': self.feature_means,
                    'feature_stds': self.feature_stds
                }
                joblib.dump(scaler_state, path)
                print(f"Scaler state saved to {path}")
            except Exception as e:
                print(f"Error saving scaler state to {path}: {e}")
        else:
            print("Scaler not fitted. Cannot save state.")

    def load_scaler(self, path):
        """Loads the scaler and feature names from a file using joblib."""
        if os.path.exists(path):
            try:
                scaler_state = joblib.load(path)
                self.scaler = scaler_state['scaler']
                self.feature_names = scaler_state['feature_names']
                self.feature_means = scaler_state.get('feature_means') # Load if exists
                self.feature_stds = scaler_state.get('feature_stds')   # Load if exists
                self.is_fitted = True
                print(f"Scaler state loaded successfully from {path} ({len(self.feature_names)} features).")
            except Exception as e:
                print(f"Error loading scaler state from {path}: {e}. Scaler remains unfitted.")
                self.is_fitted = False
                self.feature_names = []
        else:
            print(f"Scaler state file not found at {path}. Scaler remains unfitted.")
            self.is_fitted = False
            self.feature_names = []


    def _extract_all_features(self, market_data):
        """Internal method to orchestrate feature extraction using pandas-ta."""
        df = market_data.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            # Try converting to lowercase if columns exist but case is wrong
            df.columns = df.columns.str.lower()
            if not all(col in df.columns for col in required_cols):
                 raise ValueError(f"Market data must contain columns: {required_cols}")

        # Ensure columns are lowercase for pandas-ta compatibility
        df.columns = df.columns.str.lower()

        # --- Use pandas-ta Strategy or individual indicators ---
        # Consider using a try-except block for each group or indicator
        # if specific ones are prone to errors with certain data.

        try: self._add_price_features_pta(df)
        except Exception as e: print(f"Error adding price features: {e}")

        try: self._add_volume_features_pta(df)
        except Exception as e: print(f"Error adding volume features: {e}")

        try: self._add_technical_indicators_pta(df)
        except Exception as e: print(f"Error adding technical indicators: {e}")

        try: self._add_statistical_features_pta(df)
        except Exception as e: print(f"Error adding statistical features: {e}")

        try: self._add_volatility_features_pta(df)
        except Exception as e: print(f"Error adding volatility features: {e}")

        try: self._add_trend_features_pta(df)
        except Exception as e: print(f"Error adding trend features: {e}")

        try: self._add_seasonality_features(df)
        except Exception as e: print(f"Error adding seasonality features: {e}")

        # Select only the calculated feature columns (excluding original OHLCV)
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        features = df.drop(columns=original_cols, errors='ignore')

        # Final check for inf values (should be handled earlier, but as safety)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)

        return features

    # --- Individual Feature Group Methods using pandas-ta ---
    # (Keep these methods as they were, they use pandas-ta correctly)

    def _add_price_features_pta(self, df):
        """Adds price-based features using pandas-ta."""
        df.ta.log_return(cumulative=False, append=True) # Adds 'LOGRET_1'
        df.ta.percent_return(cumulative=False, append=True) # Adds 'PERCENT_1'

        # Ratios (calculate manually)
        # Use .loc to avoid SettingWithCopyWarning if df is a slice
        df.loc[:, 'high_low_ratio'] = (df['high'] / df['low']).replace([np.inf, -np.inf], np.nan)
        df.loc[:, 'close_open_ratio'] = (df['close'] / df['open']).replace([np.inf, -np.inf], np.nan)

        for window in self.window_sizes:
            df.ta.sma(length=window, append=True) # Adds SMA_window
            df.ta.ema(length=window, append=True) # Adds EMA_window
            # Price relative to MA (calculate manually)
            sma_col = f'SMA_{window}'
            ema_col = f'EMA_{window}'
            if sma_col in df.columns:
                 df.loc[:, f'price_vs_sma_{window}'] = (df['close'] / df[sma_col]).replace([np.inf, -np.inf], np.nan)
            if ema_col in df.columns:
                 df.loc[:, f'price_vs_ema_{window}'] = (df['close'] / df[ema_col]).replace([np.inf, -np.inf], np.nan)
            df.ta.mom(length=window, append=True) # Adds MOM_window

        # MA Crossover (calculate manually)
        if 5 in self.window_sizes and 20 in self.window_sizes:
             sma5, sma20 = 'SMA_5', 'SMA_20'
             ema5, ema20 = 'EMA_5', 'EMA_20'
             if sma5 in df.columns and sma20 in df.columns:
                  df.loc[:, 'sma_5_20_diff'] = df[sma5] - df[sma20]
             if ema5 in df.columns and ema20 in df.columns:
                  df.loc[:, 'ema_5_20_diff'] = df[ema5] - df[ema20]

    def _add_volume_features_pta(self, df):
        """Adds volume-based features using pandas-ta."""
        df.loc[:, 'log_volume'] = np.log1p(df['volume'])

        for window in self.window_sizes:
            # Use the original 'volume' column for SMA calculation
            df.ta.sma(close='volume', length=window, prefix='VOL', append=True) # Adds VOL_SMA_window
            # Volume relative to its SMA (calculate manually)
            vol_sma_col = f'VOL_SMA_{window}'
            if vol_sma_col in df.columns:
                 df.loc[:, f'volume_vs_sma_{window}'] = (df['volume'] / df[vol_sma_col]).replace([np.inf, -np.inf], np.nan)

        df.ta.obv(append=True) # Adds 'OBV'

    def _add_technical_indicators_pta(self, df):
        """Adds common technical indicators using pandas-ta."""
        df.ta.rsi(length=14, append=True) # Adds RSI_14
        df.ta.macd(fast=12, slow=26, signal=9, append=True) # Adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df.ta.bbands(length=20, std=2, append=True) # Adds BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True) # Adds STOCHk_14_3_3, STOCHd_14_3_3
        df.ta.adx(length=14, append=True) # Adds ADX_14, DMP_14, DMN_14
        df.ta.cci(length=14, append=True) # Adds CCI_14_0.015
        df.ta.willr(length=14, append=True) # Adds WILLR_14

    def _add_statistical_features_pta(self, df):
        """Adds rolling statistical features using pandas-ta."""
        log_returns_col = 'LOGRET_1' # Default name from pandas-ta
        if log_returns_col not in df.columns:
             print("Warning: Log returns column 'LOGRET_1' not found for statistical features.")
             # Attempt to calculate log returns if missing
             df.ta.log_return(cumulative=False, append=True)
             if log_returns_col not in df.columns:
                  print("Error: Failed to calculate log returns. Skipping statistical features.")
                  return # Skip if log returns still cannot be calculated

        # Ensure log returns column is numeric and handle potential NaNs before rolling calculations
        df[log_returns_col] = pd.to_numeric(df[log_returns_col], errors='coerce')
        df[log_returns_col] = df[log_returns_col].fillna(0) # Fill NaNs with 0 for rolling calculations

        for window in self.window_sizes:
            if window <= 1: continue
            # Use the log returns column directly with pandas-ta indicators that accept 'close' argument
            df.ta.stdev(close=df[log_returns_col], length=window, append=True, suffix=f"_{log_returns_col}") # Adds STDEV_window_LOGRET_1
            df.ta.skew(close=df[log_returns_col], length=window, append=True, suffix=f"_{log_returns_col}") # Adds SKEW_window_LOGRET_1
            df.ta.kurtosis(close=df[log_returns_col], length=window, append=True, suffix=f"_{log_returns_col}") # Adds KURT_window_LOGRET_1

            # Z-Score (calculate manually using pandas rolling mean/std)
            rolling_mean = df[log_returns_col].rolling(window=window).mean()
            rolling_std = df[log_returns_col].rolling(window=window).std()
            # Avoid division by zero or near-zero std dev
            rolling_std = rolling_std.replace(0, np.nan) # Replace 0 std with NaN
            df[f'roll_zscore_{window}'] = ((df[log_returns_col] - rolling_mean) / rolling_std).fillna(0) # Fill resulting NaNs with 0

    def _add_volatility_features_pta(self, df):
        """Adds volatility features using pandas-ta."""
        df.ta.atr(length=14, append=True) # Adds ATR_14

        # GARCH-like proxies (calculate manually using returns)
        log_returns = df.get('LOGRET_1', pd.Series(dtype=float)).fillna(0) # Get log returns safely
        abs_log_returns = np.abs(log_returns)
        for window in self.window_sizes:
             if window <= 1: continue
             df[f'abs_ret_ma_{window}'] = abs_log_returns.rolling(window=window).mean()
             df[f'vol_of_vol_{window}'] = abs_log_returns.rolling(window=window).std().fillna(0) # Fill NaN std dev

    def _add_trend_features_pta(self, df):
        """Adds trend detection features using pandas-ta."""
        # Use ADX already calculated in technical indicators
        # df['adx'] = df['ADX_14'] # Example alias if needed

        # Aroon Indicator
        df.ta.aroon(length=14, append=True) # Adds AROOND_14, AROONU_14, AROONOSC_14

    def _add_seasonality_features(self, df):
        """Adds time-based features (hour, day of week)."""
        if isinstance(df.index, pd.DatetimeIndex):
            # Use .loc to avoid SettingWithCopyWarning
            df.loc[:, 'hour'] = df.index.hour
            df.loc[:, 'day_of_week'] = df.index.dayofweek
            df.loc[:, 'hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df.loc[:, 'hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df.loc[:, 'day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df.loc[:, 'day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        else:
            print("Warning: DataFrame index is not DatetimeIndex, cannot add seasonality features.")

    # --- Update methods remain the same ---
    def update_tick(self, tick_data):
        """Passes tick data update to the feature extractor."""
        self.tick_buffer.append(tick_data)

    def update_order_book(self, order_book):
        """Passes order book update to the feature extractor."""
        self.order_book_buffer.append(order_book)


class FeaturePipeline:
    """
    Manages the overall feature engineering process using the pandas-ta based extractor.
    Handles scaler persistence.
    """
    def __init__(self, config=None):
        self.config = config or {}
        # Pass relevant sub-config down if needed
        self.feature_extractor = FeatureExtractor(self.config.get('feature_extractor_config', self.config))
        self.selected_features = [] # Store names of selected features if selection is done

    def fit(self, market_data, target=None):
        """Fits the feature extractor's scaler."""
        print("Fitting FeaturePipeline (pandas-ta)...")
        self.feature_extractor.fit(market_data)
        # Store feature names available after fitting the scaler
        self.selected_features = self.feature_extractor.feature_names
        print(f"Pipeline fitted. Using {len(self.selected_features)} numeric features for scaling.")
        return self

    def transform(self, market_data):
        """Transforms data using the fitted feature extractor."""
        return self.feature_extractor.transform(market_data)

    def save_scaler(self, path):
        """Saves the fitted scaler state via the FeatureExtractor."""
        self.feature_extractor.save_scaler(path)

    def load_scaler(self, path):
        """Loads the scaler state via the FeatureExtractor."""
        self.feature_extractor.load_scaler(path)

    # --- Update methods remain the same ---
    def update_tick(self, tick_data):
        """Passes tick data update to the feature extractor."""
        self.feature_extractor.update_tick(tick_data)

    def update_order_book(self, order_book):
        """Passes order book update to the feature extractor."""
        self.feature_extractor.update_order_book(order_book)


# --- Factory Function ---
def create_feature_pipeline(config):
    """Factory function to create the FeaturePipeline."""
    feature_config = config.get('feature_config', {})
    return FeaturePipeline(config=feature_config)

