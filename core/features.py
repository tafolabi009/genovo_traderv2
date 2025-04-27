# core/features.py

import numpy as np
import pandas as pd
from scipy import stats
# from statsmodels.tsa.stattools import adfuller # Not used currently
from sklearn.preprocessing import StandardScaler
# import talib # Removed talib
import pandas_ta  # Import pandas_ta
from collections import deque
import joblib # Keep joblib in case we re-add scaler persistence later
import os

class FeatureExtractor:
    """
    Comprehensive feature extraction pipeline using pandas-ta.
    Generates price, volume, technical, statistical, volatility,
    trend, seasonality features.
    NOTE: Scaler saving/loading methods are removed for now.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.window_sizes = self.config.get('window_sizes', [5, 10, 20, 50, 100])
        self.feature_names = []

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.tick_buffer = deque(maxlen=200)
        self.order_book_buffer = deque(maxlen=200)

        self.feature_means = None
        self.feature_stds = None
        print("FeatureExtractor initialized (using pandas-ta).")

    def fit(self, market_data):
        """Fits the scaler based on historical market data."""
        features = self._extract_all_features(market_data, scale=False)
        features_cleaned = features.replace([np.inf, -np.inf], np.nan).dropna()

        if features_cleaned.empty:
            print("Warning: No valid features generated during fit. Scaler not fitted.")
            self.is_fitted = False
            return self

        try:
            # Ensure columns used for fitting are numeric
            numeric_cols = features_cleaned.select_dtypes(include=np.number).columns
            if len(numeric_cols) != len(features_cleaned.columns):
                 print(f"Warning: Non-numeric columns found: {features_cleaned.select_dtypes(exclude=np.number).columns.tolist()}. Fitting scaler only on numeric columns.")

            if not numeric_cols.empty:
                self.scaler.fit(features_cleaned[numeric_cols])
                self.is_fitted = True
                self.feature_names = numeric_cols.tolist() # Store only names of scaled features
                self.feature_means = features_cleaned[numeric_cols].mean()
                self.feature_stds = features_cleaned[numeric_cols].std()
                print(f"FeatureExtractor fitted with {len(self.feature_names)} numeric features.")
            else:
                 print("Warning: No numeric features found to fit the scaler.")
                 self.is_fitted = False

        except ValueError as e:
             print(f"Error fitting scaler: {e}. Check if data has constant features.")
             self.is_fitted = False

        return self

    def transform(self, market_data):
        """Transforms new market data into a scaled feature matrix."""
        features = self._extract_all_features(market_data, scale=False)
        features_cleaned = features.replace([np.inf, -np.inf], np.nan).dropna()

        if features_cleaned.empty:
            return pd.DataFrame()

        if self.is_fitted:
            # Only attempt to scale columns that the scaler was fitted on
            cols_to_scale = [col for col in self.feature_names if col in features_cleaned.columns]
            missing_cols = [col for col in self.feature_names if col not in features_cleaned.columns]
            if missing_cols:
                 print(f"Warning: Features missing during transform (were present during fit): {missing_cols}")

            if not cols_to_scale:
                 print("Warning: No features available to scale matching the fitted scaler.")
                 # Return unscaled but cleaned features if none match
                 return features_cleaned

            # Select only the columns to be scaled
            features_to_scale_df = features_cleaned[cols_to_scale]

            try:
                 scaled_array = self.scaler.transform(features_to_scale_df)
                 scaled_features = pd.DataFrame(scaled_array, index=features_to_scale_df.index, columns=cols_to_scale)
                 # Return only the scaled features
                 return scaled_features
            except ValueError as e:
                 print(f"Error transforming features: {e}. Returning unscaled features.")
                 return features_cleaned # Fallback to unscaled
            except Exception as e:
                 print(f"Unexpected error during feature transformation: {e}")
                 return features_cleaned # Fallback to unscaled
        else:
            print("Warning: Scaler not fitted. Returning unscaled features.")
            return features_cleaned # Return unscaled but cleaned features

    # --- Scaler save/load methods removed for now ---
    # def save_scaler(...): pass
    # def load_scaler(...): pass

    def _extract_all_features(self, market_data, scale=True):
        """Internal method to orchestrate feature extraction using pandas-ta."""
        df = market_data.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Market data must contain columns: {required_cols}")

        # pandas-ta works directly on the DataFrame
        # Ensure columns are lowercase if pandas-ta expects it (usually does)
        df.columns = df.columns.str.lower()
        required_cols_lower = [c.lower() for c in required_cols]
        if not all(col in df.columns for col in required_cols_lower):
             raise ValueError(f"Market data must contain lowercase columns: {required_cols_lower}")


        # --- Use pandas-ta Strategy or individual indicators ---
        # Example using a built-in strategy (adjust or use individual calls)
        # df.ta.strategy("CommonStrategy", verbose=False) # Example strategy

        # --- Or call indicators individually ---
        self._add_price_features_pta(df)
        self._add_volume_features_pta(df)
        self._add_technical_indicators_pta(df)
        self._add_statistical_features_pta(df)
        self._add_volatility_features_pta(df)
        self._add_trend_features_pta(df)
        self._add_seasonality_features(df) # Seasonality doesn't use pandas-ta

        # Rename columns generated by pandas-ta if needed (they often have standard names)
        # Example: df.rename(columns={'SMA_10': 'sma_10'}, inplace=True)

        # Select only the calculated feature columns (excluding original OHLCV)
        # Be careful with column names generated by pandas-ta
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        features = df.drop(columns=original_cols, errors='ignore')

        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    # --- Individual Feature Group Methods using pandas-ta ---

    def _add_price_features_pta(self, df):
        """Adds price-based features using pandas-ta."""
        df.ta.log_return(cumulative=False, append=True) # Adds 'LOGRET_1'
        df.ta.percent_return(cumulative=False, append=True) # Adds 'PERCENT_1'

        # Ratios (calculate manually)
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        for window in self.window_sizes:
            df.ta.sma(length=window, append=True) # Adds SMA_window
            df.ta.ema(length=window, append=True) # Adds EMA_window
            # Price relative to MA (calculate manually)
            df[f'price_vs_sma_{window}'] = df['close'] / df[f'SMA_{window}']
            df[f'price_vs_ema_{window}'] = df['close'] / df[f'EMA_{window}']
            df.ta.mom(length=window, append=True) # Adds MOM_window

        # MA Crossover (calculate manually)
        if 5 in self.window_sizes and 20 in self.window_sizes:
             df['sma_5_20_diff'] = df['SMA_5'] - df['SMA_20']
             df['ema_5_20_diff'] = df['EMA_5'] - df['EMA_20']
        # No return needed, modifies df inplace

    def _add_volume_features_pta(self, df):
        """Adds volume-based features using pandas-ta."""
        # Keep original volume if needed, pandas-ta might use it internally
        # df['volume_orig'] = df['volume']
        df['log_volume'] = np.log1p(df['volume'])

        for window in self.window_sizes:
            df.ta.sma(close='volume', length=window, prefix='VOL', append=True) # Adds VOL_SMA_window
            # Volume relative to its SMA (calculate manually)
            df[f'volume_vs_sma_{window}'] = df['volume'] / df[f'VOL_SMA_{window}']

        # On-Balance Volume (OBV)
        df.ta.obv(append=True) # Adds 'OBV'
        # No return needed

    def _add_technical_indicators_pta(self, df):
        """Adds common technical indicators using pandas-ta."""
        df.ta.rsi(length=14, append=True) # Adds RSI_14
        df.ta.macd(fast=12, slow=26, signal=9, append=True) # Adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df.ta.bbands(length=20, std=2, append=True) # Adds BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True) # Adds STOCHk_14_3_3, STOCHd_14_3_3
        df.ta.adx(length=14, append=True) # Adds ADX_14, DMP_14, DMN_14
        df.ta.cci(length=14, append=True) # Adds CCI_14_0.015
        df.ta.willr(length=14, append=True) # Adds WILLR_14
        # No return needed

    def _add_statistical_features_pta(self, df):
        """Adds rolling statistical features using pandas-ta."""
        # Use log returns calculated earlier
        log_returns_col = 'LOGRET_1' # Default name from pandas-ta
        if log_returns_col not in df.columns:
             print("Warning: Log returns column 'LOGRET_1' not found for statistical features.")
             return # Skip if log returns weren't calculated

        for window in self.window_sizes:
            if window <= 1: continue
            df.ta.stdev(length=window, append=True, col=log_returns_col) # Adds STDEV_window
            df.ta.skew(length=window, append=True, col=log_returns_col) # Adds SKEW_window
            df.ta.kurtosis(length=window, append=True, col=log_returns_col) # Adds KURT_window
            # Z-Score (calculate manually using pandas-ta mean/std)
            rolling_mean = df[log_returns_col].rolling(window=window).mean() # pandas rolling
            rolling_std_col = f'STDEV_{window}' # Name from pandas-ta stdev
            if rolling_std_col in df.columns:
                 df[f'roll_zscore_{window}'] = (df[log_returns_col] - rolling_mean) / df[rolling_std_col]
            else:
                 print(f"Warning: Rolling std dev column '{rolling_std_col}' not found for Z-Score calculation.")
        # No return needed

    def _add_volatility_features_pta(self, df):
        """Adds volatility features using pandas-ta."""
        df.ta.atr(length=14, append=True) # Adds ATR_14

        # Use rolling standard deviation already calculated if needed
        # df['volatility_20'] = df['STDEV_20'] # Example alias

        # GARCH-like proxies (calculate manually using returns)
        log_returns = df.get('LOGRET_1', pd.Series(dtype=float)).fillna(0) # Get log returns safely
        abs_log_returns = np.abs(log_returns)
        for window in self.window_sizes:
             if window <= 1: continue
             df[f'abs_ret_ma_{window}'] = abs_log_returns.rolling(window=window).mean()
             df[f'vol_of_vol_{window}'] = abs_log_returns.rolling(window=window).std()
        # No return needed

    def _add_trend_features_pta(self, df):
        """Adds trend detection features using pandas-ta."""
        # Use ADX already calculated
        # df['adx'] = df['ADX_14']

        # Aroon Indicator
        df.ta.aroon(length=14, append=True) # Adds AROOND_14, AROONU_14, AROONOSC_14
        # No return needed

    def _add_seasonality_features(self, df):
        """Adds time-based features (hour, day of week)."""
        # This part remains the same as it doesn't use technical libraries
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        else:
            print("Warning: DataFrame index is not DatetimeIndex, cannot add seasonality features.")
        # No return needed

    def update_tick(self, tick_data):
        """Passes tick data update to the feature extractor."""
        self.tick_buffer.append(tick_data)

    def update_order_book(self, order_book):
        """Passes order book update to the feature extractor."""
        self.order_book_buffer.append(order_book)


class FeaturePipeline:
    """
    Manages the overall feature engineering process using the pandas-ta based extractor.
    Scaler persistence methods removed for now.
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

    # --- Scaler save/load methods removed ---
    # def save_scaler(...): pass
    # def load_scaler(...): pass

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

