# data/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler # Keep scaler if needed for other steps
import numpy as np
# import mt5 as mt5 # No longer needed here if loading is centralized

class DataPreprocessor:
    """
    Handles cleaning and basic preprocessing of market data
    *after* it has been loaded (e.g., from MT5 in main.py).
    """
    def __init__(self, config=None):
        """
        Initializes the DataPreprocessor.

        Args:
            config (dict, optional): Configuration dictionary (currently unused).
        """
        self.config = config or {}
        # Scaler might be used if specific pre-feature-extraction scaling is needed,
        # but primary scaling happens in FeatureExtractor.
        self.scaler = None
        print("DataPreprocessor initialized.")

    # Removed load_data_from_mt5 as it's redundant with main.py's loading

    def clean_data(self, data, symbol=""):
        """
        Performs basic data cleaning on the input DataFrame.
        Assumes data has already been loaded.

        Args:
            data (pd.DataFrame): Input DataFrame (assumed to have OHLCV columns
                                 and potentially a datetime index).
            symbol (str, optional): Symbol name for logging purposes.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if data is None or data.empty:
            print(f"No data provided for cleaning [{symbol}].")
            return pd.DataFrame()

        print(f"Cleaning data for {symbol} ({len(data)} rows)...")
        df = data.copy() # Work on a copy

        # Ensure required columns exist (case-insensitive)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.lower() # Standardize to lower case
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Data for {symbol} is missing required columns: {missing_cols}")

        # Ensure data types are numeric
        for col in required_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle NaNs introduced by coercion or already present
        initial_nans = df.isnull().sum().sum()
        df = df.ffill() # Forward fill first
        df = df.bfill() # Backward fill remaining NaNs at the start
        final_nans = df.isnull().sum().sum()
        if initial_nans > 0:
            print(f"NaNs handled for {symbol}: {initial_nans} -> {final_nans}")

        # Remove rows where essential price data might still be NaN (shouldn't happen after ffill/bfill)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        # Remove duplicate timestamps (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            initial_len = len(df)
            df = df[~df.index.duplicated(keep='first')]
            if len(df) < initial_len:
                print(f"Removed {initial_len - len(df)} duplicate timestamp entries for {symbol}.")
        # else:
             # Optional: If index is not datetime, check for duplicate rows based on columns?
             # print("Warning: Index is not DatetimeIndex, cannot check for duplicate timestamps.")


        # Optional: Add more cleaning steps like outlier filtering if needed
        # Example: Filter out bars with zero range (H=L) or zero volume
        # zero_range_count = len(df[df['high'] == df['low']])
        # zero_volume_count = len(df[df['volume'] == 0])
        # if zero_range_count > 0: print(f"Found {zero_range_count} bars with zero range for {symbol}.")
        # if zero_volume_count > 0: print(f"Found {zero_volume_count} bars with zero volume for {symbol}.")
        # df = df[(df['high'] != df['low']) & (df['volume'] > 0)]


        print(f"Data cleaning finished for {symbol}. Final shape: {df.shape}")
        return df

    # Removed preprocess_symbol_data as loading is now external

# --- Factory Function ---
def create_preprocessor(config=None):
    """
    Factory function to create the DataPreprocessor.
    """
    return DataPreprocessor(config=config)

# Example Usage (can be removed)
if __name__ == '__main__':
    # Create dummy data for testing cleaning
    dates = pd.date_range(start='2023-01-01', periods=10, freq='h')
    dummy_data = pd.DataFrame({
        'open': [1.0, 1.1, 1.1, 1.2, np.nan, 1.3, 1.4, 1.4, 1.5, 1.6],
        'high': [1.1, 1.2, 1.15, 1.3, 1.35, 1.4, 1.5, 1.45, 1.6, 1.7],
        'low': [0.9, 1.0, 1.05, 1.1, 1.2, 1.25, 1.3, 1.4, 1.4, 1.5],
        'close': [1.1, 1.15, 1.05, 1.25, 1.3, 1.35, 1.45, 1.42, 1.55, 1.65],
        'volume': [100, 150, 120, 200, 180, 220, 250, 0, 300, 350]
    }, index=dates)
    # Add a duplicate index
    dummy_data = pd.concat([dummy_data, dummy_data.iloc[-1:]])
    # Add a NaN row start
    nan_row = pd.DataFrame({c: [np.nan] for c in dummy_data.columns}, index=[dates[0]-pd.Timedelta(hours=1)])
    dummy_data = pd.concat([nan_row, dummy_data])


    print("--- Original Dummy Data ---")
    print(dummy_data)

    preprocessor = create_preprocessor()
    cleaned_data = preprocessor.clean_data(dummy_data, symbol="DUMMY")

    print("\n--- Cleaned Dummy Data ---")
    print(cleaned_data)
    print("\nInfo:")
    cleaned_data.info()

