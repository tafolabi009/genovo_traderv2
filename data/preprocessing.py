# data/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler # Keep scaler if needed for other steps
import numpy as np
import mt5 as mt5 # Import for MT5 constants if needed later
import time

class DataPreprocessor:
    """
    Handles loading (primarily from MT5 via main script now), cleaning,
    and basic preprocessing of market data before feature engineering.
    """
    def __init__(self, config=None):
        """
        Initializes the DataPreprocessor.

        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.config = config or {}
        # Scaler might be used if specific pre-feature-extraction scaling is needed,
        # but primary scaling happens in FeatureExtractor.
        self.scaler = None
        print("DataPreprocessor initialized.")

    def load_data_from_mt5(self, symbol, timeframe_str, num_bars):
        """
        Loads historical data directly from MetaTrader 5 terminal.
        (Note: This logic is duplicated in main.py, consider centralizing later if needed)

        Args:
            symbol (str): The trading symbol.
            timeframe_str (str): Timeframe string (e.g., 'M1', 'H1').
            num_bars (int): Number of bars to download.

        Returns:
            pd.DataFrame or None: Loaded data or None on failure.
        """
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
        }
        timeframe = timeframe_map.get(timeframe_str.upper())
        if timeframe is None:
            print(f"Error: Invalid timeframe '{timeframe_str}'.")
            return None

        print(f"DataPreprocessor: Attempting to load {num_bars} bars of {symbol} {timeframe_str} data from MT5...")

        if not mt5.terminal_info():
            print("Error: MT5 terminal not initialized.")
            # Initialization should happen in main.py before calling this
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found in MetaTrader 5.")
            return None
        if not symbol_info.visible:
            print(f"Symbol {symbol} is not visible, enabling...")
            if not mt5.symbol_select(symbol, True):
                print(f"mt5.symbol_select failed, error = {mt5.last_error()}")
                return None
            time.sleep(1)

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            print(f"No data received from MT5 for {symbol} {timeframe_str}. Error: {mt5.last_error()}")
            return None

        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data = data.set_index('time')
        data.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'}, inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']]
        print(f"DataPreprocessor: Loaded {len(data)} bars for {symbol}.")
        return data


    def clean_data(self, data, symbol=""):
        """
        Performs basic data cleaning on the input DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame (assumed to have OHLCV columns).
            symbol (str, optional): Symbol name for logging purposes.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if data is None or data.empty:
            print(f"No data provided for cleaning [{symbol}].")
            return pd.DataFrame()

        print(f"Cleaning data for {symbol} ({len(data)} rows)...")
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data for {symbol} is missing required columns: {missing_cols}")

        # Handle NaNs
        initial_nans = data.isnull().sum().sum()
        data = data.ffill() # Forward fill first
        data = data.bfill() # Backward fill remaining NaNs at the start
        final_nans = data.isnull().sum().sum()
        if initial_nans > 0:
            print(f"NaNs handled for {symbol}: {initial_nans} -> {final_nans}")

        # Remove duplicate timestamps (if index is datetime)
        if isinstance(data.index, pd.DatetimeIndex):
            initial_len = len(data)
            data = data[~data.index.duplicated(keep='first')]
            if len(data) < initial_len:
                print(f"Removed {initial_len - len(data)} duplicate timestamp entries for {symbol}.")
        else:
             print("Warning: Index is not DatetimeIndex, cannot check for duplicate timestamps.")


        # Optional: Add more cleaning steps like outlier filtering if needed
        # Example: Filter out bars with zero range (H=L) or extreme volume spikes

        print(f"Data cleaning finished for {symbol}.")
        return data

    def preprocess_symbol_data(self, symbol, timeframe_str, num_bars):
        """
        Loads and cleans data for a specific symbol from MT5.
        (This might be redundant if main.py handles loading/cleaning directly).

        Args:
            symbol (str): The trading symbol.
            timeframe_str (str): Timeframe string (e.g., 'M1', 'H1').
            num_bars (int): Number of bars to download.

        Returns:
            pd.DataFrame: Preprocessed data ready for feature engineering.
        """
        data = self.load_data_from_mt5(symbol, timeframe_str, num_bars)
        data = self.clean_data(data, symbol)
        return data

# --- Factory Function ---
def create_preprocessor(config=None):
    """
    Factory function to create the DataPreprocessor.
    """
    return DataPreprocessor(config=config)

# Example Usage (can be removed)
# Note: Requires MT5 connection initialized externally
if __name__ == '__main__':
    # Example: Initialize MT5 connection here for standalone testing
    # (Replace with your actual details and path)
    # if not mt5.initialize(login=123456, password="PASSWORD", server="SERVER", path="PATH_TO_MT5"):
    #     print("MT5 initialization failed for example.")
    # else:
    #     print("MT5 Initialized for Preprocessor Example.")
    #     try:
    #         preprocessor = create_preprocessor()
    #         # Test loading and cleaning
    #         eurusd_data = preprocessor.preprocess_symbol_data('EURUSD', 'M1', 1000)
    #         if eurusd_data is not None and not eurusd_data.empty:
    #             print("\nProcessed EURUSD Data Head:")
    #             print(eurusd_data.head())
    #             print("\nProcessed EURUSD Data Info:")
    #             eurusd_data.info()
    #         else:
    #              print("Failed to process EURUSD data.")

    #     except Exception as e:
    #         print(f"\nError during example run: {e}")
    #     finally:
    #          if mt5.terminal_info(): mt5.shutdown() # Shutdown MT5 if initialized here
    print("DataPreprocessor example finished (requires external MT5 init).")


