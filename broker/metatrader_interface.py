# broker/metatrader_interface.py

import mt5 as mt5
import time
import pandas as pd
import numpy as np

from main import initialize_mt5


class MetaTraderInterface:
    """
    Provides an interface to interact with the MetaTrader 5 terminal
    for live trading operations (getting data, placing orders, managing positions).
    """
    def __init__(self, config):
        """
        Initializes the MetaTraderInterface.
        Assumes MT5 is already initialized externally before calling methods here,
        or handles initialization internally if needed (less ideal).

        Args:
            config (dict): Configuration dictionary, potentially containing
                           symbol details, default slippage, etc.
        """
        self.config = config or {}
        self.symbol_details = {} # Cache for symbol properties
        self.default_slippage = self.config.get('mt5_config', {}).get('default_slippage', 2) # Slippage in points
        self.retry_attempts = self.config.get('mt5_config', {}).get('retry_attempts', 3)
        self.retry_delay_sec = self.config.get('mt5_config', {}).get('retry_delay_sec', 0.5)

        print("MetaTraderInterface initialized.")
        # It's generally better to initialize/shutdown MT5 in the main script

    def _ensure_connection(self):
        """Checks if MT5 terminal is available."""
        if not mt5.terminal_info():
            print("Error: MetaTrader 5 terminal is not initialized or connection lost.")
            # Optionally try to re-initialize here, but might be problematic
            return False
        return True

    def get_tick(self, symbol):
        """Gets the latest tick data for a symbol."""
        if not self._ensure_connection(): return None

        tick = mt5.symbol_info_tick(symbol)
        if tick:
            # Convert tuple to a more usable dictionary
            return {
                'symbol': symbol,
                'timestamp': pd.to_datetime(tick.time, unit='s'),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last, # Last deal price
                'volume': tick.volume, # Volume for the last deal
                'flags': tick.flags
            }
        else:
            # print(f"Failed to get tick for {symbol}, error code = {mt5.last_error()}")
            return None

    def get_recent_bars(self, symbol, timeframe, count):
        """Gets the most recent 'count' bars for a symbol and timeframe."""
        if not self._ensure_connection(): return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            # print(f"Failed to get recent bars for {symbol} {timeframe}. Error: {mt5.last_error()}")
            return None

        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data = data.set_index('time')
        data.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']]
        return data


    def get_symbol_info(self, symbol):
        """Gets detailed information about a symbol."""
        if symbol in self.symbol_details:
            return self.symbol_details[symbol]

        if not self._ensure_connection(): return None

        info = mt5.symbol_info(symbol)
        if info:
            self.symbol_details[symbol] = info._asdict() # Convert named tuple to dict
            return self.symbol_details[symbol]
        else:
            print(f"Failed to get info for {symbol}, error code = {mt5.last_error()}")
            return None

    def calculate_lots(self, symbol, notional_size, account_balance):
        """
        Calculates the order volume in lots based on desired notional size.
        (This is a simplified calculation, real calculation might involve margin rates).
        """
        info = self.get_symbol_info(symbol)
        if not info: return 0.0

        contract_size = info.get('trade_contract_size', 100000) # e.g., 100,000 units per lot for Forex
        point_value = info.get('point', 0.00001) # Size of one point
        digits = info.get('digits', 5)

        if contract_size <= 0:
            print(f"Warning: Invalid contract size {contract_size} for {symbol}")
            return 0.0

        # Lots = Notional Size / Contract Size
        lots = notional_size / contract_size

        # Apply volume constraints from symbol info
        volume_min = info.get('volume_min', 0.01)
        volume_max = info.get('volume_max', 100.0)
        volume_step = info.get('volume_step', 0.01)

        # Clamp to min/max
        lots = max(volume_min, min(lots, volume_max))

        # Adjust to volume step (round down to nearest step)
        lots = np.floor(lots / volume_step) * volume_step

        # Final check against min volume
        lots = max(volume_min, lots)

        # TODO: Add margin check based on account balance and leverage

        return round(lots, int(-np.log10(volume_step)) if volume_step > 0 else 2) # Round to appropriate digits


    def open_position(self, symbol, order_cmd, volume_lots, price=None, stop_loss=None, take_profit=None, comment="genovo_v2"):
        """Opens a new market or pending order."""
        if not self._ensure_connection(): return None
        info = self.get_symbol_info(symbol)
        if not info: return None

        order_type_map = {'BUY': mt5.ORDER_TYPE_BUY, 'SELL': mt5.ORDER_TYPE_SELL}
        if order_cmd.upper() not in order_type_map:
            print(f"Error: Invalid order_cmd '{order_cmd}'. Use 'BUY' or 'SELL'.")
            return None
        mt5_order_cmd = order_type_map[order_cmd.upper()]

        point = info['point']
        current_price = self.get_tick(symbol)
        if not current_price:
             print(f"Cannot get current price for {symbol} to place order.")
             return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL, # Market execution
            "symbol": symbol,
            "volume": float(volume_lots),
            "type": mt5_order_cmd,
            "price": current_price['ask'] if mt5_order_cmd == mt5.ORDER_TYPE_BUY else current_price['bid'], # Market price
            "sl": float(stop_loss) if stop_loss is not None else 0.0,
            "tp": float(take_profit) if take_profit is not None else 0.0,
            "deviation": int(self.default_slippage), # Allowable slippage in points
            "magic": 12345, # Example magic number
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good till cancelled
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel (check broker requirements, FOK might be needed)
        }

        print(f"Sending order request: {request}")

        # Send order with retries
        result = None
        for attempt in range(self.retry_attempts):
            result = mt5.order_send(request)
            if result:
                # Check result code (see MT5 documentation for codes)
                if result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == mt5.TRADE_RETCODE_PLACED:
                    print(f"Order placed successfully: Deal={result.deal}, Order={result.order}, Comment={result.comment}")
                    return result._asdict()
                else:
                    print(f"Order send failed on attempt {attempt+1}: retcode={result.retcode}, comment={result.comment}. Retrying...")
                    time.sleep(self.retry_delay_sec)
            else:
                print(f"mt5.order_send() failed on attempt {attempt+1}, error code = {mt5.last_error()}. Retrying...")
                time.sleep(self.retry_delay_sec)

        print(f"Order placement failed permanently after {self.retry_attempts} attempts. Last result: {result}")
        if result: return result._asdict() # Return last failed result
        return None


    def close_position(self, symbol, volume_lots=None, position_info=None, comment="genovo_v2_close"):
        """Closes an existing position by symbol or using position_info."""
        if not self._ensure_connection(): return None
        info = self.get_symbol_info(symbol)
        if not info: return None

        if not position_info:
            position_info = self.get_position_info(symbol)
            if not position_info:
                print(f"No open position found for {symbol} to close.")
                return None

        position_id = position_info['ticket']
        position_type = position_info['type'] # 0 = Buy, 1 = Sell
        close_volume = float(volume_lots) if volume_lots is not None else position_info['volume']

        # Determine opposite order type for closing
        close_order_cmd = mt5.ORDER_TYPE_SELL if position_type == 0 else mt5.ORDER_TYPE_BUY

        current_price = self.get_tick(symbol)
        if not current_price:
             print(f"Cannot get current price for {symbol} to close position.")
             return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_order_cmd,
            "position": position_id, # Specify the position ticket to close
            "price": current_price['bid'] if close_order_cmd == mt5.ORDER_TYPE_SELL else current_price['ask'], # Market price
            "deviation": int(self.default_slippage),
            "magic": 12345,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"Sending close request: {request}")

        # Send close order with retries
        result = None
        for attempt in range(self.retry_attempts):
             result = mt5.order_send(request)
             if result:
                 if result.retcode == mt5.TRADE_RETCODE_DONE:
                     print(f"Position {position_id} closed successfully: Deal={result.deal}, Order={result.order}, Comment={result.comment}")
                     return result._asdict()
                 else:
                     print(f"Position close failed on attempt {attempt+1}: retcode={result.retcode}, comment={result.comment}. Retrying...")
                     time.sleep(self.retry_delay_sec)
             else:
                 print(f"mt5.order_send() failed on attempt {attempt+1}, error code = {mt5.last_error()}. Retrying...")
                 time.sleep(self.retry_delay_sec)

        print(f"Position close failed permanently after {self.retry_attempts} attempts. Last result: {result}")
        if result: return result._asdict()
        return None


    def get_positions(self, symbol=None):
        """Gets all open positions or positions for a specific symbol."""
        if not self._ensure_connection(): return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            # Check if the error is just "no positions found" (code 0) or a real error
            if mt5.last_error() == 0:
                 return [] # No positions is not an error
            print(f"Failed to get positions, error code = {mt5.last_error()}")
            return [] # Return empty list on error

        # Convert tuples to dictionaries
        return [p._asdict() for p in positions]

    def get_position_info(self, symbol):
        """Gets position info for a single symbol (assumes only one position per symbol)."""
        positions = self.get_positions(symbol=symbol)
        if positions:
            if len(positions) > 1:
                print(f"Warning: Found multiple positions for {symbol}. Returning the first one.")
            return positions[0]
        return None

    def get_account_summary(self):
        """Gets account balance, equity, margin, etc."""
        if not self._ensure_connection(): return None
        info = mt5.account_info()
        if info:
            return info._asdict()
        else:
            print(f"Failed to get account info, error code = {mt5.last_error()}")
            return None


# --- Factory Function ---
def create_mt5_interface(config):
    """
    Factory function to create the MetaTraderInterface.

    Args:
        config (dict): Main configuration dictionary.

    Returns:
        MetaTraderInterface: The interface instance.
    """
    # Pass relevant sub-config if needed, or the whole config
    return MetaTraderInterface(config=config)

# Example Usage (requires MT5 terminal running and configured)
if __name__ == '__main__':
    # Assumes MT5 is initialized externally or via a config passed to initialize_mt5
    example_config = {
         'mt5_config': {
             'login': 12345678, # Replace with your Exness MT5 login
             'password': 'YOUR_PASSWORD',
             'server': 'Exness-MT5RealX', # Replace with your Exness server
             'path': 'C:/Program Files/MetaTrader 5/terminal64.exe' # Replace with your MT5 path
         },
         'symbol': 'EURUSD'
    }

    print("--- MT5 Interface Example ---")
    if initialize_mt5(example_config):
        mt5_interface = create_mt5_interface(example_config)
        symbol = example_config['symbol']

        print(f"\nGetting tick for {symbol}...")
        tick = mt5_interface.get_tick(symbol)
        print(tick)

        print(f"\nGetting account summary...")
        summary = mt5_interface.get_account_summary()
        print(summary)

        print(f"\nGetting positions for {symbol}...")
        pos = mt5_interface.get_position_info(symbol)
        if pos:
            print(pos)
        else:
            print("No open position found.")

        # --- Example: Open/Close (Use with extreme caution on a demo account!) ---
        # print("\nAttempting to open a small BUY order (DEMO ONLY!)...")
        # lots = mt5_interface.calculate_lots(symbol, 1000, summary['balance']) # Calc lots for $1000 notional
        # if lots > 0:
        #      open_result = mt5_interface.open_position(symbol, 'BUY', lots)
        #      print(f"Open Result: {open_result}")
        #      if open_result and open_result['retcode'] == mt5.TRADE_RETCODE_DONE:
        #          time.sleep(5) # Wait 5 seconds
        #          print("\nAttempting to close the position...")
        #          close_result = mt5_interface.close_position(symbol)
        #          print(f"Close Result: {close_result}")
        # else:
        #      print(f"Calculated lot size is zero ({lots}), cannot open order.")
        # ----------------------------------------------------------------------

        print("\nShutting down MT5 connection.")
        mt5.shutdown()
    else:
        print("\nFailed to initialize MT5 for example.")

    print("--- Example Finished ---")

