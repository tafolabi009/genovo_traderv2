# broker/metatrader_interface.py

import mt5 as mt5
import time
import pandas as pd
import numpy as np
import logging # Import logging

# Get logger instance (assuming setup_logger is called in main)
logger = logging.getLogger("genovo_traderv2")

class MetaTraderInterface:
    """
    Provides an interface to interact with the MetaTrader 5 terminal
    for live trading operations (getting data, placing orders, managing positions).
    Includes improved error handling and logging.
    """
    def __init__(self, config):
        """
        Initializes the MetaTraderInterface.
        Assumes MT5 is already initialized externally.

        Args:
            config (dict): Configuration dictionary, potentially containing
                           symbol details, default slippage, etc.
        """
        self.config = config or {}
        self.symbol_details = {} # Cache for symbol properties
        self.mt5_config = self.config.get('mt5_config', {})
        self.default_slippage = self.mt5_config.get('default_slippage', 3) # Slippage in points
        self.retry_attempts = self.mt5_config.get('retry_attempts', 3)
        self.retry_delay_sec = self.mt5_config.get('retry_delay_sec', 0.5)
        self.default_magic = self.mt5_config.get('magic_number', 12345) # Default magic number

        logger.info("MetaTraderInterface initialized.")

    def _ensure_connection(self):
        """Checks if MT5 terminal is available."""
        term_info = mt5.terminal_info()
        if not term_info:
            logger.error("MetaTrader 5 terminal is not connected or available.")
            # Attempting re-initialization here is risky, should be handled externally.
            return False
        # Optional: Check connection status more specifically if needed
        # if not term_info.connected:
        #     logger.warning("MT5 Terminal Info available, but not connected to trade server.")
        #     return False # Or True depending on whether connection is needed for the operation
        return True

    def get_tick(self, symbol):
        """Gets the latest tick data for a symbol."""
        if not self._ensure_connection(): return None

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                # Convert tuple to a more usable dictionary
                return {
                    'symbol': symbol,
                    'timestamp': pd.to_datetime(tick.time, unit='s', utc=True), # Add UTC timezone
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last, # Last deal price
                    'volume': tick.volume, # Volume for the last deal
                    'flags': tick.flags
                }
            else:
                last_error = mt5.last_error()
                logger.warning(f"Failed to get tick for {symbol}, error code = {last_error}")
                return None
        except Exception as e:
            logger.error(f"Exception getting tick for {symbol}: {e}", exc_info=True)
            return None

    def get_recent_bars(self, symbol, timeframe, count):
        """Gets the most recent 'count' bars for a symbol and timeframe."""
        if not self._ensure_connection(): return None

        try:
            # Ensure symbol is available/visible
            if not self._ensure_symbol_visible(symbol):
                 return None

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                last_error = mt5.last_error()
                # Error code 0 might just mean no history, check context
                if last_error != 0:
                     logger.warning(f"Failed to get recent bars for {symbol} {timeframe}. Error: {last_error}")
                else:
                     logger.debug(f"No recent bars returned for {symbol} {timeframe} (mt5.last_error()=0).")
                return None

            data = pd.DataFrame(rates)
            # Convert time to datetime with UTC timezone
            data['time'] = pd.to_datetime(data['time'], unit='s', utc=True)
            data = data.set_index('time')
            # Standardize column names to lowercase
            data.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                'tick_volume': 'volume', 'real_volume': 'real_volume', 'spread': 'spread'
            }, inplace=True)
            # Select common columns, keep others if they exist
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
            data = data[[col for col in cols_to_keep if col in data.columns]]
            return data
        except Exception as e:
            logger.error(f"Exception getting recent bars for {symbol} {timeframe}: {e}", exc_info=True)
            return None

    def _ensure_symbol_visible(self, symbol):
        """Checks if a symbol exists and is visible in MarketWatch, enables if not."""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Symbol {symbol} not found in MetaTrader 5.")
                return False
            if not info.visible:
                logger.info(f"Symbol {symbol} is not visible, attempting to enable...")
                if not mt5.symbol_select(symbol, True):
                    last_error = mt5.last_error()
                    logger.error(f"mt5.symbol_select failed for {symbol}, error = {last_error}")
                    return False
                time.sleep(0.5) # Short pause after enabling
                info = mt5.symbol_info(symbol) # Re-check
                if not info or not info.visible:
                     logger.error(f"Failed to enable symbol {symbol} after attempting.")
                     return False
                logger.info(f"Symbol {symbol} enabled successfully.")
            return True
        except Exception as e:
            logger.error(f"Exception checking/enabling symbol {symbol}: {e}", exc_info=True)
            return False

    def get_symbol_info(self, symbol):
        """Gets detailed information about a symbol, caching results."""
        if symbol in self.symbol_details:
            return self.symbol_details[symbol]

        if not self._ensure_connection(): return None
        if not self._ensure_symbol_visible(symbol): return None # Ensure visible before getting info

        try:
            info = mt5.symbol_info(symbol)
            if info:
                self.symbol_details[symbol] = info._asdict() # Convert named tuple to dict
                return self.symbol_details[symbol]
            else:
                last_error = mt5.last_error()
                logger.warning(f"Failed to get info for {symbol}, error code = {last_error}")
                return None
        except Exception as e:
            logger.error(f"Exception getting symbol info for {symbol}: {e}", exc_info=True)
            return None

    def calculate_lots(self, symbol, notional_size, account_currency='USD'):
        """
        Calculates the order volume in lots based on desired notional size.
        Requires account currency and symbol info for accurate calculation.

        Args:
            symbol (str): The trading symbol (e.g., 'EURUSD').
            notional_size (float): The desired trade size in the account currency (e.g., 1000 USD).
            account_currency (str): The currency of the trading account (e.g., 'USD').

        Returns:
            float: Volume in lots, rounded to the symbol's volume step. Returns 0.0 on error.
        """
        info = self.get_symbol_info(symbol)
        if not info: logger.error(f"Cannot calculate lots for {symbol}: Symbol info unavailable."); return 0.0

        contract_size = info.get('trade_contract_size')
        volume_min = info.get('volume_min')
        volume_max = info.get('volume_max')
        volume_step = info.get('volume_step')
        base_currency = info.get('currency_base') # e.g., EUR in EURUSD
        quote_currency = info.get('currency_profit') # e.g., USD in EURUSD

        if not all([contract_size, volume_min, volume_max, volume_step, base_currency, quote_currency]):
             logger.error(f"Cannot calculate lots for {symbol}: Missing critical symbol info (contract size, volume limits, currencies).")
             return 0.0
        if contract_size <= 0 or volume_step <= 0:
             logger.error(f"Cannot calculate lots for {symbol}: Invalid contract size ({contract_size}) or volume step ({volume_step}).")
             return 0.0

        # --- Determine Conversion Rate ---
        # We need to convert the notional size (in account currency) to the value of 1 lot in the account currency.
        # Value of 1 Lot = Contract Size * Price_of_Base_in_Account_Currency
        conversion_rate = 1.0
        if base_currency != account_currency:
            # Need rate to convert Base Currency to Account Currency
            # Example: If symbol=EURUSD, base=EUR, account=USD, we need EURUSD price.
            # Example: If symbol=USDJPY, base=USD, account=EUR, we need USDEUR price (1/EURUSD).
            # Example: If symbol=GBPJPY, base=GBP, account=USD, we need GBPUSD price.
            conversion_symbol = None
            invert_rate = False
            if base_currency + account_currency == symbol: # e.g. EURUSD, account USD
                 conversion_symbol = symbol
            elif account_currency + base_currency == symbol: # e.g., USDJPY, account JPY (unlikely but possible)
                 conversion_symbol = symbol; invert_rate = True # Need 1 / USDJPY
            else:
                 # Try common pairs (e.g., if base=GBP, account=USD, look for GBPUSD)
                 common_pair = base_currency + account_currency
                 common_pair_inverted = account_currency + base_currency
                 if self.get_symbol_info(common_pair): conversion_symbol = common_pair
                 elif self.get_symbol_info(common_pair_inverted): conversion_symbol = common_pair_inverted; invert_rate = True
                 else: logger.warning(f"Cannot find direct conversion rate for {base_currency} to {account_currency}. Assuming rate=1.0. Lot size may be inaccurate."); conversion_symbol = None

            if conversion_symbol:
                 tick = self.get_tick(conversion_symbol)
                 if tick and tick.get('ask') > 0: # Use ask price for conversion approx
                      conversion_rate = tick['ask']
                      if invert_rate:
                           conversion_rate = 1.0 / conversion_rate
                      logger.debug(f"Conversion rate ({conversion_symbol}): {conversion_rate:.5f} (Inverted: {invert_rate})")
                 else:
                      logger.warning(f"Could not get tick for conversion symbol {conversion_symbol}. Assuming rate=1.0. Lot size may be inaccurate.")
                      conversion_rate = 1.0
            else:
                 conversion_rate = 1.0 # Fallback if no conversion symbol found

        # --- Calculate Lots ---
        # Value of 1 lot in account currency
        value_per_lot = contract_size * conversion_rate
        if value_per_lot <= 0:
             logger.error(f"Cannot calculate lots for {symbol}: Calculated value per lot is zero or negative ({value_per_lot}).")
             return 0.0

        # Lots = Desired Notional Size / Value per Lot
        lots = notional_size / value_per_lot

        # Apply volume constraints
        lots = max(volume_min, min(lots, volume_max)) # Clamp to min/max
        # Adjust to volume step (round down to nearest step)
        lots = np.floor(lots / volume_step) * volume_step
        lots = max(volume_min, lots) # Ensure it's still >= min after rounding

        # Round to appropriate digits based on volume step
        digits = int(-np.log10(volume_step)) if volume_step > 0 else 2
        lots = round(lots, digits)

        # TODO: Add margin check based on account balance, leverage, and required margin for the symbol
        # required_margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lots, current_price) ...

        logger.debug(f"Calculated lots for {symbol}: {lots:.{digits}f} (Notional: {notional_size} {account_currency}, Value/Lot: {value_per_lot:.2f} {account_currency})")
        return lots


    def _send_order_request(self, request):
        """Internal helper to send order request with retries and logging."""
        result = None
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                result = mt5.order_send(request)
                last_error = mt5.last_error() # Get error immediately after send
                if result:
                    # Check MT5 return codes
                    # See: https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_return_codes
                    if result.retcode == mt5.TRADE_RETCODE_DONE or \
                       result.retcode == mt5.TRADE_RETCODE_PLACED or \
                       result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL: # Consider partial fills success
                        logger.info(f"Order request successful (Attempt {attempt+1}): Code={result.retcode}, Deal={result.deal}, Order={result.order}, Volume={result.volume}, Price={result.price}, Comment={result.comment}")
                        return result._asdict() # Success
                    else:
                        logger.warning(f"Order send failed (Attempt {attempt+1}): RetCode={result.retcode} ({mt5.trade_retcode_description(result.retcode)}), Comment={result.comment}. Retrying...")
                        # Optional: Check specific retcodes for non-retryable errors
                        if result.retcode in [mt5.TRADE_RETCODE_INVALID_ACCOUNT, mt5.TRADE_RETCODE_TRADE_DISABLED, mt5.TRADE_RETCODE_MARKET_CLOSED, mt5.TRADE_RETCODE_NO_MONEY, mt5.TRADE_RETCODE_INVALID_PRICE, mt5.TRADE_RETCODE_INVALID_STOPS, mt5.TRADE_RETCODE_INVALID_VOLUME]:
                             logger.error(f"Non-retryable error code {result.retcode}. Aborting order.")
                             return result._asdict() # Return failed result
                else:
                    # order_send itself returned None
                    logger.warning(f"mt5.order_send() returned None (Attempt {attempt+1}), last_error = {last_error}. Retrying...")

            except Exception as e:
                logger.error(f"Exception during mt5.order_send() (Attempt {attempt+1}): {e}", exc_info=True)
                # Break on unexpected exception

            # Wait before retrying
            if attempt < self.retry_attempts - 1:
                 time.sleep(self.retry_delay_sec)

        # If loop finishes without success
        logger.error(f"Order placement failed permanently after {self.retry_attempts} attempts. Last result: {result}, Last Error Code: {last_error}")
        if result: return result._asdict() # Return last failed result if available
        return None # Return None if order_send consistently returned None

    def open_position(self, symbol, order_cmd, volume_lots, stop_loss=None, take_profit=None, price=None, comment="genovo_v2"):
        """
        Opens a new market or pending order.

        Args:
            symbol (str): Trading symbol.
            order_cmd (str): 'BUY' or 'SELL' (for market orders).
                             TODO: Add 'BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP'.
            volume_lots (float): Order volume in lots.
            stop_loss (float, optional): Stop loss price level.
            take_profit (float, optional): Take profit price level.
            price (float, optional): Entry price for pending orders.
            comment (str): Order comment.

        Returns:
            dict or None: Result dictionary from MT5 or None on failure.
        """
        if not self._ensure_connection(): return None
        info = self.get_symbol_info(symbol)
        if not info: return None

        order_type_map = {'BUY': mt5.ORDER_TYPE_BUY, 'SELL': mt5.ORDER_TYPE_SELL}
        if order_cmd.upper() not in order_type_map:
            logger.error(f"Invalid order_cmd '{order_cmd}'. Use 'BUY' or 'SELL'.")
            return None
        mt5_order_cmd = order_type_map[order_cmd.upper()]

        point = info['point']
        digits = info['digits']

        # Get current prices for market order execution
        current_tick = self.get_tick(symbol)
        if not current_tick:
             logger.error(f"Cannot get current tick for {symbol} to place market order.")
             return None

        # Determine execution price based on order type
        exec_price = current_tick['ask'] if mt5_order_cmd == mt5.ORDER_TYPE_BUY else current_tick['bid']

        # Round SL/TP to the correct number of digits for the symbol
        sl_price = round(float(stop_loss), digits) if stop_loss is not None else 0.0
        tp_price = round(float(take_profit), digits) if take_profit is not None else 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL, # Market execution
            "symbol": symbol,
            "volume": float(volume_lots),
            "type": mt5_order_cmd,
            "price": exec_price, # Current market price for market orders
            "sl": sl_price,
            "tp": tp_price,
            "deviation": int(self.default_slippage), # Allowable slippage in points
            "magic": self.default_magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good till cancelled
            # Consider FOK vs IOC based on broker/strategy needs
            "type_filling": mt5.ORDER_FILLING_FOK, # Fill Or Kill (ensure full volume or cancel)
            # "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
        }

        logger.info(f"Sending Market Order Request: {request}")
        return self._send_order_request(request)


    def close_position(self, symbol, volume_lots=None, position_info=None, comment="genovo_v2_close"):
        """
        Closes an existing position by symbol or using position_info.

        Args:
            symbol (str): Trading symbol.
            volume_lots (float, optional): Volume to close (defaults to full position).
            position_info (dict, optional): Pre-fetched position details (ticket, volume, type).
            comment (str): Order comment.

        Returns:
            dict or None: Result dictionary from MT5 or None on failure.
        """
        if not self._ensure_connection(): return None
        info = self.get_symbol_info(symbol)
        if not info: return None

        if not position_info:
            position_info = self.get_position_info(symbol) # Fetch if not provided
            if not position_info:
                logger.warning(f"No open position found for {symbol} to close.")
                return None # Not an error, just nothing to close

        position_id = position_info['ticket']
        position_type = position_info['type'] # 0 = Buy, 1 = Sell
        position_volume = position_info['volume']

        # Determine volume to close
        close_volume = float(volume_lots) if volume_lots is not None else position_volume
        if close_volume <= 0:
             logger.error(f"Invalid close volume ({close_volume}) for position {position_id}.")
             return None
        if close_volume > position_volume:
             logger.warning(f"Requested close volume ({close_volume}) > position volume ({position_volume}). Closing full position.")
             close_volume = position_volume

        # Determine opposite order type for closing
        close_order_cmd = mt5.ORDER_TYPE_SELL if position_type == 0 else mt5.ORDER_TYPE_BUY

        # Get current price for closing
        current_tick = self.get_tick(symbol)
        if not current_tick:
             logger.error(f"Cannot get current tick for {symbol} to close position {position_id}.")
             return None

        # Price for closing market order
        close_price = current_tick['bid'] if close_order_cmd == mt5.ORDER_TYPE_SELL else current_tick['ask']

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_order_cmd,
            "position": position_id, # Specify the position ticket to close/reduce
            "price": close_price, # Market price for closing
            "deviation": int(self.default_slippage),
            "magic": self.default_magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK, # Use FOK or IOC
        }

        logger.info(f"Sending Close Order Request: {request}")
        return self._send_order_request(request)


    def get_positions(self, symbol=None):
        """Gets all open positions or positions for a specific symbol."""
        if not self._ensure_connection(): return []

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                # Check if the error is just "no positions found" (code 0) or a real error
                last_error = mt5.last_error()
                if last_error != 0:
                     logger.warning(f"Failed to get positions (symbol: {symbol}), error code = {last_error}")
                return [] # Return empty list on error or no positions

            # Convert tuples to dictionaries
            return [p._asdict() for p in positions]
        except Exception as e:
            logger.error(f"Exception getting positions (symbol: {symbol}): {e}", exc_info=True)
            return []

    def get_position_info(self, symbol):
        """Gets position info for a single symbol (assumes only one position per symbol)."""
        positions = self.get_positions(symbol=symbol)
        if positions:
            if len(positions) > 1:
                logger.warning(f"Found multiple positions for {symbol}. Returning the first one (Ticket: {positions[0]['ticket']}). Hedging or multiple strategies might be active.")
            return positions[0]
        return None

    def get_account_summary(self):
        """Gets account balance, equity, margin, etc."""
        if not self._ensure_connection(): return None
        try:
            info = mt5.account_info()
            if info:
                return info._asdict()
            else:
                last_error = mt5.last_error()
                logger.warning(f"Failed to get account info, error code = {last_error}")
                return None
        except Exception as e:
            logger.error(f"Exception getting account info: {e}", exc_info=True)
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
# if __name__ == '__main__':
#     # Assuming logger and MT5 initialized externally
#     # ... (example usage code) ...
#     pass
