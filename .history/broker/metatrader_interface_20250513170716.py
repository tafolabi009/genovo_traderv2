# broker/metatrader_interface.py

import MetaTrader5 as mt5
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import time
import pandas as pd
from datetime import datetime, timedelta
import pytz
from types import SimpleNamespace # Added for fallback info object
import os

# Try to use MT5 timeout extension if available
try:
    from utils.mt5_timeout_fix import extend_ipc_timeout
    # Apply timeout extension
    extend_ipc_timeout()
except ImportError:
    # If the timeout fix isn't available, set environment variable directly
    try:
        os.environ['MT5IPC_TIMEOUT'] = '120000'  # 120 seconds
    except Exception as e:
        print(f"Warning: Could not set MT5 timeout: {e}")

# Assuming logger setup is in utils
try:
    from ..utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback basic logging if relative import fails (e.g., running script directly)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class MetatraderInterface:
    """
    Handles interaction with the MetaTrader 5 terminal.
    Uses Tenacity for retrying connection and order placement.
    """

    MAX_BARS = 5000  # Max bars MT5 usually allows in one request

    def __init__(self, params):
        self.params = params.get('broker', {}) # Get broker specific params
        self.login_params = {
            "login": self.params.get('account_id'),
            "password": self.params.get('password'),
            "server": self.params.get('server'),
            "path": self.params.get('mt5_path')
        }
        self.connected = False
        self.account_currency = None
        self.account_info = None

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5), retry=retry_if_exception_type(ConnectionError))
    def connect(self):
        """Establishes connection to MetaTrader 5 terminal."""
        logger.info(f"Attempting to connect to MetaTrader 5: Server='{self.login_params['server']}', Login='{self.login_params['login']}'")
        try:
            if not mt5.initialize(
                login=self.login_params["login"],
                password=self.login_params["password"],
                server=self.login_params["server"],
                path=self.login_params["path"] # path to terminal.exe
            ):
                error_code = mt5.last_error()
                logger.error(f"MT5 connection failed: Error {error_code}")
                mt5.shutdown()
                raise ConnectionError(f"MT5 Initialization failed with code {error_code}")

            terminal_info = mt5.terminal_info()
            if not terminal_info:
                logger.error("MT5 connection failed: Could not get terminal info.")
                mt5.shutdown()
                raise ConnectionError("MT5 connected but could not get terminal info.")

            if not terminal_info.connected:
                 logger.error("MT5 Initialization successful but not connected to trade server.")
                 mt5.shutdown()
                 raise ConnectionError("MT5 not connected to trade server.")


            logger.info(f"MT5 Connection successful: Terminal Version {mt5.version()}, Broker: {terminal_info.name}")
            self.connected = True
            self.account_info = self.get_account_info()
            if self.account_info:
                 self.account_currency = self.account_info.currency
                 logger.info(f"Account Info: Currency={self.account_currency}, Balance={self.account_info.balance:.2f}, Equity={self.account_info.equity:.2f}")
            else:
                 logger.error("Failed to retrieve account info after connection.")
                 self.disconnect()
                 raise ConnectionError("MT5 connected but failed to get account info.")

        except Exception as e:
            logger.exception(f"An unexpected error occurred during MT5 connection: {e}")
            self.connected = False
            mt5.shutdown() # Ensure shutdown on unexpected error
            raise ConnectionError(f"MT5 connection process failed: {e}")

    def disconnect(self):
        """Shuts down the connection to MetaTrader 5."""
        if self.connected:
            logger.info("Disconnecting from MetaTrader 5.")
            mt5.shutdown()
            self.connected = False
        else:
            logger.info("Already disconnected from MetaTrader 5.")

    def ensure_connection(self):
        """Checks connection and attempts to reconnect if necessary."""
        if not self.connected or mt5.terminal_info() is None or not mt5.terminal_info().connected:
            logger.warning("MT5 connection lost or invalid. Attempting to reconnect...")
            self.connected = False # Mark as disconnected before attempting reconnect
            self.connect() # This will retry based on its decorator

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def get_account_info(self):
        """Retrieves account information."""
        self.ensure_connection()
        account_info = mt5.account_info()
        if account_info is None:
            raise ConnectionError("Failed to get account info from MT5.")
        return account_info

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
    def _get_symbol_tick(self, symbol_name):
        """Helper function to get symbol tick with retry and symbol selection."""
        self.ensure_connection()
        tick = mt5.symbol_info_tick(symbol_name)

        if tick is None or tick.time == 0:
            # Maybe the symbol is not visible in MarketWatch? Attempt to add it.
            logger.debug(f"Tick for {symbol_name} is initially invalid, attempting select/refresh.")
            if mt5.symbol_select(symbol_name, True):
                logger.info(f"Symbol {symbol_name} selected in MarketWatch, retrying tick fetch.")
                time.sleep(0.5) # Give MT5 time to potentially fetch/update the quote
                tick = mt5.symbol_info_tick(symbol_name)
            else:
                 logger.warning(f"Failed to select symbol {symbol_name} in MarketWatch. Tick data might remain unavailable.")

        # Check tick validity again after potential select/refresh
        if tick is None:
             raise ValueError(f"Could not get tick for symbol {symbol_name} after attempting select.")
        if tick.time == 0 : # Tick exists but has no valid timestamp/price yet
             raise ValueError(f"Tick retrieved for {symbol_name} but has zero timestamp (invalid quote).")

        # Optional: Add check for stale tick based on current time if needed
        # current_time_utc = datetime.now(pytz.utc)
        # tick_time = datetime.fromtimestamp(tick.time, tz=pytz.utc)
        # if current_time_utc - tick_time > timedelta(minutes=5): # Example: 5 minutes staleness threshold
        #     logger.warning(f"Tick for {symbol_name} is older than 5 minutes.")
        #     raise ValueError(f"Tick for {symbol_name} is stale (time: {tick_time}).")

        return tick


    def _get_conversion_rate(self, currency_from, currency_to):
        """
        Calculates the conversion rate between two currencies.
        Uses ask price for FROM->TO conversion, and 1/bid for TO->FROM conversion.
        Tries direct construction first, then consults params.yaml for overrides.
        """
        if currency_from == currency_to:
            return 1.0

        # 1. Try direct construction (e.g., "GBPUSD")
        pair1 = f"{currency_from}{currency_to}"
        try:
            tick1 = self._get_symbol_tick(pair1) # Uses retry logic internally
            rate = tick1.ask
            if rate == 0: # Check for invalid price
                 raise ValueError(f"Ask price for {pair1} is zero.")
            logger.debug(f"Conversion rate {currency_from}->{currency_to} using {pair1} ask: {rate}")
            return rate
        except Exception as e:
            logger.debug(f"Direct symbol {pair1} check failed: {e}")

        # 2. Try inverse construction (e.g., "USDGBP")
        pair2 = f"{currency_to}{currency_from}"
        try:
            tick2 = self._get_symbol_tick(pair2) # Uses retry logic internally
            if tick2.bid == 0: # Avoid division by zero
                 raise ValueError(f"Bid price for {pair2} is zero.")
            rate = 1.0 / tick2.bid
            logger.debug(f"Conversion rate {currency_from}->{currency_to} using 1/{pair2} bid: {rate}")
            return rate
        except Exception as e:
            logger.debug(f"Inverse symbol {pair2} check failed: {e}")

        # 3. Try configured symbols from params.yaml
        conversion_config = self.params.get('conversion_symbols', {}) # Get from broker params
        config_key = f"{currency_from}{currency_to}" # e.g., "GBPUSD"

        if config_key in conversion_config:
            configured_symbol = conversion_config[config_key]
            if not configured_symbol or not isinstance(configured_symbol, str):
                 logger.warning(f"Invalid configuration for conversion symbol key '{config_key}'. Skipping.")
            else:
                logger.debug(f"Attempting conversion using configured symbol '{configured_symbol}' for {config_key}")
                try:
                    tick_config = self._get_symbol_tick(configured_symbol) # Uses retry logic

                    # Determine if the configured symbol is direct (FROMTO) or inverse (TOFROM)
                    # Check against the actual symbol info currencies if possible for robustness
                    config_symbol_info = self.get_symbol_info(configured_symbol) # Use existing method
                    is_direct_match = False
                    is_inverse_match = False
                    if config_symbol_info:
                         is_direct_match = (config_symbol_info.currency_base == currency_from and
                                            config_symbol_info.currency_profit == currency_to)
                         is_inverse_match = (config_symbol_info.currency_base == currency_to and
                                             config_symbol_info.currency_profit == currency_from)
                    else:
                         # Fallback to simple name check if info failed (less reliable)
                         logger.warning(f"Could not get symbol info for configured symbol '{configured_symbol}'. Falling back to name check.")
                         is_direct_match = configured_symbol.upper().startswith(currency_from.upper())
                         is_inverse_match = configured_symbol.upper().startswith(currency_to.upper())


                    if is_direct_match:
                        rate = tick_config.ask
                        if rate == 0:
                            raise ValueError(f"Ask price for configured direct symbol {configured_symbol} is zero.")
                        logger.debug(f"Conversion rate {currency_from}->{currency_to} using configured direct symbol {configured_symbol} ask: {rate}")
                        return rate
                    elif is_inverse_match:
                        if tick_config.bid == 0:
                            raise ValueError(f"Bid price for configured inverse symbol {configured_symbol} is zero.")
                        rate = 1.0 / tick_config.bid
                        logger.debug(f"Conversion rate {currency_from}->{currency_to} using 1 / configured inverse symbol {configured_symbol} bid: {rate}")
                        return rate
                    else:
                        logger.error(f"Configured symbol '{configured_symbol}' found for {config_key}, but its currencies ({config_symbol_info.currency_base if config_symbol_info else '?'}/{config_symbol_info.currency_profit if config_symbol_info else '?'}) do not match required conversion. Cannot use.")

                except Exception as e:
                    logger.warning(f"Configured symbol '{configured_symbol}' for {config_key} failed: {e}")

        # 4. If all attempts fail
        logger.error(f"Cannot find valid direct, inverse, or configured conversion rate for {currency_from} to {currency_to}. Assuming rate=1.0. Lot size may be inaccurate.")
        return 1.0

    # Helper method to get volume precision based on step
    def _get_volume_precision(self, volume_step):
        """Calculates the number of decimal places needed for volume based on step."""
        if volume_step >= 1:
            return 0
        # Use decimal representation to avoid floating point inaccuracies
        import decimal
        step_str = str(decimal.Decimal(str(volume_step))) # Convert to string via Decimal
        if '.' in step_str:
            return len(step_str.split('.')[-1])
        return 0


    def _calculate_lots(self, symbol, risk_capital, stop_loss_pips, account_currency):
        """Calculates the lot size based on risk capital and stop loss in pips."""
        try:
            if account_currency is None:
                 logger.error("Account currency not set. Cannot calculate lots.")
                 return None

            symbol_info = self.get_symbol_info(symbol) # Uses retry and fallback logic
            if not symbol_info:
                logger.error(f"Cannot calculate lots. Symbol info not found or invalid for {symbol}.")
                return None

            # Get currencies
            base_currency = symbol_info.currency_base
            quote_currency = symbol_info.currency_profit # Profit is calculated in quote currency

            # Standard contract size, point value, digits
            contract_size = symbol_info.trade_contract_size
            point_value = symbol_info.point # The value of a point (smallest price change)
            digits = symbol_info.digits # Number of decimal places in price

            if contract_size is None or point_value is None or digits is None:
                 logger.error(f"Missing critical symbol info for {symbol} (contract_size, point, or digits). Cannot calculate lots.")
                 return None
            if contract_size <= 0 or point_value <= 0:
                 logger.error(f"Invalid symbol info for {symbol} (contract_size or point is zero/negative). Cannot calculate lots.")
                 return None

            # Calculate pip size (e.g., 0.0001 for 5-digit brokers, 0.01 for JPY pairs)
            # Heuristic: If digits >= 3 (e.g., JPY pairs), assume 2 decimal pip. Otherwise assume 4 decimal pip.
            # A more robust way might involve checking symbol name for JPY or using a config.
            pip_denominator = 100 if 'JPY' in symbol.upper() or digits <= 3 else 10000
            pip_size = 1 / pip_denominator # More direct way to get pip size (e.g., 0.01 or 0.0001)
            # Ensure pip_size aligns with point value for consistency check (optional)
            # expected_point_based_pip = point_value * (10 ** (digits - (2 if 'JPY' in symbol.upper() or digits <= 3 else 4)))
            # logger.debug(f"Pip size calculation for {symbol}: point={point_value}, digits={digits}, Calculated Pip Size={pip_size}")


            # Value of 1 pip for 1 standard lot, in the quote currency
            value_per_pip_per_lot = pip_size * contract_size

            if value_per_pip_per_lot <= 0:
                 logger.error(f"Calculated value per pip per lot is zero or negative for {symbol}. Cannot calculate lots.")
                 return None

            # Convert pip value to account currency if necessary
            pip_value_in_account_currency = 0
            if quote_currency == account_currency:
                pip_value_in_account_currency = value_per_pip_per_lot
            else:
                # We need to convert the pip value (in quote currency) to the account currency
                conversion_rate_quote_to_account = self._get_conversion_rate(quote_currency, account_currency)
                # Check if conversion failed (returned 1.0 as fallback but currencies differ)
                if conversion_rate_quote_to_account == 1.0 and quote_currency != account_currency:
                     logger.error(f"Lot calculation failed: Using fallback rate 1.0 for {quote_currency}->{account_currency} conversion resulted in potentially inaccurate pip value.")
                     # Decide if you want to stop or proceed with inaccurate lots
                     # return None # Option: Stop calculation if conversion fails
                     logger.warning(f"Proceeding with potentially inaccurate lot size due to failed conversion for {quote_currency}->{account_currency}.")
                     # If proceeding, use the possibly inaccurate value
                     pip_value_in_account_currency = value_per_pip_per_lot * conversion_rate_quote_to_account
                elif conversion_rate_quote_to_account <= 0:
                     logger.error(f"Lot calculation failed: Invalid conversion rate {conversion_rate_quote_to_account} for {quote_currency}->{account_currency}.")
                     return None
                else:
                     pip_value_in_account_currency = value_per_pip_per_lot * conversion_rate_quote_to_account


            if pip_value_in_account_currency <= 0:
                 logger.error(f"Calculated pip value in account currency is zero or negative ({pip_value_in_account_currency:.5f}) for {symbol}. Cannot calculate lots.")
                 return None

            # Calculate risk per lot in account currency
            if stop_loss_pips <= 0:
                logger.error(f"Stop loss pips ({stop_loss_pips}) must be positive. Cannot calculate lots for {symbol}.")
                return None
            risk_per_lot = stop_loss_pips * pip_value_in_account_currency

            if risk_per_lot <= 0:
                logger.error(f"Calculated risk per lot is zero or negative (${risk_per_lot:.2f}). Cannot calculate lots for {symbol}.")
                return None

            # Calculate desired lot size
            if risk_capital <= 0:
                logger.error(f"Risk capital (${risk_capital:.2f}) must be positive. Cannot calculate lots for {symbol}.")
                return None
            lots = risk_capital / risk_per_lot

            # Adjust to broker's volume step
            volume_step = symbol_info.volume_step
            min_volume = symbol_info.volume_min
            max_volume = symbol_info.volume_max

            if volume_step is None or min_volume is None or max_volume is None:
                 logger.error(f"Missing volume step/min/max info for {symbol}. Cannot finalize lots.")
                 return None
            if volume_step <= 0 or min_volume <= 0:
                 logger.error(f"Invalid volume step ({volume_step}) or min volume ({min_volume}) for {symbol}. Cannot finalize lots.")
                 return None


            # Use Decimal for precision with volume steps
            from decimal import Decimal, ROUND_HALF_UP
            lots_decimal = Decimal(str(lots))
            step_decimal = Decimal(str(volume_step))
            # Quantize to the volume step (round to nearest step)
            lots_adjusted = (lots_decimal / step_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * step_decimal
            # Convert back to float for MT5 order request
            lots_final = float(lots_adjusted)


            # Enforce min/max volume limits AFTER rounding to step
            lots_final = max(min_volume, min(lots_final, max_volume))

            # Final check: Ensure lot size is not zero or less than min volume AFTER clamping
            if lots_final < min_volume:
                 logger.warning(f"Calculated lot size {lots_final:.{self._get_volume_precision(volume_step)}f} for {symbol} is below minimum volume {min_volume} after adjustments. Cannot place trade.")
                 return None # Return None if lot size is effectively zero or too small


            logger.info(f"Calculated Lots for {symbol}: {lots_final:.{self._get_volume_precision(volume_step)}f} "
                        f"(RiskCap: ${risk_capital:.2f}, SL Pips: {stop_loss_pips}, "
                        f"PipValueInAcct: ${pip_value_in_account_currency:.5f}, RiskPerLot: ${risk_per_lot:.2f})")
            return lots_final

        except Exception as e:
            logger.exception(f"Error calculating lots for symbol {symbol}: {e}")
            return None


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def get_symbol_info(self, symbol):
        """Gets symbol information, attempts selection, and uses fallback from params."""
        self.ensure_connection()
        symbol_info = mt5.symbol_info(symbol)

        if symbol_info is None:
             logger.debug(f"mt5.symbol_info({symbol}) returned None. Attempting symbol selection.")
             # Attempt to select symbol if not found initially
             if mt5.symbol_select(symbol, True):
                 logger.info(f"Symbol {symbol} selected in MarketWatch, retrying info fetch.")
                 time.sleep(0.5) # Give MT5 time to potentially update
                 symbol_info = mt5.symbol_info(symbol)
             else:
                 logger.warning(f"Failed to select symbol {symbol} in MarketWatch. Info might remain unavailable.")


        if symbol_info is None:
            logger.warning(f"mt5.symbol_info({symbol}) still None after select attempt. Trying fallback config.")
            fallback_info_map = self.params.get('symbol_info_fallback', {})
            fallback_info = fallback_info_map.get(symbol) if isinstance(fallback_info_map, dict) else None

            if fallback_info and isinstance(fallback_info, dict):
                logger.warning(f"Using fallback info from params.yaml for {symbol}.")
                # Define default structure expected by calculation logic
                defaults = {
                    'name': symbol,
                    'currency_base': symbol[:3],
                    'currency_profit': symbol[3:], # Assuming standard naming XXXYYY
                    'currency_margin': symbol[3:], # Usually quote currency
                    'trade_contract_size': 100000.0,
                    'point': 0.00001,
                    'digits': 5,
                    'volume_step': 0.01,
                    'volume_min': 0.01,
                    'volume_max': 1000.0,
                    'trade_mode': mt5.SYMBOL_TRADE_MODE_FULL # Assume full trading allowed
                    # Add any other attributes used elsewhere if needed
                }
                # Merge defaults with fallback config from yaml
                final_fallback = {**defaults, **fallback_info}

                # Basic validation of fallback data
                required_keys = ['currency_base', 'currency_profit', 'trade_contract_size', 'point', 'digits', 'volume_step', 'volume_min', 'volume_max']
                if not all(key in final_fallback and final_fallback[key] is not None for key in required_keys):
                     logger.error(f"Fallback info for {symbol} is missing required keys or has None values. Cannot use.")
                     return None
                if final_fallback['trade_contract_size'] <= 0 or final_fallback['point'] <= 0 or final_fallback['volume_step'] <= 0 or final_fallback['volume_min'] <= 0:
                     logger.error(f"Fallback info for {symbol} contains invalid numeric values (zero or negative). Cannot use.")
                     return None

                # Return as a SimpleNamespace object to mimic mt5.symbol_info structure
                return SimpleNamespace(**final_fallback)
            else:
                 logger.error(f"Symbol {symbol} not found in MT5 and no valid fallback info provided in params.yaml.")
                 return None # Explicitly return None if truly not found

        return symbol_info


    def get_current_price(self, symbol):
        """Gets the current ask price for a symbol."""
        try:
            tick = self._get_symbol_tick(symbol) # Use retry helper
            return tick.ask
        except Exception as e:
            logger.error(f"Could not get current price for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, timeframe_str, start_dt_utc, end_dt_utc):
        """
        Fetches historical OHLC data for a given symbol and timeframe.
        Handles pagination if necessary.
        """
        self.ensure_connection()

        timeframe_map = {
            '1m': mt5.TIMEFRAME_M1, '5m': mt5.TIMEFRAME_M5, '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30, '1h': mt5.TIMEFRAME_H1, '4h': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1, '1W': mt5.TIMEFRAME_W1, '1MN': mt5.TIMEFRAME_MN1
        }
        timeframe = timeframe_map.get(timeframe_str)
        if timeframe is None:
            logger.error(f"Invalid timeframe string: {timeframe_str}")
            return None

        all_rates_df = pd.DataFrame()
        current_start = start_dt_utc

        logger.info(f"Fetching historical data for {symbol} ({timeframe_str}) from {start_dt_utc} to {end_dt_utc}")

        while current_start < end_dt_utc:
            try:
                rates = mt5.copy_rates_range(symbol, timeframe, current_start, end_dt_utc)

                if rates is None or len(rates) == 0:
                    error_code = mt5.last_error()
                    # Check if error suggests end of available data or other issue
                    if error_code[0] != 0 and len(all_rates_df) > 0:
                         logger.warning(f"mt5.copy_rates_range returned no more data for {symbol} starting {current_start}. Error: {error_code}. Stopping fetch.")
                         break # Assume end of data for this period
                    elif error_code[0] != 0:
                         logger.error(f"Failed to fetch historical data for {symbol} from {current_start}. Error: {error_code}")
                         return None # Return None on critical error
                    else:
                         logger.info(f"No historical data found for {symbol} in range starting {current_start}. Ending fetch for this range.")
                         break # No data in this specific range

                # Convert to DataFrame
                rates_df = pd.DataFrame(rates)
                rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s', utc=True)

                # Append to the main DataFrame, dropping duplicates if any overlap
                all_rates_df = pd.concat([all_rates_df, rates_df]).drop_duplicates(subset='time', keep='first')

                # Check if we received fewer bars than theoretically possible (indicates end of data)
                # Or if the last timestamp received is already beyond our target end time
                last_time_fetched = rates_df['time'].iloc[-1]
                logger.debug(f"Fetched {len(rates_df)} bars for {symbol}. Last timestamp: {last_time_fetched}")

                if last_time_fetched >= end_dt_utc:
                    logger.debug("Last fetched bar timestamp is at or after target end time. Fetch complete.")
                    break

                # Prepare for the next iteration if needed
                # Start next fetch from the time of the last bar + 1 second
                current_start = last_time_fetched + timedelta(seconds=1)

                # Safety break if stuck in loop somehow (e.g., continuously getting same old bar)
                if len(rates) < 2 and len(all_rates_df) > self.MAX_BARS:
                     logger.warning(f"Fetched very few bars ({len(rates)}) but already have {len(all_rates_df)}. Potential issue or end of data. Stopping fetch.")
                     break

            except Exception as e:
                logger.exception(f"Error fetching historical data chunk for {symbol}: {e}")
                return None # Return None on unexpected exception

        if all_rates_df.empty:
            logger.warning(f"No historical data retrieved for {symbol} in the specified range.")
            return None

        # Ensure data is sorted by time and index is reset
        all_rates_df = all_rates_df.sort_values(by='time').reset_index(drop=True)

        # Filter final DataFrame to exact start/end times (inclusive)
        all_rates_df = all_rates_df[(all_rates_df['time'] >= start_dt_utc) & (all_rates_df['time'] <= end_dt_utc)]

        logger.info(f"Successfully fetched {len(all_rates_df)} bars for {symbol} ({timeframe_str})")
        return all_rates_df


    def _format_order_result(self, result):
        """Formats the MT5 order result object into a dictionary."""
        if result is None:
            return {"status": "Error", "message": "Order send resulted in None (likely exception)."}

        # Check if the attribute exists before calling it
        retcode_desc = f"Code: {result.retcode}" # Default description
        if hasattr(mt5, 'trade_retcode_description'):
             try:
                 retcode_desc = mt5.trade_retcode_description(result.retcode)
             except Exception as e:
                  logger.error(f"Error calling mt5.trade_retcode_description({result.retcode}): {e}")
                  # Keep the default description

        formatted = {
            "retcode": result.retcode,
            "retcode_desc": retcode_desc,
            "comment": result.comment,
            "order_ticket": result.order,
            "deal_ticket": result.deal,
            "volume": result.volume,
            "price": result.price,
            "bid": result.bid,
            "ask": result.ask,
            "request_id": result.request_id,
        }
        if result.request:
            formatted["request"] = {
                "action": result.request.action,
                "symbol": result.request.symbol,
                "volume": result.request.volume,
                "type": result.request.type,
                "price": result.request.price,
                "sl": result.request.sl,
                "tp": result.request.tp,
                "magic": result.request.magic,
                "comment": result.request.comment,
                "type_filling": result.request.type_filling,
                "type_time": result.request.type_time,
            }

        # Determine status based on retcode
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            formatted["status"] = "Success"
            formatted["message"] = f"Order executed successfully (Order: {result.order}, Deal: {result.deal})"
        elif result.retcode == mt5.TRADE_RETCODE_PLACED:
            formatted["status"] = "Placed"
            formatted["message"] = f"Order placed successfully (Order: {result.order})"
        else:
            formatted["status"] = "Error"
            formatted["message"] = f"Order failed: {result.comment} ({retcode_desc})"

        return formatted


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(Exception))
    def _send_order_request(self, request):
        """
        Sends the order request dictionary to MT5.
        Includes retry logic and checks for non-retryable errors.
        """
        self.ensure_connection()
        logger.debug(f"Sending order request: {request}")

        result = None # Initialize result
        retries = self.params.get('order_retries', 3)
        retry_delay = self.params.get('retry_delay', 1.5) # Use a slightly longer default delay

        for attempt in range(retries):
            try:
                result = mt5.order_send(request)

                if result is None:
                     # This shouldn't happen if MT5 is working, indicates a deeper issue
                     logger.error(f"mt5.order_send() returned None (Attempt {attempt+1}). Check MT5 connection/logs.")
                     # Treat as failure, maybe raise exception or wait before retry
                     time.sleep(retry_delay)
                     continue # Go to next attempt

                # Check for immediate success or placement
                if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
                    logger.info(f"Order successful (Attempt {attempt+1}): Code={result.retcode}, Order={result.order}, Deal={result.deal}")
                    return result # Return successful result

                # Handle failure codes
                # Check if the description function exists
                retcode_desc = f"Code: {result.retcode}" # Default
                if hasattr(mt5, 'trade_retcode_description'):
                    try:
                        retcode_desc = mt5.trade_retcode_description(result.retcode)
                    except Exception as e:
                         logger.error(f"Error calling mt5.trade_retcode_description({result.retcode}): {e}")

                # Check for non-retryable errors
                non_retryable_codes = {
                    10004, # Requote
                    10008, # Request timed out (maybe retry once?)
                    10013, # Invalid request (logic error in parameters)
                    10014, # Invalid volume
                    10015, # Invalid price
                    10016, # Invalid stops (SL/TP)
                    10017, # Trade is disabled
                    10018, # Market is closed
                    10019, # Not enough money
                    10020, # Price changed
                    10021, # Off quotes (no prices for symbol)
                    10022, # Broker disabled trade
                    10024, # Order quantity limit reached
                    10025, # Position quantity limit reached (on account)
                    10026, # Account disabled
                    10027, # AutoTrading disabled by client (terminal setting)
                    10030, # Invalid filling type
                    10031, # No connection to trade server
                    10038, # Position volume limit reached (for symbol)
                    # Add more codes from MT5 documentation if needed
                }

                if result.retcode in non_retryable_codes:
                     logger.error(f"Order send failed permanently (Attempt {attempt+1}): RetCode={result.retcode} ({retcode_desc}), Comment='{result.comment}'. Not retrying.")
                     return result # Return the failed result immediately

                # Log warning for retryable errors (e.g., temporary network issues, server busy)
                logger.warning(f"Order send failed (Attempt {attempt+1}): RetCode={result.retcode} ({retcode_desc}), Comment='{result.comment}'. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                # Optional: Slightly increase delay for subsequent retries
                # retry_delay *= 1.5

            except Exception as e:
                # Catch exceptions during the order send itself (e.g., connection lost mid-request)
                logger.exception(f"Exception during mt5.order_send() (Attempt {attempt+1}): {e}")
                if attempt == retries - 1:
                    logger.error(f"Order placement failed permanently after {retries} attempts due to exception.")
                    # Create a dummy result to indicate failure if needed by calling code
                    return SimpleNamespace(retcode=mt5.TRADE_RETCODE_CONNECTION, comment=f"Exception: {e}", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=request)
                time.sleep(retry_delay)

        # If loop finishes without success or non-retryable error
        logger.error(f"Order placement failed permanently after {retries} attempts. Last RetCode: {result.retcode if result else 'N/A'}")
        return result # Return the last failed result (or None if initial send failed badly)


    def place_market_order(self, symbol, order_type, volume, stop_loss=None, take_profit=None, magic_number=23456, comment="genovo_v2", deviation=3):
        """Places a market order (buy or sell)."""
        self.ensure_connection()

        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Cannot place order. Symbol info not found for {symbol}.")
            return self._format_order_result(None) # Return formatted error

        # Determine order type constant
        if order_type.lower() == 'buy':
            action = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask # Use current ask for buy
        elif order_type.lower() == 'sell':
            action = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid # Use current bid for sell
        else:
            logger.error(f"Invalid order type: {order_type}")
            return self._format_order_result(None) # Return formatted error

        if price is None or price == 0:
             logger.error(f"Could not get valid market price ({'ask' if action == mt5.ORDER_TYPE_BUY else 'bid'}) for {symbol}. Cannot place order.")
             # Create a dummy result indicating price error
             dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_PRICE, comment="Invalid market price", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
             return self._format_order_result(dummy_result)

        # Normalize SL/TP levels
        point = symbol_info.point
        digits = symbol_info.digits

        sl_level = 0.0
        if stop_loss is not None:
            sl_level = round(stop_loss, digits)

        tp_level = 0.0
        if take_profit is not None:
            tp_level = round(take_profit, digits)

        # Basic SL/TP validation against current price
        if action == mt5.ORDER_TYPE_BUY:
            if sl_level != 0.0 and sl_level >= price:
                logger.error(f"Invalid Stop Loss for BUY order: SL {sl_level} >= Price {price}")
                dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_STOPS, comment="Invalid SL for BUY", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
                return self._format_order_result(dummy_result)
            if tp_level != 0.0 and tp_level <= price:
                logger.error(f"Invalid Take Profit for BUY order: TP {tp_level} <= Price {price}")
                dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_STOPS, comment="Invalid TP for BUY", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
                return self._format_order_result(dummy_result)
        elif action == mt5.ORDER_TYPE_SELL:
             if sl_level != 0.0 and sl_level <= price:
                logger.error(f"Invalid Stop Loss for SELL order: SL {sl_level} <= Price {price}")
                dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_STOPS, comment="Invalid SL for SELL", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
                return self._format_order_result(dummy_result)
             if tp_level != 0.0 and tp_level >= price:
                logger.error(f"Invalid Take Profit for SELL order: TP {tp_level} >= Price {price}")
                dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_STOPS, comment="Invalid TP for SELL", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
                return self._format_order_result(dummy_result)


        # Define order request dictionary
        request = {
            "action": mt5.TRADE_ACTION_DEAL, # Use DEAL for market orders
            "symbol": symbol,
            "volume": float(volume), # Ensure volume is float
            "type": action,
            "price": price,
            "sl": sl_level,
            "tp": tp_level,
            "deviation": deviation,
            "magic": magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancelled (standard for market)
             # Use default filling type from symbol info or fallback
            "type_filling": symbol_info.filling_mode if hasattr(symbol_info, 'filling_mode') else mt5.ORDER_FILLING_FOK,
        }

        logger.info(f"Attempting Trade: {symbol} Action={action}, Lots={volume:.{self._get_volume_precision(symbol_info.volume_step)}f}, Price={price:.{digits}f}, SL={sl_level:.{digits}f}, TP={tp_level:.{digits}f}")

        # Send the request using the retry mechanism
        result = self._send_order_request(request)

        # Format and return the result
        formatted_result = self._format_order_result(result)
        if formatted_result["status"] != "Success" and formatted_result["status"] != "Placed":
             logger.error(f"Failed Order Details: {formatted_result}")
        return formatted_result


    def close_position(self, position_ticket, volume=None, deviation=3):
        """Closes a position by ticket ID."""
        self.ensure_connection()

        try:
            position = mt5.positions_get(ticket=position_ticket)
            if not position or len(position) == 0:
                logger.error(f"Position with ticket {position_ticket} not found.")
                return self._format_order_result(None) # Indicate error

            position = position[0] # Get the position object

            symbol = position.symbol
            position_volume = position.volume
            order_type = position.type # 0 for buy, 1 for sell
            magic = position.magic
            comment = f"Close Pos {position_ticket}"

            # Determine close action and price
            if order_type == mt5.ORDER_TYPE_BUY: # Closing a Buy position means Selling
                close_action = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            elif order_type == mt5.ORDER_TYPE_SELL: # Closing a Sell position means Buying
                close_action = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            else:
                logger.error(f"Unknown position type {order_type} for ticket {position_ticket}")
                return self._format_order_result(None)

            if price is None or price == 0:
                logger.error(f"Could not get valid closing price for {symbol}. Cannot close position {position_ticket}.")
                dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_PRICE, comment="Invalid closing price", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
                return self._format_order_result(dummy_result)

            close_volume = position_volume if volume is None else float(volume)
            if close_volume > position_volume or close_volume <= 0:
                 logger.error(f"Invalid volume {close_volume} specified for closing position {position_ticket} (Volume: {position_volume}).")
                 dummy_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_INVALID_VOLUME, comment="Invalid close volume", order=0, deal=0, volume=0.0, price=0.0, bid=0.0, ask=0.0, request_id=0, request=None)
                 return self._format_order_result(dummy_result)


            # Get symbol info for filling type etc.
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                 logger.error(f"Cannot close position {position_ticket}. Symbol info not found for {symbol}.")
                 return self._format_order_result(None)


            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position_ticket, # Specify the position ticket to close
                "symbol": symbol,
                "volume": close_volume,
                "type": close_action,
                "price": price,
                "deviation": deviation,
                "magic": magic, # Use original magic number
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": symbol_info.filling_mode if hasattr(symbol_info, 'filling_mode') else mt5.ORDER_FILLING_FOK,
            }

            logger.info(f"Attempting to Close Position: Ticket={position_ticket}, Symbol={symbol}, Volume={close_volume:.{self._get_volume_precision(symbol_info.volume_step)}f}, Type={close_action}, Price={price:.{symbol_info.digits}f}")

            result = self._send_order_request(request)
            formatted_result = self._format_order_result(result)
            if formatted_result["status"] != "Success":
                logger.error(f"Failed Close Position Details: {formatted_result}")
            return formatted_result

        except Exception as e:
            logger.exception(f"Exception occurred while trying to close position {position_ticket}: {e}")
            return self._format_order_result(None) # Indicate error


    def get_open_positions(self, symbol=None, magic_number=None):
        """Retrieves open positions, optionally filtered by symbol and/or magic number."""
        self.ensure_connection()
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                logger.error(f"Failed to get positions. Error: {mt5.last_error()}")
                return [] # Return empty list on failure

            # Convert tuple of Position objects to list of dictionaries for easier handling
            position_list = []
            for pos in positions:
                pos_dict = {
                    "ticket": pos.ticket,
                    "time": datetime.fromtimestamp(pos.time, tz=pytz.utc),
                    "type": pos.type, # 0: Buy, 1: Sell
                    "magic": pos.magic,
                    "identifier": pos.identifier,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "symbol": pos.symbol,
                    "comment": pos.comment,
                    "swap": pos.swap,
                }
                position_list.append(pos_dict)

            # Filter by magic number if specified
            if magic_number is not None:
                position_list = [p for p in position_list if p["magic"] == magic_number]

            return position_list

        except Exception as e:
            logger.exception(f"Exception occurred while fetching open positions: {e}")
            return []

    def get_account_summary(self):
         """Provides a summary dictionary of the account status."""
         try:
             info = self.get_account_info() # Uses retry
             if not info:
                 return {"error": "Failed to retrieve account info."}

             summary = {
                 "login": info.login,
                 "currency": info.currency,
                 "balance": info.balance,
                 "equity": info.equity,
                 "profit": info.profit,
                 "margin": info.margin,
                 "margin_free": info.margin_free,
                 "margin_level": info.margin_level,
                 "server": info.server,
                 "trade_mode": info.trade_mode # 0=Disabled, 1=LongOnly, 2=ShortOnly, 3=Long&Short, 4=CloseOnly
             }
             return summary
         except Exception as e:
             logger.exception(f"Error getting account summary: {e}")
             return {"error": f"Exception getting account summary: {e}"}

    def get_recent_bars(self, symbol, timeframe, num_bars):
        """
        Retrieves the most recent bars for the specified symbol and timeframe.
        
        Args:
            symbol (str): Symbol name (e.g., 'EURUSD', 'EURUSDm')
            timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1)
            num_bars (int): Number of bars to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with columns: time, open, high, low, close, tick_volume
                         Returns None on error
        """
        self.ensure_connection()
        
        try:
            # Try multiple methods to make this more robust
            for attempt in range(3):
                try:
                    # Make sure the symbol is selected in Market Watch
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Failed to select symbol {symbol} in Market Watch")
                    
                    # Try different retrieval methods based on attempt number
                    if attempt == 0:
                        # First try copy_rates_from_pos (standard method)
                        bars = self._get_bars_from_pos(symbol, timeframe, num_bars)
                    elif attempt == 1:
                        # Try a time-based approach with recent timeframe
                        end_time = datetime.now(pytz.utc)
                        start_time = end_time - self._calculate_time_period(timeframe, num_bars)
                        bars = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
                    else:
                        # Last resort: try with a longer lookback
                        end_time = datetime.now(pytz.utc)
                        # Double the lookback period
                        start_time = end_time - self._calculate_time_period(timeframe, num_bars * 2)
                        bars = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
                        
                    if bars is not None and len(bars) > 0:
                        df = pd.DataFrame(bars)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        
                        # Make sure we got enough bars or all available
                        if len(df) >= num_bars or attempt == 2:
                            if len(df) > num_bars:
                                df = df.tail(num_bars)
                            logger.debug(f"Successfully retrieved {len(df)} bars for {symbol} (attempt {attempt+1})")
                            return df.reset_index(drop=True)
                            
                        logger.debug(f"Got only {len(df)} bars on attempt {attempt+1}, trying another method...")
                    else:
                        logger.warning(f"No bars returned for {symbol} using method {attempt+1}")
                    
                    time.sleep(1)  # Brief pause before next attempt
                        
                except Exception as e:
                    logger.warning(f"Error on attempt {attempt+1} retrieving bars for {symbol}: {e}")
                    time.sleep(1)
            
            logger.error(f"Failed to retrieve sufficient bars for {symbol} after all attempts")
            return None
                
        except Exception as e:
            logger.error(f"Error retrieving recent bars for {symbol}: {e}", exc_info=True)
            return None
    
    def _get_bars_from_pos(self, symbol, timeframe, num_bars):
        """Helper method to get bars using copy_rates_from_pos in chunks if needed"""
        try:
            # MT5 limits the number of bars per request, handle in chunks if needed
            max_request_size = min(num_bars, self.MAX_BARS)
            
            # Request the bars
            bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, max_request_size)
            
            if bars is None or len(bars) == 0:
                return None
                
            # If we need more bars than MAX_BARS, fetch them in chunks
            if num_bars > self.MAX_BARS and len(bars) == self.MAX_BARS:
                all_bars = list(bars)
                remaining_bars = num_bars - self.MAX_BARS
                current_pos = self.MAX_BARS
                
                while remaining_bars > 0:
                    chunk_size = min(remaining_bars, self.MAX_BARS)
                    additional_bars = mt5.copy_rates_from_pos(symbol, timeframe, current_pos, chunk_size)
                    
                    if additional_bars is None or len(additional_bars) == 0:
                        # No more bars available
                        break
                        
                    all_bars.extend(additional_bars)
                    remaining_bars -= len(additional_bars)
                    current_pos += len(additional_bars)
                    
                    if len(additional_bars) < chunk_size:
                        # Fewer bars returned than requested, no more available
                        break
                
                return all_bars
            
            return bars
            
        except Exception as e:
            logger.warning(f"Error in _get_bars_from_pos for {symbol}: {e}")
            return None
    
    def _calculate_time_period(self, timeframe, num_bars):
        """Calculate time period based on timeframe and number of bars"""
        # Approximate time calculation based on timeframe
        if timeframe == mt5.TIMEFRAME_M1:
            return timedelta(minutes=num_bars)
        elif timeframe == mt5.TIMEFRAME_M5: 
            return timedelta(minutes=5*num_bars)
        elif timeframe == mt5.TIMEFRAME_M15:
            return timedelta(minutes=15*num_bars)
        elif timeframe == mt5.TIMEFRAME_M30:
            return timedelta(minutes=30*num_bars)
        elif timeframe == mt5.TIMEFRAME_H1:
            return timedelta(hours=num_bars)
        elif timeframe == mt5.TIMEFRAME_H4:
            return timedelta(hours=4*num_bars)
        elif timeframe == mt5.TIMEFRAME_D1:
            return timedelta(days=num_bars)
        elif timeframe == mt5.TIMEFRAME_W1:
            return timedelta(weeks=num_bars)
        elif timeframe == mt5.TIMEFRAME_MN1:
            return timedelta(days=30*num_bars)  # Approximate
        else:
            # Default fallback - assume minutes
            return timedelta(minutes=num_bars)


# Function to create and configure a MetatraderInterface instance
def create_mt5_interface(config):
    """
    Creates and returns a configured MetatraderInterface instance.
    
    Args:
        config (dict): Configuration dictionary containing broker settings
        
    Returns:
        MetatraderInterface: Configured interface instance
    """
    mt_interface = MetatraderInterface(config)
    try:
        mt_interface.connect()
    except Exception as e:
        logger.error(f"Failed to connect to MetaTrader: {e}")
    return mt_interface


# Example usage (optional, for testing)
if __name__ == '__main__':
    import yaml
    from pathlib import Path

    # Assuming params.yaml is in a 'configs' directory relative to this script
    config_path = Path(__file__).parent.parent / 'configs' / 'params.yaml'

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        exit()

    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        exit()

    # Ensure broker section exists
    if 'broker' not in params:
         print("Error: 'broker' section not found in params.yaml")
         exit()


    # --- Basic Tests ---
    print("--- Initializing Broker Interface ---")
    broker = MetatraderInterface(params)

    try:
        print("\n--- Connecting ---")
        broker.connect() # Will attempt connection with retries

        if broker.connected:
            print("\n--- Getting Account Summary ---")
            summary = broker.get_account_summary()
            print(summary)

            print("\n--- Getting Open Positions (All) ---")
            positions = broker.get_open_positions()
            if positions:
                print(f"Found {len(positions)} open positions.")
                # print(positions[0]) # Print details of first position if exists
            else:
                print("No open positions found.")

            # --- Test Calculation (Example: Requires a symbol in your params) ---
            test_symbol = params.get('strategy',{}).get('symbols', ['EURUSDm'])[0] # Get first symbol from strategy params or default
            print(f"\n--- Testing Lot Calculation for {test_symbol} ---")
            if broker.account_currency:
                 # Use realistic values - adjust risk/sl as needed
                 lots = broker._calculate_lots(
                     symbol=test_symbol,
                     risk_capital=100.0, # Example: Risk $100
                     stop_loss_pips=20.0, # Example: 20 pips SL
                     account_currency=broker.account_currency
                 )
                 if lots is not None:
                     print(f"Calculated lots for {test_symbol}: {lots}")
                 else:
                     print(f"Lot calculation failed for {test_symbol}.")
            else:
                 print("Cannot test lot calculation, account currency not found.")


            # --- Test Rate Conversion (Example: GBP -> USD) ---
            print("\n--- Testing Currency Conversion (GBP -> USD) ---")
            # Ensure you have a relevant entry in params.yaml if direct fails
            # e.g., broker -> conversion_symbols -> "GBPUSD": "GBPUSD."
            rate = broker._get_conversion_rate("GBP", "USD")
            if rate != 1.0 or "GBP" == "USD":
                 print(f"GBP to USD conversion rate: {rate}")
            else:
                 print(f"GBP to USD conversion rate defaulted to: {rate} (Check logs/config)")

             # --- Test Historical Data Fetch (Example: EURUSDm, 1h) ---
            print(f"\n--- Testing Historical Data Fetch for {test_symbol} (1h) ---")
            end_time = datetime.now(pytz.utc)
            start_time = end_time - timedelta(days=5) # Fetch last 5 days
            hist_data = broker.get_historical_data(test_symbol, '1h', start_time, end_time)
            if hist_data is not None:
                 print(f"Fetched {len(hist_data)} bars. First bar:\n{hist_data.head(1)}")
                 print(f"Last bar:\n{hist_data.tail(1)}")
            else:
                 print("Failed to fetch historical data.")


            # --- !! CAUTION: PLACING/CLOSING ORDERS WILL AFFECT YOUR ACCOUNT !! ---
            # --- !! Uncomment ONLY if you understand the risk and have Algo Trading enabled !! ---
            # print(f"\n--- Testing Market Order Placement for {test_symbol} ---")
            # # Ensure Algo Trading is enabled in MT5 terminal first!
            # symbol_info_test = broker.get_symbol_info(test_symbol)
            # if symbol_info_test:
            #     test_volume = symbol_info_test.volume_min # Use minimum volume for test
            #     print(f"Attempting to place BUY order for {test_volume} lots of {test_symbol}")
            #     buy_result = broker.place_market_order(test_symbol, 'buy', test_volume, comment="test_buy")
            #     print("BUY Order Result:", buy_result)
            #
            #     # Wait a bit before closing if order was successful
            #     if buy_result and buy_result.get('status') in ["Success", "Placed"]:
            #         order_ticket_to_close = buy_result.get('order_ticket')
            #         # Need to find the position ticket from the order/deal ticket if possible
            #         # This part is tricky as deal ticket might not be position ticket directly
            #         # A safer way is to find the position by symbol/magic after placing
            #         time.sleep(5)
            #         print("\n--- Finding and Closing Test Position ---")
            #         test_positions = broker.get_open_positions(symbol=test_symbol, magic_number=23456) # Assuming default magic
            #         if test_positions:
            #             pos_to_close = test_positions[0] # Close the first found test position
            #             print(f"Attempting to close position ticket {pos_to_close['ticket']}")
            #             close_result = broker.close_position(pos_to_close['ticket'])
            #             print("Close Order Result:", close_result)
            #         else:
            #             print("Could not find test position to close.")
            # else:
            #     print(f"Could not get symbol info for {test_symbol}, skipping order test.")
            # --- END OF CAUTION BLOCK ---


        else:
            print("Connection failed after retries.")

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
    finally:
        print("\n--- Disconnecting ---")
        broker.disconnect()
