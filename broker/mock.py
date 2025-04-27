# broker/mock.py

import time
import random
import numpy as np
from collections import deque
import logging

# Get logger instance (assuming setup_logger is called in main)
logger = logging.getLogger("genovo_traderv2")

class MockBroker:
    """
    Simulates broker interactions for backtesting or paper trading.
    Handles order placement, fill simulation, latency, slippage, commission,
    and basic Stop Loss / Take Profit triggering.
    Returns fill details instead of directly managing capital.
    """
    def __init__(self, config=None):
        """
        Initializes the Mock Broker.

        Args:
            config (dict): Configuration for mock broker behavior.
                           Expected keys: 'latency_ms', 'fill_probability',
                           'base_slippage_pct', 'volume_slippage_factor',
                           'commission_per_notional'.
        """
        self.config = config or self._get_default_config()

        # Simulation parameters
        self.latency_ms = self.config.get('latency_ms', (5, 20)) # Range (min, max) ms
        self.fill_probability = self.config.get('fill_probability', 0.99) # Base probability of an order getting filled
        self.base_slippage_pct = self.config.get('base_slippage_pct', 0.0001) # 0.01% base slippage
        self.volume_slippage_factor = self.config.get('volume_slippage_factor', 5e-10) # Additional slippage per unit of notional size
        self.commission_per_notional = self.config.get('commission_per_notional', 0.00005) # 0.005% commission ($5 per $100k)

        # Order tracking
        self.pending_orders = {} # order_id -> order_details
        self.open_positions = {} # symbol -> position_details { 'notional_size', 'entry_price', 'type', 'sl', 'tp', 'id' }
        self.order_id_counter = 0
        self.position_id_counter = 0

        logger.info("MockBroker initialized.")

    def _get_default_config(self):
        """Provides default configuration parameters."""
        return {
            'latency_ms': (5, 20),
            'fill_probability': 0.99,
            'base_slippage_pct': 0.0001,
            'volume_slippage_factor': 5e-10,
            'commission_per_notional': 0.00005,
        }

    def _generate_order_id(self):
        """Generates a unique order ID."""
        self.order_id_counter += 1
        return f"mock_ord_{self.order_id_counter}"

    def _generate_position_id(self):
        """Generates a unique position ID."""
        self.position_id_counter += 1
        return f"mock_pos_{self.position_id_counter}"

    def _simulate_latency(self):
        """Simulates network and processing latency time."""
        min_lat, max_lat = self.latency_ms
        latency_sec = random.uniform(min_lat, max_lat) / 1000.0
        # In backtesting, we don't actually sleep, just return the delay
        return latency_sec

    def _simulate_slippage(self, order_size_notional, current_price, order_type):
        """
        Simulates slippage based on order size and base slippage.
        Returns the actual fill price.
        """
        # Base slippage (random component) - can be positive or negative
        slippage_pct = random.uniform(-self.base_slippage_pct, self.base_slippage_pct)

        # Volume-based slippage (larger orders move the price more)
        volume_slippage = order_size_notional * self.volume_slippage_factor

        # Slippage direction depends on order direction (buy orders slip up, sell orders slip down)
        # This is a simplification; real slippage depends on order book depth and order type (market vs limit)
        slippage_direction = 1 if order_type == 'buy' else -1
        slippage_pct += volume_slippage * slippage_direction # Additive volume slippage

        fill_price = current_price * (1 + slippage_pct)
        return fill_price

    def _calculate_commission(self, order_size_notional):
        """Calculates commission based on notional value."""
        # Commission typically charged on open and close, so apply half here?
        # Or apply full commission on each transaction (open/close). Let's assume full.
        return order_size_notional * self.commission_per_notional

    def place_order(self, symbol, order_type, quantity_notional, price=None, stop_loss=None, take_profit=None, current_time=None):
        """
        Places an order with the mock broker. For backtesting, orders are processed immediately
        in the `process_orders` step based on market data for that step.

        Args:
            symbol (str): The trading symbol (e.g., 'EURUSD').
            order_type (str): 'buy' or 'sell' (market orders for now).
            quantity_notional (float): The size of the order in notional value.
            price (float, optional): Target price (unused for market orders).
            stop_loss (float, optional): Stop loss price level.
            take_profit (float, optional): Take profit price level.
            current_time (datetime/timestamp, optional): Simulation time for timestamping.

        Returns:
            str or None: The order ID if accepted, None otherwise.
        """
        if quantity_notional <= 0:
            logger.warning("MockBroker: Order quantity must be positive.")
            return None

        order_id = self._generate_order_id()
        latency = self._simulate_latency() # Simulate time delay (not used actively in backtest)

        order_details = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type.lower(), # Ensure lowercase ('buy' or 'sell')
            'quantity': quantity_notional,
            'status': 'pending', # Pending simulation based on next market data
            'latency': latency,
            'timestamp': current_time or time.time(), # Simulation time or real time
            'sl': stop_loss,
            'tp': take_profit,
            'target_price': price # Store target price if provided
        }

        # Basic check: Does this order conflict with an existing position?
        # Simple model: No hedging allowed. If long, only sell orders accepted. If short, only buy orders accepted.
        current_pos = self.open_positions.get(symbol)
        if current_pos:
             if current_pos['type'] == 'buy' and order_details['type'] == 'buy':
                  logger.warning(f"MockBroker: Cannot place new BUY for {symbol}, already long. Ignoring order {order_id}.")
                  return None
             if current_pos['type'] == 'sell' and order_details['type'] == 'sell':
                  logger.warning(f"MockBroker: Cannot place new SELL for {symbol}, already short. Ignoring order {order_id}.")
                  return None
             # If order is opposite, check if quantity matches to close
             if (current_pos['type'] == 'buy' and order_details['type'] == 'sell') or \
                (current_pos['type'] == 'sell' and order_details['type'] == 'buy'):
                  if abs(quantity_notional - current_pos['notional_size']) > 1e-6: # Allow small tolerance
                       logger.warning(f"MockBroker: Order {order_id} quantity ({quantity_notional}) doesn't match open position ({current_pos['notional_size']}) for closing {symbol}. Assuming full close intent.")
                       # Adjust order quantity to match position size for closing
                       order_details['quantity'] = current_pos['notional_size']


        self.pending_orders[order_id] = order_details
        logger.debug(f"MockBroker: Order {order_id} ({order_details['type']} {order_details['quantity']:.2f} {symbol}) received, pending processing.")
        return order_id

    def process_orders(self, market_data_at_step):
        """
        Processes pending orders and checks SL/TP based on current market data.
        This should be called at each simulation step *before* the agent acts on the new state.

        Args:
            market_data_at_step (dict): Dictionary mapping symbols to their current
                                        OHLC data (e.g., {'EURUSD': {'open':_, 'high':_, 'low':_, 'close':_}}).

        Returns:
            list: A list of fill event dictionaries. Each dict contains details
                  of a filled order (open, close, sl, tp) including
                  'fill_price', 'commission', 'pnl', 'order_id', 'position_id', 'fill_type'.
        """
        fill_events = []
        processed_order_ids = set()

        # --- 1. Check SL/TP on existing open positions ---
        for symbol, position in list(self.open_positions.items()):
            if symbol not in market_data_at_step:
                logger.warning(f"No market data for {symbol} to check SL/TP for position {position['id']}.")
                continue

            bar_data = market_data_at_step[symbol]
            high_price = bar_data['high']
            low_price = bar_data['low']
            close_price = bar_data['close'] # Use close price if SL/TP not hit within bar

            exit_price = None
            fill_type = None # 'sl', 'tp'

            # Check SL/TP based on position type
            if position['type'] == 'buy':
                if position['sl'] and low_price <= position['sl']:
                    exit_price = position['sl'] # Assume SL filled at the level
                    fill_type = 'sl'
                    logger.info(f"MockBroker: Position {position['id']} ({symbol}) hit Stop Loss at {exit_price:.5f} (Low: {low_price:.5f})")
                elif position['tp'] and high_price >= position['tp']:
                    exit_price = position['tp'] # Assume TP filled at the level
                    fill_type = 'tp'
                    logger.info(f"MockBroker: Position {position['id']} ({symbol}) hit Take Profit at {exit_price:.5f} (High: {high_price:.5f})")
            elif position['type'] == 'sell':
                if position['sl'] and high_price >= position['sl']:
                    exit_price = position['sl']
                    fill_type = 'sl'
                    logger.info(f"MockBroker: Position {position['id']} ({symbol}) hit Stop Loss at {exit_price:.5f} (High: {high_price:.5f})")
                elif position['tp'] and low_price <= position['tp']:
                    exit_price = position['tp']
                    fill_type = 'tp'
                    logger.info(f"MockBroker: Position {position['id']} ({symbol}) hit Take Profit at {exit_price:.5f} (Low: {low_price:.5f})")

            # If SL/TP was triggered, create fill event and close position
            if exit_price is not None and fill_type:
                # Calculate PnL and commission for the closing trade
                pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['notional_size'] * (1 if position['type'] == 'buy' else -1)
                commission = self._calculate_commission(position['notional_size'])

                fill_event = {
                    'symbol': symbol,
                    'order_id': f"sl_tp_{position['id']}", # Generate pseudo order ID
                    'position_id': position['id'],
                    'fill_type': fill_type, # 'sl' or 'tp'
                    'action_type': 'sell' if position['type'] == 'buy' else 'buy', # Closing action
                    'quantity': position['notional_size'],
                    'fill_price': exit_price,
                    'commission': commission,
                    'pnl': pnl,
                    'timestamp': market_data_at_step[symbol].get('timestamp', time.time()) # Use bar timestamp if available
                }
                fill_events.append(fill_event)

                # Remove the closed position
                del self.open_positions[symbol]


        # --- 2. Process Pending Orders ---
        for order_id, order in list(self.pending_orders.items()):
            # Skip if order was related to a position already closed by SL/TP (e.g., a manual close order)
            # This needs careful handling if manual close orders can be placed while SL/TP exists.
            # Simple approach: Process remaining pending orders.

            symbol = order['symbol']
            if symbol not in market_data_at_step:
                logger.warning(f"No market data for {symbol} to process order {order_id}.")
                continue

            # Use close price of the current bar for market order simulation
            # More complex: use open or average price? Close is common for backtesting.
            market_price = market_data_at_step[symbol]['close']

            # Simulate fill probability
            if random.random() > self.fill_probability:
                logger.info(f"MockBroker: Order {order_id} ({order['type']} {order['quantity']} {symbol}) failed to fill (simulated probability).")
                order['status'] = 'rejected'
                # Optionally create a rejection event?
                processed_order_ids.add(order_id)
                continue

            # Simulate fill price with slippage
            fill_price = self._simulate_slippage(order['quantity'], market_price, order['type'])
            commission = self._calculate_commission(order['quantity'])
            pnl = 0.0 # PnL is realized only when closing

            # Determine if opening or closing
            current_pos = self.open_positions.get(symbol)
            is_opening = not current_pos
            is_closing = current_pos and (
                (current_pos['type'] == 'buy' and order['type'] == 'sell') or
                (current_pos['type'] == 'sell' and order['type'] == 'buy')
            )

            position_id = None
            fill_type = None

            if is_opening:
                position_id = self._generate_position_id()
                fill_type = 'open'
                self.open_positions[symbol] = {
                    'id': position_id,
                    'notional_size': order['quantity'],
                    'entry_price': fill_price,
                    'type': order['type'],
                    'sl': order['sl'],
                    'tp': order['tp'],
                    'open_order_id': order_id,
                    'open_timestamp': order['timestamp']
                }
                logger.debug(f"MockBroker: Opened position {position_id} for {symbol} via order {order_id}.")

            elif is_closing:
                # Ensure the closing order quantity matches the open position
                if abs(order['quantity'] - current_pos['notional_size']) > 1e-6:
                     logger.warning(f"Closing order {order_id} quantity ({order['quantity']}) differs from open position {current_pos['id']} ({current_pos['notional_size']}). Adjusting to close full position.")
                     order['quantity'] = current_pos['notional_size']
                     # Recalculate costs based on actual closed quantity
                     fill_price = self._simulate_slippage(order['quantity'], market_price, order['type'])
                     commission = self._calculate_commission(order['quantity'])


                position_id = current_pos['id']
                fill_type = 'close'
                # Calculate PnL based on entry price of the closed position
                entry_price = current_pos['entry_price']
                pnl = (fill_price - entry_price) / entry_price * order['quantity'] * (1 if current_pos['type'] == 'buy' else -1)

                logger.debug(f"MockBroker: Closing position {position_id} for {symbol} via order {order_id}. PnL: {pnl:.2f}")
                del self.open_positions[symbol] # Remove closed position

            else:
                 # Should not happen with the check in place_order, but handle defensively
                 logger.error(f"MockBroker: Order {order_id} logic error. Current Pos: {current_pos}, Order Type: {order['type']}. Ignoring.")
                 processed_order_ids.add(order_id)
                 continue


            # Create fill event dictionary
            fill_event = {
                'symbol': symbol,
                'order_id': order_id,
                'position_id': position_id,
                'fill_type': fill_type, # 'open' or 'close'
                'action_type': order['type'], # 'buy' or 'sell'
                'quantity': order['quantity'],
                'fill_price': fill_price,
                'commission': commission,
                'pnl': pnl, # PnL is non-zero only for closing fills
                'timestamp': market_data_at_step[symbol].get('timestamp', time.time()) # Use bar timestamp
            }
            fill_events.append(fill_event)
            processed_order_ids.add(order_id)


        # Clean up processed pending orders
        for order_id in processed_order_ids:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

        return fill_events

    def get_open_positions(self):
        """Returns a copy of the current open positions held by the mock broker."""
        # Return a list of position dictionaries for consistency with MT5 interface
        return list(self.open_positions.values())

    def get_position(self, symbol):
        """Returns the current open position details for a specific symbol."""
        return self.open_positions.get(symbol, None) # Return None if no position


# --- Factory Function ---
def create_mock_broker(config):
    """
    Factory function to create the Mock Broker.

    Args:
        config (dict): Configuration dictionary for the mock broker.

    Returns:
        MockBroker: The configured mock broker instance.
    """
    # Pass relevant sub-config if needed
    return MockBroker(
        config=config.get('mock_broker_config', config) # Allow nested or direct config
    )
