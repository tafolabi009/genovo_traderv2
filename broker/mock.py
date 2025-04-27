# broker/mock.py

import time
import random
import numpy as np
from collections import deque

class MockBroker:
    """
    Simulates broker interactions for backtesting or paper trading.
    Handles order placement, fill simulation, latency, and slippage.
    """
    def __init__(self, capital_manager, config=None):
        """
        Initializes the Mock Broker.

        Args:
            capital_manager (CapitalManager): Instance to track capital and positions.
                                              (Note: May need adjustment, as the broker
                                               often holds the definitive position state)
            config (dict): Configuration for mock broker behavior.
                           Expected keys: 'latency_ms', 'fill_probability',
                           'base_slippage_pct', 'volume_slippage_factor',
                           'commission_per_notional'.
        """
        self.capital_manager = capital_manager # Might need refactoring later
        self.config = config or self._get_default_config()

        # Simulation parameters
        self.latency_ms = self.config.get('latency_ms', (10, 50)) # Range (min, max) ms
        self.fill_probability = self.config.get('fill_probability', 0.98) # Base probability of an order getting filled
        self.base_slippage_pct = self.config.get('base_slippage_pct', 0.0001) # 0.01% base slippage
        self.volume_slippage_factor = self.config.get('volume_slippage_factor', 1e-9) # Additional slippage per unit of notional size
        self.commission_per_notional = self.config.get('commission_per_notional', 0.00005) # 0.005% commission ($5 per $100k)

        # Order tracking
        self.pending_orders = {} # Store orders waiting for fill simulation
        self.order_id_counter = 0
        self.positions = {} # Symbol -> Notional Size (e.g., {'EURUSD': 10000})
        self.entry_prices = {} # Symbol -> Average Entry Price

        print("MockBroker initialized.")

    def _get_default_config(self):
        """Provides default configuration parameters."""
        return {
            'latency_ms': (10, 50),
            'fill_probability': 0.98,
            'base_slippage_pct': 0.0001,
            'volume_slippage_factor': 1e-9,
            'commission_per_notional': 0.00005,
        }

    def _generate_order_id(self):
        """Generates a unique order ID."""
        self.order_id_counter += 1
        return f"mock_{self.order_id_counter}"

    def _simulate_latency(self):
        """Simulates network and processing latency."""
        min_lat, max_lat = self.latency_ms
        latency_sec = random.uniform(min_lat, max_lat) / 1000.0
        # In a real-time sim, you might actually sleep
        # time.sleep(latency_sec)
        # For backtesting, we just factor this into potential price changes later
        return latency_sec

    def _simulate_slippage(self, order_size_notional, current_price, order_type='market'):
        """
        Simulates slippage based on order size and base slippage.
        Returns the actual fill price.
        """
        # Base slippage (random component)
        slippage_pct = random.uniform(-self.base_slippage_pct, self.base_slippage_pct)

        # Volume-based slippage (larger orders move the price more)
        volume_slippage = order_size_notional * self.volume_slippage_factor
        # Assume slippage is worse for aggressive orders (market buys/sells)
        # Slippage direction depends on order direction (buy orders slip up, sell orders slip down)
        # This is a simplification; real slippage depends on order book depth
        slippage_direction = 1 if order_type == 'buy' else -1
        slippage_pct += volume_slippage * slippage_direction

        fill_price = current_price * (1 + slippage_pct)
        return fill_price

    def _calculate_commission(self, order_size_notional):
        """Calculates commission based on notional value."""
        return order_size_notional * self.commission_per_notional

    def place_order(self, symbol, order_type, quantity_notional, price=None, stop_loss=None, take_profit=None):
        """
        Places an order with the mock broker.

        Args:
            symbol (str): The trading symbol (e.g., 'EURUSD').
            order_type (str): 'market', 'limit', 'stop', 'buy', 'sell'.
                              (Simplified: using 'buy'/'sell' for market orders here).
            quantity_notional (float): The size of the order in notional value.
            price (float, optional): The limit or stop price for limit/stop orders.
            stop_loss (float, optional): Stop loss price level.
            take_profit (float, optional): Take profit price level.

        Returns:
            str or None: The order ID if accepted, None otherwise.
        """
        if quantity_notional <= 0:
            print("MockBroker Warning: Order quantity must be positive.")
            return None

        order_id = self._generate_order_id()
        latency = self._simulate_latency() # Simulate time delay

        order_details = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type, # e.g., 'buy' or 'sell' for market
            'quantity': quantity_notional,
            'status': 'pending', # Pending simulation
            'latency': latency,
            'timestamp': time.time(), # Or simulation time
            'sl': stop_loss,
            'tp': take_profit
        }

        # Basic check: Can capital manager support this (margin check placeholder)
        # required_margin = quantity_notional / self.capital_manager.max_leverage # Simplified
        # if required_margin > self.capital_manager.get_current_capital():
        #     print(f"MockBroker Warning: Order {order_id} rejected due to insufficient margin (simplified check).")
        #     return None

        self.pending_orders[order_id] = order_details
        # print(f"MockBroker: Order {order_id} ({order_type} {quantity_notional} {symbol}) received, pending execution.")
        return order_id

    def process_pending_orders(self, current_market_data):
        """
        Processes pending orders based on current market data.
        This should be called at each simulation step.

        Args:
            current_market_data (dict): Dictionary mapping symbols to their current
                                        price information (e.g., {'EURUSD': {'close': 1.1050}}).

        Returns:
            list: A list of filled order details dictionaries.
        """
        filled_orders_info = []
        orders_to_remove = []

        for order_id, order in list(self.pending_orders.items()):
            symbol = order['symbol']
            if symbol not in current_market_data:
                # print(f"MockBroker Warning: No market data for {symbol} to process order {order_id}.")
                continue

            current_price = current_market_data[symbol]['close'] # Use close price for simulation

            # Simulate fill probability
            if random.random() > self.fill_probability:
                print(f"MockBroker Info: Order {order_id} ({order['type']} {order['quantity']} {symbol}) failed to fill (simulated).")
                order['status'] = 'rejected'
                # Potentially add to filled_orders_info with rejected status
                orders_to_remove.append(order_id)
                continue

            # Simulate fill price with slippage
            fill_price = self._simulate_slippage(order['quantity'], current_price, order['type'])

            # Calculate commission
            commission = self._calculate_commission(order['quantity'])

            # Update position and capital (CRITICAL section - needs careful handling)
            current_position = self.positions.get(symbol, 0.0)
            current_entry_price = self.entry_prices.get(symbol, 0.0)
            order_qty_signed = order['quantity'] if order['type'] == 'buy' else -order['quantity']
            new_position = current_position + order_qty_signed

            realized_pnl = 0.0
            # Handle closing or reducing positions
            if np.sign(new_position) != np.sign(current_position) and current_position != 0: # Position flip or close
                # Calculate PnL on the closed portion
                closed_qty = abs(current_position)
                price_diff = fill_price - current_entry_price
                realized_pnl = (price_diff / current_entry_price) * closed_qty * np.sign(current_position)
                # print(f"Closed PNL: {realized_pnl:.2f} on {closed_qty} units")

                # Update entry price if position flipped (partial close not fully handled here)
                if new_position != 0:
                    self.entry_prices[symbol] = fill_price # New entry price for remaining position
                else:
                     if symbol in self.entry_prices: del self.entry_prices[symbol] # No position, no entry price
            elif new_position != 0 and current_position != 0: # Adding to existing position
                 # Update average entry price
                 new_total_value = (current_position * current_entry_price) + (order_qty_signed * fill_price)
                 self.entry_prices[symbol] = new_total_value / new_position
            elif new_position != 0 and current_position == 0: # Opening new position
                 self.entry_prices[symbol] = fill_price


            # Update broker's position state
            if abs(new_position) < 1e-9: # Position closed
                if symbol in self.positions: del self.positions[symbol]
                if symbol in self.entry_prices: del self.entry_prices[symbol]
            else:
                self.positions[symbol] = new_position


            # Update capital manager (deduct commission, add realized PnL)
            # This interaction needs refinement. The simulator might own the capital
            # manager and update it based on fills reported by the broker.
            self.capital_manager.update_capital(realized_pnl - commission)

            # Mark order as filled and store details
            order['status'] = 'filled'
            order['fill_price'] = fill_price
            order['commission'] = commission
            order['realized_pnl'] = realized_pnl # PnL realized by this specific fill
            filled_orders_info.append(order.copy()) # Store a copy
            orders_to_remove.append(order_id)

            # print(f"MockBroker: Order {order_id} filled. Price: {fill_price:.5f}, Comm: ${commission:.4f}, PnL: ${realized_pnl:.4f}")


        # Clean up processed orders
        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

        # TODO: Add logic to check SL/TP levels against current market data for open positions

        return filled_orders_info

    def get_open_positions(self):
        """Returns the current open positions held by the mock broker."""
        return self.positions.copy()

    def get_position(self, symbol):
        """Returns the current open position for a specific symbol."""
        return self.positions.get(symbol, 0.0)

    def get_entry_price(self, symbol):
        """Returns the average entry price for a specific symbol."""
        return self.entry_prices.get(symbol, 0.0)

# --- Factory Function ---
def create_mock_broker(capital_manager, config):
    """
    Factory function to create the Mock Broker.

    Args:
        capital_manager (CapitalManager): The capital manager instance.
        config (dict): Configuration dictionary for the mock broker.

    Returns:
        MockBroker: The configured mock broker instance.
    """
    return MockBroker(
        capital_manager=capital_manager,
        config=config.get('mock_broker_config', {}) # Nested config
    )
