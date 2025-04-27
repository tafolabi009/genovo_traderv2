# core/portfolio_compound.py

import numpy as np
import math

class PortfolioCapitalManager:
    """
    Manages total trading capital and allocates risk across multiple symbols
    based on portfolio-level constraints and signal confidence.
    """
    def __init__(self, initial_capital=20.0, config=None):
        """
        Initializes the Portfolio Capital Manager.

        Args:
            initial_capital (float): Starting total capital for trading.
            config (dict): Configuration for portfolio management.
                           Expected keys from 'portfolio_capital_config':
                           'max_total_risk_pct', 'max_allocation_per_trade_pct',
                           'allocation_method', 'kelly_fraction',
                           'min_position_notional'.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config or self._get_default_config()

        # Extract config parameters
        self.max_total_risk_pct = self.config.get('max_total_risk_pct', 0.05) # Max 5% total risk
        self.max_allocation_per_trade_pct = self.config.get('max_allocation_per_trade_pct', 0.02) # Max 2% risk per trade
        self.allocation_method = self.config.get('allocation_method', 'confidence_weighted')
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)
        self.min_position_notional = self.config.get('min_position_notional', 100)
        # self.target_win_loss_ratio = self.config.get('target_win_loss_ratio', 1.5) # May be needed later

        # State tracking (needs to be updated by the live/sim loop)
        self.open_positions_risk = {} # symbol -> risked_capital
        self.current_total_risk_pct = 0.0

        print(f"PortfolioCapitalManager initialized: Initial Capital=${initial_capital:.2f}, Max Total Risk={self.max_total_risk_pct*100:.1f}%")

    def _get_default_config(self):
        """Provides default configuration parameters."""
        return {
            'max_total_risk_pct': 0.05,
            'max_allocation_per_trade_pct': 0.02,
            'allocation_method': 'confidence_weighted',
            'kelly_fraction': 0.5,
            'min_position_notional': 100,
        }

    def update_capital(self, pnl):
        """Updates the current total capital based on realized PnL."""
        self.current_capital += pnl
        if self.current_capital < 0:
            print("Warning: Total capital is negative. Resetting to 0.")
            self.current_capital = 0

    def update_open_risk(self, open_positions_info):
        """
        Recalculates the total capital currently at risk based on open positions.
        This should be called periodically or when positions change in the live/sim loop.

        Args:
            open_positions_info (dict): Dict mapping symbol -> position details
                                        (e.g., {'EURUSD': {'volume': 0.1, 'entry_price': 1.1, 'sl_price': 1.09}}).
                                        Needs info to calculate risk per position.
        """
        self.open_positions_risk = {}
        total_risked_capital = 0.0

        if not open_positions_info:
             self.current_total_risk_pct = 0.0
             return

        for symbol, pos_info in open_positions_info.items():
            volume = pos_info.get('volume_lots', 0) # Assuming lots
            entry_price = pos_info.get('entry_price', 0)
            sl_price = pos_info.get('sl_price', None) # Stop loss price level
            contract_size = pos_info.get('contract_size', 100000) # Need contract size
            position_type = pos_info.get('type', None) # 0: Buy, 1: Sell

            if volume <= 0 or entry_price <= 0 or sl_price is None or position_type is None:
                continue # Cannot calculate risk for this position

            # Calculate risk per unit (difference between entry and SL)
            risk_per_unit = abs(entry_price - sl_price)

            # Calculate total notional size
            notional_size = volume * contract_size

            # Calculate capital risked for this position
            # Risked Capital = Risk per Unit * Notional Size / Entry Price (approx)
            # More accurately: Risked Capital = Risk per Unit * Units = Risk per Unit * Lots * Contract Size
            risked_capital = risk_per_unit * volume * contract_size

            self.open_positions_risk[symbol] = risked_capital
            total_risked_capital += risked_capital

        if self.current_capital > 0:
            self.current_total_risk_pct = total_risked_capital / self.current_capital
        else:
            self.current_total_risk_pct = 0.0

        # print(f"Debug: Updated total risk: {self.current_total_risk_pct*100:.2f}% ({len(self.open_positions_risk)} open positions)")


    def get_available_risk_capital(self):
        """Calculates how much more capital can be risked based on limits."""
        max_allowed_total_risk_capital = self.current_capital * self.max_total_risk_pct
        current_risked_capital = self.current_capital * self.current_total_risk_pct
        available_risk_capital = max(0, max_allowed_total_risk_capital - current_risked_capital)
        return available_risk_capital

    def allocate_risk_capital(self, potential_trades):
        """
        Determines how much risk capital to allocate to each potential new trade
        based on the chosen allocation method and available risk capital.

        Args:
            potential_trades (list): A list of dictionaries, where each dict represents
                                     a potential trade signal. Expected keys:
                                     'symbol', 'signal_strength' (e.g., confidence,
                                     probability, model output score), 'stop_loss_pct'
                                     (from model, e.g., 0.01 for 1%).

        Returns:
            dict: A dictionary mapping symbol -> allocated_risk_capital for new trades.
        """
        allocations = {}
        available_total_risk_capital = self.get_available_risk_capital()

        if not potential_trades or available_total_risk_capital <= 0:
            return allocations

        # --- Calculate Max Risk Per Trade ---
        max_risk_per_trade = self.current_capital * self.max_allocation_per_trade_pct

        # --- Filter trades exceeding max risk per trade ---
        valid_trades = []
        for trade in potential_trades:
             # Estimate capital needed to risk max_allocation_per_trade_pct
             sl_pct = trade.get('stop_loss_pct', None)
             if sl_pct is None or sl_pct <= 0:
                  print(f"Warning: Invalid stop_loss_pct for {trade['symbol']}, cannot allocate risk.")
                  continue
             # Min capital needed to risk 1 unit with this SL = 1 / sl_pct
             # Capital risked = Notional Size * sl_pct
             # If we risk max_risk_per_trade, Max Notional = max_risk_per_trade / sl_pct
             # We just need the requested risk per trade here
             trade['requested_risk'] = max_risk_per_trade # Assume each trade initially requests the max allowed risk
             valid_trades.append(trade)

        if not valid_trades:
            return allocations

        # --- Apply Allocation Method ---
        total_requested_risk = sum(t['requested_risk'] for t in valid_trades)
        total_allocatable_risk = min(available_total_risk_capital, total_requested_risk)

        if total_allocatable_risk <= 0:
             return allocations

        if self.allocation_method == 'equal_risk':
            # Distribute available risk equally among valid signals
            risk_per_signal = total_allocatable_risk / len(valid_trades)
            for trade in valid_trades:
                 # Allocate the smaller of equal share or the max allowed per trade
                 allocations[trade['symbol']] = min(risk_per_signal, max_risk_per_trade)

        elif self.allocation_method == 'confidence_weighted':
            # Weight allocation by signal strength (ensure strength is positive)
            total_strength = sum(max(0, t.get('signal_strength', 0)) for t in valid_trades)
            if total_strength > 1e-6: # Avoid division by zero
                for trade in valid_trades:
                    strength = max(0, trade.get('signal_strength', 0))
                    weight = strength / total_strength
                    allocated_risk = total_allocatable_risk * weight
                    # Ensure it doesn't exceed max per trade
                    allocations[trade['symbol']] = min(allocated_risk, max_risk_per_trade)
            else:
                 # Fallback to equal risk if total strength is zero
                 risk_per_signal = total_allocatable_risk / len(valid_trades)
                 for trade in valid_trades:
                      allocations[trade['symbol']] = min(risk_per_signal, max_risk_per_trade)

        # TODO: Implement 'kelly_portfolio' (more complex, needs covariance estimates etc.)

        else: # Default to equal risk
            print(f"Warning: Unknown allocation method '{self.allocation_method}'. Defaulting to 'equal_risk'.")
            risk_per_signal = total_allocatable_risk / len(valid_trades)
            for trade in valid_trades:
                 allocations[trade['symbol']] = min(risk_per_signal, max_risk_per_trade)

        return allocations


    def calculate_position_size_notional(self, symbol, allocated_risk_capital, current_price, stop_loss_pct):
        """
        Calculates the actual position size in notional value based on the
        capital allocated for risk and the stop loss percentage.

        Args:
            symbol (str): The trading symbol.
            allocated_risk_capital (float): The amount of capital allocated to risk on this trade.
            current_price (float): The current price of the asset.
            stop_loss_pct (float): The stop loss as a fraction of the price (e.g., 0.01 for 1%).

        Returns:
            float: The calculated position size in notional value (e.g., dollars).
                   Returns 0 if inputs are invalid.
        """
        if allocated_risk_capital <= 0 or current_price <= 0 or stop_loss_pct <= 0:
            return 0.0

        # Position Size (Notional) = Allocated Risk Capital / Stop Loss Percentage
        notional_size = allocated_risk_capital / stop_loss_pct

        # Apply minimum position size constraint
        if notional_size < self.min_position_notional:
             # Check if we can afford the minimum size within the allocated risk
             min_notional_risk = self.min_position_notional * stop_loss_pct
             if min_notional_risk <= allocated_risk_capital:
                  # We can afford the minimum, use it
                  notional_size = self.min_position_notional
             else:
                  # Cannot afford minimum size with this risk allocation
                  # print(f"Debug: Cannot meet min notional ${self.min_position_notional} for {symbol} with allocated risk ${allocated_risk_capital:.2f} and SL {stop_loss_pct*100:.2f}%")
                  return 0.0 # Cannot trade

        # TODO: Add constraint based on max leverage / available margin if needed,
        # although risk allocation should implicitly handle much of this.
        # max_notional_leverage = self.current_capital * self.max_leverage
        # notional_size = min(notional_size, max_notional_leverage)

        return notional_size

    def get_current_capital(self):
        """Returns the current total trading capital."""
        return self.current_capital

    def get_total_risk_pct(self):
        """Returns the percentage of capital currently at risk."""
        return self.current_total_risk_pct


# --- Factory Function ---
def create_portfolio_capital_manager(config):
    """
    Factory function to create the Portfolio Capital Manager.

    Args:
        config (dict): Configuration dictionary containing initial_capital
                       and portfolio capital management settings.

    Returns:
        PortfolioCapitalManager: The configured manager instance.
    """
    capital_config = config.get('portfolio_capital_config', {}) # Use the new config section
    return PortfolioCapitalManager(
        initial_capital=capital_config.get('initial_capital', 20.0),
        config=capital_config
    )
