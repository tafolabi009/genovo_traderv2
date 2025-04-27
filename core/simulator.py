# core/simulator.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm # Progress bar
import matplotlib.pyplot as plt
import torch # Import torch here

# Import necessary components from other modules
# (Assuming they are structured correctly in the 'core' directory)
from core.features import create_feature_pipeline # Using the factory function
from core.model import create_model # Using the factory function
from core.strategy import create_strategy # Using the factory function
from core.portfolio_compound import create_portfolio_capital_manager # Using the factory function

class TradingSimulator:
    """
    Simulates the trading environment for backtesting and RL agent training.
    Integrates feature extraction, model prediction, strategy execution,
    and capital management. Includes basic SL/TP hit detection within bars.
    """
    def __init__(self, data, config):
        """
        Initializes the Trading Simulator.

        Args:
            data (pd.DataFrame): DataFrame containing historical market data.
                                 Must include 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
                                 Timestamp should be the index or a column.
            config (dict): Configuration dictionary containing settings for all components
                           (features, model, strategy, capital, simulator).
        """
        self.config = config
        self.sim_config = config.get('simulator_config', self._get_default_sim_config())

        # --- Data Handling ---
        self.data = self._prepare_data(data)
        self.total_steps = len(self.data)
        self.seq_length = config.get('model_config', {}).get('seq_length', 100) # Get from model config

        if self.total_steps <= self.seq_length:
            raise ValueError(f"Data length ({self.total_steps}) must be greater than sequence length ({self.seq_length})")

        # --- Core Components ---
        print("Initializing simulator components...")
        self.feature_pipeline = create_feature_pipeline(config.get('feature_config', {}))
        # Fit feature pipeline scaler (optional, could be pre-fitted)
        # self.feature_pipeline.fit(self.data.iloc[:self.seq_length*2]) # Fit on initial chunk

        self.model = create_model(config.get('model_config', {}))
        # Load pre-trained model weights if specified
        model_cfg = config.get('model_config', {}) # Get model config again
        model_path = model_cfg.get('load_path') # Use the potentially updated path from main.py
        if model_path and os.path.exists(model_path):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device)
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights from {model_path}. Error: {e}")
        else:
             print(f"No valid model path found ({model_path}). Model using initial weights.")
        self.device = next(self.model.parameters()).device # Get device model is on

        self.agent = create_strategy(self.model, config.get('strategy_config', {}))
        # Use PortfolioCapitalManager
        self.capital_manager = create_portfolio_capital_manager(config.get('portfolio_capital_config', {}))
        print("Components initialized.")

        # --- Simulation State ---
        self.current_step = self.seq_length
        self.current_position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.current_notional_size = 0.0 # Size of the current position in notional value
        self.steps_in_trade = 0
        self.last_action = 0 # Hold
        self.stop_loss_level = None # Store SL price for open position
        self.take_profit_level = None # Store TP price for open position

        # --- Simulation Parameters ---
        self.commission_pct = self.sim_config.get('commission_pct', 0.00005) # Example: 0.005%
        self.slippage_pct = self.sim_config.get('slippage_pct', 0.0001)  # Example: 0.01%

        # --- Logging ---
        self.history = {
            'timestamp': [], 'capital': [], 'position': [], 'action': [],
            'reward': [], 'pnl': [], 'notional_size': [], 'entry_price': [],
            'close_price': [], 'stop_loss': [], 'take_profit': [], 'hit_sl_tp': []
        }

    def _get_default_sim_config(self):
        """Provides default simulator configuration."""
        return {
            'commission_pct': 0.00005,
            'slippage_pct': 0.0001,
            'update_agent_every': 128, # Steps between PPO updates
            'print_every': 1000, # Steps between status prints
            'plot_results': False
        }

    def _prepare_data(self, data):
        """Ensures data has correct format and timestamp index."""
        df = data.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
             # Attempt to convert index if it's not datetime
             try:
                  df.index = pd.to_datetime(df.index)
             except Exception as e:
                  raise ValueError(f"Data index is not DatetimeIndex and could not be converted: {e}")

        df = df.sort_index()
        # Ensure required columns exist (case-insensitive check)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.lower() # Standardize to lower case
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        # Handle potential NaNs (forward fill is common)
        df.ffill(inplace=True)
        df.bfill(inplace=True) # Fill any remaining NaNs at the beginning
        return df

    def _get_state(self):
        """Extracts the current state (feature matrix) for the agent."""
        if self.current_step < self.seq_length:
             raise IndexError("Not enough data points for sequence length.")

        # Get data window [current_step - seq_length : current_step]
        market_window = self.data.iloc[self.current_step - self.seq_length : self.current_step]

        # Fit feature pipeline if not already fitted (should ideally be done before run)
        if not self.feature_pipeline.feature_extractor.is_fitted:
             print("Warning: Feature extractor not fitted. Fitting on initial data chunk.")
             fit_end_idx = min(self.seq_length * 5, len(self.data))
             self.feature_pipeline.fit(self.data.iloc[:fit_end_idx])
             if not self.feature_pipeline.feature_extractor.is_fitted:
                  raise RuntimeError("Failed to fit feature pipeline.")

        # Extract features - transform should handle scaling and column matching
        features_df = self.feature_pipeline.transform(market_window)

        if features_df.empty or features_df.isnull().values.any():
             print(f"Warning: Feature transformation returned empty or NaN data at step {self.current_step}. Check feature extraction.")
             # Handle this case, e.g., return zeros or previous state?
             # Returning zeros might be problematic for the model.
             # For now, raise an error as it indicates a problem.
             raise ValueError(f"Invalid features generated at step {self.current_step}")

        state_array = features_df.values
        return state_array # Shape: [seq_length, num_features]


    def _calculate_pnl(self, exit_price):
        """Calculates PnL based on entry and exit price for the current position."""
        if self.current_position == 0 or self.entry_price == 0:
            return 0.0

        price_diff = exit_price - self.entry_price
        # PnL = (Price Diff / Entry Price) * Notional Size * Position Direction
        pnl = (price_diff / self.entry_price) * self.current_notional_size * self.current_position
        return pnl

    def _apply_costs(self, trade_value, is_opening_trade):
        """Applies commission and slippage costs."""
        # Commission applied on both open and close based on traded value
        commission = trade_value * self.commission_pct
        # Slippage applied primarily on entry (or significant modification)
        slippage_cost = trade_value * self.slippage_pct if is_opening_trade else 0
        return commission + slippage_cost

    def step(self):
        """Performs one simulation step."""
        if self.current_step >= self.total_steps -1: # Need one more step for reward calc
            return None, True # End of simulation

        # --- Get Current Bar Data ---
        current_bar = self.data.iloc[self.current_step]
        next_bar = self.data.iloc[self.current_step + 1] # Needed for reward calculation based on next open
        current_price_close = current_bar['close']
        current_price_high = current_bar['high']
        current_price_low = current_bar['low']
        next_price_open = next_bar['open'] # Price for calculating step PnL/reward

        # --- Check for SL/TP Hit Before Getting New Action ---
        pnl = 0.0
        cost = 0.0
        closed_trade = False
        hit_sl_tp = False # Flag specifically for SL/TP hit

        if self.current_position == 1: # Long Position Check
            if self.stop_loss_level and current_price_low <= self.stop_loss_level:
                # SL Hit - exit at SL price (or slightly worse due to slippage simulated here)
                exit_price = self.stop_loss_level * (1 - self.slippage_pct)
                pnl = self._calculate_pnl(exit_price)
                cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                self.capital_manager.update_capital(pnl - cost)
                closed_trade = True
                hit_sl_tp = True
                print(f"Step {self.current_step}: Long SL hit at {exit_price:.5f} (Level: {self.stop_loss_level:.5f})")
            elif self.take_profit_level and current_price_high >= self.take_profit_level:
                # TP Hit - exit at TP price (or slightly better due to slippage simulated here)
                exit_price = self.take_profit_level * (1 + self.slippage_pct)
                pnl = self._calculate_pnl(exit_price)
                cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                self.capital_manager.update_capital(pnl - cost)
                closed_trade = True
                hit_sl_tp = True
                print(f"Step {self.current_step}: Long TP hit at {exit_price:.5f} (Level: {self.take_profit_level:.5f})")

        elif self.current_position == -1: # Short Position Check
            if self.stop_loss_level and current_price_high >= self.stop_loss_level:
                # SL Hit - exit at SL price (or slightly worse)
                exit_price = self.stop_loss_level * (1 + self.slippage_pct)
                pnl = self._calculate_pnl(exit_price)
                cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                self.capital_manager.update_capital(pnl - cost)
                closed_trade = True
                hit_sl_tp = True
                print(f"Step {self.current_step}: Short SL hit at {exit_price:.5f} (Level: {self.stop_loss_level:.5f})")
            elif self.take_profit_level and current_price_low <= self.take_profit_level:
                # TP Hit - exit at TP price (or slightly better)
                exit_price = self.take_profit_level * (1 - self.slippage_pct)
                pnl = self._calculate_pnl(exit_price)
                cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                self.capital_manager.update_capital(pnl - cost)
                closed_trade = True
                hit_sl_tp = True
                print(f"Step {self.current_step}: Short TP hit at {exit_price:.5f} (Level: {self.take_profit_level:.5f})")

        # Reset position if SL/TP was hit
        if closed_trade:
            self.current_position = 0
            self.entry_price = 0.0
            self.current_notional_size = 0.0
            self.steps_in_trade = 0
            self.stop_loss_level = None
            self.take_profit_level = None


        # --- If no SL/TP hit, get Agent Action ---
        action = 0 # Default to Hold if already closed or error occurs
        log_prob, value, model_outputs = None, None, {}
        if not closed_trade: # Only get action if still in market or flat
            # 1. Get current state (features)
            try:
                 state = self._get_state()
                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim
            except ValueError as e:
                 print(f"Error getting state at step {self.current_step}: {e}. Skipping step.")
                 # How to handle? Skip storing transition? Assume Hold?
                 # For RL, might need to handle this state error more gracefully.
                 # For now, log history with NaNs and continue.
                 self._log_history(current_bar.name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, hit_sl_tp)
                 self.current_step += 1
                 return None, (self.current_step >= self.total_steps -1) or (self.capital_manager.get_current_capital() <= 0)


            # 2. Get action from agent
            try:
                action, log_prob, value, _, _, model_outputs = self.agent.select_action(state_tensor)
                # Action: 0=Hold, 1=Buy, 2=Sell (adjust if agent uses different mapping)
            except Exception as e:
                 print(f"Error getting action from agent at step {self.current_step}: {e}. Assuming Hold.")
                 action = 0 # Default to hold on error


            # 3. Determine Position Size based on model output and capital manager
            # Use model's suggestion for SL/TP percentages
            suggested_sl_pct = model_outputs.get('stop_loss', torch.tensor(0.01)).item() # Default 1%
            suggested_tp_pct = model_outputs.get('take_profit', torch.tensor(0.02)).item() # Default 2%
            signal_strength = torch.softmax(model_outputs.get('policy_logits', torch.zeros(1,3)), dim=-1).max().item()

            # Create potential trade info for capital manager
            potential_trade_info = {
                 'symbol': self.config.get('symbol', 'UNKNOWN'), # Get symbol from config
                 'signal_strength': signal_strength,
                 'stop_loss_pct': suggested_sl_pct,
                 # Pass other model outputs if needed by allocation logic
                 'model_outputs': model_outputs
            }
            # Note: Portfolio manager expects a list, here we simulate for one symbol
            allocated_risk = self.capital_manager.allocate_risk_capital([potential_trade_info])
            risk_capital_for_trade = allocated_risk.get(potential_trade_info['symbol'], 0.0)

            position_size_notional = 0.0
            if risk_capital_for_trade > 0 and suggested_sl_pct > 0:
                 position_size_notional = self.capital_manager.calculate_position_size_notional(
                      potential_trade_info['symbol'],
                      risk_capital_for_trade,
                      current_price_close, # Base size calculation on current close
                      suggested_sl_pct
                 )

            # 4. Execute Action (if not closed by SL/TP) & Calculate PnL
            # Use next_price_open for PnL calculation of the step
            step_pnl = 0.0
            is_opening_trade = False

            if self.current_position == 0: # Currently Flat
                if action == 1 and position_size_notional > 0: # Buy Signal
                    self.current_position = 1
                    self.entry_price = next_price_open * (1 + self.slippage_pct) # Enter at next open + slippage
                    self.current_notional_size = position_size_notional
                    cost = self._apply_costs(self.current_notional_size, True)
                    self.capital_manager.update_capital(-cost) # Deduct cost immediately
                    self.steps_in_trade = 1
                    self.stop_loss_level = self.entry_price * (1 - suggested_sl_pct)
                    self.take_profit_level = self.entry_price * (1 + suggested_tp_pct)
                    is_opening_trade = True
                elif action == 2 and position_size_notional > 0: # Sell Signal
                    self.current_position = -1
                    self.entry_price = next_price_open * (1 - self.slippage_pct) # Enter at next open - slippage
                    self.current_notional_size = position_size_notional
                    cost = self._apply_costs(self.current_notional_size, True)
                    self.capital_manager.update_capital(-cost) # Deduct cost immediately
                    self.steps_in_trade = 1
                    self.stop_loss_level = self.entry_price * (1 + suggested_sl_pct)
                    self.take_profit_level = self.entry_price * (1 - suggested_tp_pct)
                    is_opening_trade = True
                # Else (Hold signal or zero size): Do nothing

            elif self.current_position == 1: # Currently Long
                self.steps_in_trade += 1
                if action == 2: # Sell Signal (Close Long)
                    exit_price = next_price_open * (1 - self.slippage_pct) # Exit at next open - slippage
                    pnl = self._calculate_pnl(exit_price)
                    cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                    self.capital_manager.update_capital(pnl - cost)
                    step_pnl = pnl - cost # Realized PnL for this step
                    closed_trade = True
                else: # Hold Signal or Buy again (no change in position)
                    # Calculate unrealized PnL based on price change during the bar
                    step_pnl = self._calculate_pnl(next_price_open) - self._calculate_pnl(current_bar['open'])

            elif self.current_position == -1: # Currently Short
                self.steps_in_trade += 1
                if action == 1: # Buy Signal (Close Short)
                    exit_price = next_price_open * (1 + self.slippage_pct) # Exit at next open + slippage
                    pnl = self._calculate_pnl(exit_price)
                    cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                    self.capital_manager.update_capital(pnl - cost)
                    step_pnl = pnl - cost # Realized PnL for this step
                    closed_trade = True
                else: # Hold Signal or Sell again (no change in position)
                    # Calculate unrealized PnL based on price change during the bar
                    step_pnl = self._calculate_pnl(next_price_open) - self._calculate_pnl(current_bar['open'])

            # Reset position state if trade was closed by action
            if closed_trade and not hit_sl_tp: # Avoid resetting again if SL/TP already did
                self.current_position = 0
                self.entry_price = 0.0
                self.current_notional_size = 0.0
                self.steps_in_trade = 0
                self.stop_loss_level = None
                self.take_profit_level = None

        # --- Calculate Reward ---
        # Use PnL generated during this step (either realized or unrealized change)
        reward = 0.0
        if log_prob is not None: # Only calculate reward if agent acted
            environment_info = {
                'is_trade_closed': closed_trade,
                'hit_stop_loss': hit_sl_tp and pnl < 0, # True if SL specifically was hit
                'hit_take_profit': hit_sl_tp and pnl > 0, # True if TP specifically was hit
                'steps_in_trade': self.steps_in_trade if self.current_position != 0 else 0,
                'current_price': current_price_close, # Use close price for state context
                'entry_price': self.entry_price,
                'is_in_trade': self.current_position != 0,
                # Add other relevant info like trend direction if calculated in features
                # 'trend_direction': state_features.get('trend_indicator', 0)
            }
            # Use the calculated step_pnl for reward calculation
            reward = self.agent.calculate_reward(step_pnl, action, model_outputs, environment_info)

            # --- Store Transition ---
            terminal = (self.current_step == self.total_steps - 2) or (self.capital_manager.get_current_capital() <= 0)
            # Ensure state, action, etc. are valid before storing
            if state is not None and action is not None and log_prob is not None and value is not None:
                 self.agent.store_transition(state, action, reward, log_prob, value, terminal, model_outputs)
            else:
                 print(f"Warning: Invalid data for transition at step {self.current_step}. Skipping storage.")


        # --- Log History ---
        # Use the pnl calculated from SL/TP hit or the step_pnl if agent acted
        log_pnl = pnl if hit_sl_tp else step_pnl
        self._log_history(current_bar.name, action, reward, log_pnl, cost, closed_trade, current_price_close, hit_sl_tp)


        # --- Update Agent (Periodically) ---
        # Ensure memory has enough samples and size > batch_size
        if (self.current_step % self.sim_config['update_agent_every'] == 0 and
            len(self.agent.memory.get('rewards', [])) >= self.agent.batch_size):
             print(f"\n--- Step {self.current_step}: Updating Agent ({len(self.agent.memory['rewards'])} transitions) ---")
             try:
                  self.agent.update()
                  print(f"--- Agent Update Complete ---")
             except Exception as e:
                  print(f"Error during agent update: {e}")
                  # Optionally clear memory to prevent repeated errors on same data
                  # self.agent._clear_memory()


        # --- Print Status (Periodically) ---
        if self.current_step % self.sim_config['print_every'] == 0:
             print(f"Step: {self.current_step}/{self.total_steps} | "
                   f"Time: {current_bar.name} | "
                   f"Capital: ${self.capital_manager.get_current_capital():.2f} | "
                   f"Pos: {self.current_position} ({self.current_notional_size:.0f}) | "
                   f"Action: {action} | "
                   f"Reward: {reward:.4f} | "
                   f"Step PnL: {log_pnl:.4f}")

        # --- Advance Time ---
        self.current_step += 1
        self.last_action = action

        # Check for bankruptcy
        if self.capital_manager.get_current_capital() <= 0:
            print(f"Simulation stopped at step {self.current_step}: Capital depleted.")
            # Store final state if needed
            terminal = True
            # If RL, might need to store this final transition with large negative reward
            # self.agent.store_transition(...)
            return None, True # End simulation

        terminal = (self.current_step >= self.total_steps - 1)
        return reward, terminal

    def _log_history(self, timestamp, action, reward, pnl, cost, closed_trade, close_price, hit_sl_tp):
        """Helper function to log simulation step details."""
        self.history['timestamp'].append(timestamp)
        self.history['capital'].append(self.capital_manager.get_current_capital())
        self.history['position'].append(self.current_position)
        self.history['action'].append(action)
        self.history['reward'].append(reward)
        self.history['pnl'].append(pnl) # Log step PnL (realized or unrealized change)
        self.history['notional_size'].append(self.current_notional_size)
        self.history['entry_price'].append(self.entry_price if self.current_position != 0 else np.nan)
        self.history['close_price'].append(close_price)
        self.history['stop_loss'].append(self.stop_loss_level if self.current_position != 0 else np.nan)
        self.history['take_profit'].append(self.take_profit_level if self.current_position != 0 else np.nan)
        self.history['hit_sl_tp'].append(hit_sl_tp)


    def run(self):
        """Runs the entire simulation."""
        print(f"Starting simulation for {self.total_steps - self.seq_length - 1} steps...")
        # Use tqdm for progress bar
        pbar = tqdm(total=self.total_steps - self.seq_length - 1, desc="Simulating")
        while True:
            reward, terminal = self.step()
            pbar.update(1)
            if terminal:
                break
        pbar.close()
        print("Simulation finished.")

        # Final agent update if there's remaining data in memory
        if len(self.agent.memory.get('rewards', [])) > 0:
             print("Performing final agent update...")
             try:
                  self.agent.update()
                  print("Final agent update complete.")
             except Exception as e:
                  print(f"Error during final agent update: {e}")

        return self.get_results()

    def get_results(self):
        """Returns the simulation history and performance metrics."""
        if not self.history['timestamp']: # Check if history is empty
             print("Warning: Simulation history is empty. Cannot generate results.")
             return {
                  'history': pd.DataFrame(), 'initial_capital': self.capital_manager.initial_capital,
                  'final_capital': self.capital_manager.current_capital, 'total_return_pct': 0,
                  'num_trades': 0, 'win_rate': 0, 'max_drawdown': 0, 'sharpe_ratio': 0
             }

        history_df = pd.DataFrame(self.history)
        history_df.set_index('timestamp', inplace=True)

        # --- Calculate Performance Metrics ---
        initial_capital = self.capital_manager.initial_capital
        final_capital = self.capital_manager.get_current_capital()
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100 if initial_capital else 0

        # Calculate trades based on changes in position from flat or flips
        history_df['prev_position'] = history_df['position'].shift(1).fillna(0)
        trades = history_df[
             ((history_df['position'] != 0) & (history_df['prev_position'] == 0)) | # Entry from flat
             ((history_df['position'] * history_df['prev_position']) < 0) # Position flip
        ]
        num_trades = len(trades)

        # Calculate Win Rate (based on realized PnL when trade is closed)
        closed_trades_pnl = history_df.loc[history_df['hit_sl_tp'] | (history_df['action'].isin([1,2]) & history_df['prev_position'].isin([1,-1])), 'pnl']
        # Filter out costs that were already deducted for PnL calculation
        winning_trades = closed_trades_pnl[closed_trades_pnl > 0].count()
        losing_trades = closed_trades_pnl[closed_trades_pnl < 0].count()
        total_closed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed_trades) * 100 if total_closed_trades else 0

        # Calculate Max Drawdown
        capital_series = history_df['capital']
        rolling_max = capital_series.cummax()
        drawdown = (capital_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0

        # Calculate Sharpe Ratio (simplified, assumes daily returns if index is daily)
        # For higher frequency, need to annualize differently
        returns = capital_series.pct_change().dropna()
        sharpe_ratio = 0
        if not returns.empty and returns.std() != 0:
             # Assuming daily data for annualization factor sqrt(252)
             # Adjust factor based on actual data frequency (e.g., sqrt(252*24*60) for minutes)
             annualization_factor = np.sqrt(252) # Example for daily
             sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor

        results = {
            'history': history_df,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            # Add more metrics here: Sortino, Profit Factor, Avg Win/Loss etc.
        }
        print("\n--- Simulation Results ---")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital:   ${results['final_capital']:.2f}")
        print(f"Total Return:    {results['total_return_pct']:.2f}%")
        print(f"Number of Trades:{results['num_trades']}")
        print(f"Win Rate:        {results['win_rate']:.2f}%")
        print(f"Max Drawdown:    {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:    {results['sharpe_ratio']:.3f} (Annualized, assuming daily)")
        print("--------------------------")

        return results

    def plot_results(self, results=None):
        """Plots the capital curve and drawdown."""
        if results is None:
            results = self.get_results()

        history_df = results['history']

        if history_df.empty:
            print("No history data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Capital Curve
        ax1.plot(history_df.index, history_df['capital'], label='Capital')
        ax1.set_title(f"Simulation Results - {self.config.get('symbol', 'Unknown Symbol')}")
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True)
        ax1.legend()

        # Drawdown Plot
        capital_series = history_df['capital']
        rolling_max = capital_series.cummax()
        drawdown = (capital_series - rolling_max) / rolling_max * 100 # Percentage drawdown
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown, color='red', linewidth=1) # Drawdown line
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        # Optionally save the plot
        # results_dir = self.config.get('results_dir', 'results/')
        # plot_path = os.path.join(results_dir, f"{self.config.get('symbol', 'Unknown')}_results.png")
        # plt.savefig(plot_path)
        # print(f"Plot saved to {plot_path}")
        plt.show()


# --- Factory Function ---
def create_simulator(data, config):
    """
    Factory function to create the Trading Simulator.

    Args:
        data (pd.DataFrame): Historical market data.
        config (dict): Full configuration dictionary.

    Returns:
        TradingSimulator: The configured simulator instance.
    """
    # Need to import torch and os here if model loading requires it
    import torch
    import os
    return TradingSimulator(data=data, config=config)

