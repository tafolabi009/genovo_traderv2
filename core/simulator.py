# core/simulator.py

import pandas as pd
import numpy as np
from tqdm import tqdm # Progress bar
import matplotlib.pyplot as plt
import torch
import os # Import os

# Import necessary components from other modules
# from core.features import create_feature_pipeline # No longer needed here
from core.model import create_model # Using the factory function
from core.strategy import create_strategy # Using the factory function
from core.portfolio_compound import create_portfolio_capital_manager # Using the factory function

class TradingSimulator:
    """
    Simulates the trading environment using pre-calculated features.
    Integrates model prediction, strategy execution, capital management,
    and basic SL/TP hit detection within bars.
    """
    def __init__(self, data_ohlcv, data_features, config):
        """
        Initializes the Trading Simulator with pre-calculated features.

        Args:
            data_ohlcv (pd.DataFrame): DataFrame containing historical OHLCV data.
                                       Timestamp should be the index.
            data_features (pd.DataFrame): DataFrame containing pre-calculated and scaled features.
                                          Must have the same DatetimeIndex as data_ohlcv.
            config (dict): Configuration dictionary containing settings for model,
                           strategy, capital, simulator.
        """
        self.config = config
        self.sim_config = config.get('simulator_config', self._get_default_sim_config())

        # --- Data Handling ---
        self.data_ohlcv = self._prepare_data(data_ohlcv) # Clean OHLCV data
        self.features_data = data_features # Assume features are already scaled and cleaned

        # --- Alignment Check ---
        if not self.data_ohlcv.index.equals(self.features_data.index):
             # Attempt to align if possible (e.g., if one has extra rows)
             common_index = self.data_ohlcv.index.intersection(self.features_data.index)
             if len(common_index) < len(self.data_ohlcv) * 0.9: # Check if significant mismatch
                  raise ValueError("OHLCV data and Feature data indices do not align significantly.")
             self.data_ohlcv = self.data_ohlcv.loc[common_index]
             self.features_data = self.features_data.loc[common_index]
             print("Warning: Aligned OHLCV and Feature data indices.")

        self.total_steps = len(self.data_ohlcv)
        self.seq_length = config.get('model_config', {}).get('seq_length', 201) # Use updated seq_length

        if self.total_steps <= self.seq_length:
            raise ValueError(f"Data length ({self.total_steps}) must be greater than sequence length ({self.seq_length})")

        self.num_features = self.features_data.shape[1]
        logger.info(f"Simulator using {self.num_features} pre-calculated features.")


        # --- Core Components ---
        print("Initializing simulator components...")
        # Feature pipeline is NOT needed here as features are pre-calculated

        # Ensure model config num_features matches the provided features
        model_cfg = config.get('model_config', {})
        if model_cfg.get('num_features') != self.num_features:
             print(f"Warning: Model config num_features ({model_cfg.get('num_features')}) does not match actual features ({self.num_features}). Adjusting model config.")
             model_cfg['num_features'] = self.num_features
             config['model_config'] = model_cfg # Update main config dict

        self.model = create_model(config.get('model_config', {})) # Create model with correct num_features
        # Load pre-trained model weights if specified
        load_model_path = model_cfg.get('load_path') # Use the potentially updated path from main.py
        if load_model_path and os.path.exists(load_model_path):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model.load_state_dict(torch.load(load_model_path, map_location=device))
                self.model.to(device)
                print(f"Loaded model weights from {load_model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights from {load_model_path}. Error: {e}")
        else:
             print(f"No valid model path found ({load_model_path}). Model using initial weights.")
        self.device = next(self.model.parameters()).device # Get device model is on

        self.agent = create_strategy(self.model, config.get('strategy_config', {}))
        # Use PortfolioCapitalManager
        self.capital_manager = create_portfolio_capital_manager(config.get('portfolio_capital_config', {}))
        print("Components initialized.")

        # --- Simulation State ---
        self.current_step = self.seq_length # Start after initial sequence length needed for first state
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
        """Ensures OHLCV data has correct format and timestamp index."""
        # This is mostly duplicated from the preprocessor, keep basic checks
        df = data.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
             try: df.index = pd.to_datetime(df.index)
             except Exception as e: raise ValueError(f"OHLCV index is not DatetimeIndex and could not be converted: {e}")

        df = df.sort_index()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.lower() # Standardize to lower case
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"OHLCV Data must contain columns: {required_cols}")
        # Basic NaN check - should have been cleaned by preprocessor
        if df[required_cols].isnull().values.any():
             print("Warning: NaNs found in input OHLCV data. Applying ffill/bfill.")
             df.ffill(inplace=True); df.bfill(inplace=True)
        return df

    def _get_state(self):
        """
        Extracts the current state by slicing the pre-calculated feature matrix.
        """
        if self.current_step < self.seq_length:
             raise IndexError("Not enough data points for sequence length.")

        # Slice the pre-calculated features DataFrame
        # Window is [current_step - seq_length : current_step]
        start_idx = self.current_step - self.seq_length
        end_idx = self.current_step
        state_features_df = self.features_data.iloc[start_idx : end_idx]

        # The features should already be scaled and cleaned
        if state_features_df.shape[0] != self.seq_length:
             raise ValueError(f"Sliced state features have incorrect length: {state_features_df.shape[0]} != {self.seq_length}")
        if state_features_df.isnull().values.any():
             # This should ideally not happen if pre-calculation was done correctly
             raise ValueError(f"NaN values found in pre-calculated feature slice at step {self.current_step}. Check pre-calculation step.")

        state_array = state_features_df.values
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

        # --- Get Current Bar Data (from OHLCV data) ---
        current_bar = self.data_ohlcv.iloc[self.current_step]
        next_bar = self.data_ohlcv.iloc[self.current_step + 1] # Needed for reward calculation based on next open
        current_price_close = current_bar['close']
        current_price_high = current_bar['high']
        current_price_low = current_bar['low']
        next_price_open = next_bar['open'] # Price for calculating step PnL/reward

        # --- Check for SL/TP Hit Before Getting New Action ---
        pnl = 0.0
        cost = 0.0
        closed_trade = False
        hit_sl_tp = False # Flag specifically for SL/TP hit
        exit_price_sl_tp = None # Store the exit price if SL/TP hit

        if self.current_position == 1: # Long Position Check
            if self.stop_loss_level and current_price_low <= self.stop_loss_level:
                exit_price_sl_tp = self.stop_loss_level * (1 - self.slippage_pct)
                pnl = self._calculate_pnl(exit_price_sl_tp)
                cost = self._apply_costs(self.current_notional_size, False)
                closed_trade = True; hit_sl_tp = True
                # print(f"Step {self.current_step}: Long SL hit at {exit_price_sl_tp:.5f} (Level: {self.stop_loss_level:.5f})")
            elif self.take_profit_level and current_price_high >= self.take_profit_level:
                exit_price_sl_tp = self.take_profit_level * (1 + self.slippage_pct)
                pnl = self._calculate_pnl(exit_price_sl_tp)
                cost = self._apply_costs(self.current_notional_size, False)
                closed_trade = True; hit_sl_tp = True
                # print(f"Step {self.current_step}: Long TP hit at {exit_price_sl_tp:.5f} (Level: {self.take_profit_level:.5f})")

        elif self.current_position == -1: # Short Position Check
            if self.stop_loss_level and current_price_high >= self.stop_loss_level:
                exit_price_sl_tp = self.stop_loss_level * (1 + self.slippage_pct)
                pnl = self._calculate_pnl(exit_price_sl_tp)
                cost = self._apply_costs(self.current_notional_size, False)
                closed_trade = True; hit_sl_tp = True
                # print(f"Step {self.current_step}: Short SL hit at {exit_price_sl_tp:.5f} (Level: {self.stop_loss_level:.5f})")
            elif self.take_profit_level and current_price_low <= self.take_profit_level:
                exit_price_sl_tp = self.take_profit_level * (1 - self.slippage_pct)
                pnl = self._calculate_pnl(exit_price_sl_tp)
                cost = self._apply_costs(self.current_notional_size, False)
                closed_trade = True; hit_sl_tp = True
                # print(f"Step {self.current_step}: Short TP hit at {exit_price_sl_tp:.5f} (Level: {self.take_profit_level:.5f})")

        # Update capital and reset position if SL/TP was hit
        if closed_trade:
            self.capital_manager.update_capital(pnl - cost) # Update capital with realized PnL
            self.current_position = 0
            self.entry_price = 0.0
            self.current_notional_size = 0.0
            self.steps_in_trade = 0
            self.stop_loss_level = None
            self.take_profit_level = None


        # --- If no SL/TP hit, get Agent Action ---
        action = 0 # Default to Hold if already closed or error occurs
        log_prob, value, model_outputs = None, None, {}
        state = None # Initialize state
        step_pnl = 0.0 # PnL for this step if no SL/TP hit

        if not closed_trade: # Only get action if still in market or flat
            # 1. Get current state (now slicing pre-calculated features)
            try:
                 state = self._get_state() # Gets the [seq_length, num_features] array
                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim
            except ValueError as e:
                 print(f"Error getting state slice at step {self.current_step}: {e}. Skipping step.")
                 self._log_history(current_bar.name, np.nan, np.nan, np.nan, np.nan, np.nan, current_price_close, hit_sl_tp)
                 self.current_step += 1
                 return None, (self.current_step >= self.total_steps -1) or (self.capital_manager.get_current_capital() <= 0)
            except IndexError as e:
                 print(f"Error slicing state at step {self.current_step}: {e}. Likely end of data issue. Stopping.")
                 return None, True # End simulation if slicing fails


            # 2. Get action from agent
            try:
                action, log_prob, value, _, _, model_outputs = self.agent.select_action(state_tensor)
                # Action: 0=Hold, 1=Buy, 2=Sell
            except Exception as e:
                 print(f"Error getting action from agent at step {self.current_step}: {e}. Assuming Hold.")
                 action = 0 # Default to hold on error
                 log_prob, value = 0.0, 0.0 # Assign default values
                 model_outputs = {} # Empty dict


            # 3. Determine Position Size based on model output and capital manager
            suggested_sl_pct = model_outputs.get('stop_loss', torch.tensor([[0.01]], device=self.device)).item() # Default 1%
            suggested_tp_pct = model_outputs.get('take_profit', torch.tensor([[0.02]], device=self.device)).item() # Default 2%
            signal_strength = 0.0
            policy_logits = model_outputs.get('policy_logits')
            if policy_logits is not None:
                 signal_strength = torch.softmax(policy_logits, dim=-1).max().item()

            potential_trade_info = {
                 'symbol': self.config.get('symbol', 'UNKNOWN'),
                 'signal_strength': signal_strength,
                 'stop_loss_pct': suggested_sl_pct,
                 'model_outputs': {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in model_outputs.items()} # Convert tensors for manager
            }
            allocated_risk = self.capital_manager.allocate_risk_capital([potential_trade_info])
            risk_capital_for_trade = allocated_risk.get(potential_trade_info['symbol'], 0.0)

            position_size_notional = 0.0
            if risk_capital_for_trade > 0 and suggested_sl_pct > 0:
                 position_size_notional = self.capital_manager.calculate_position_size_notional(
                      potential_trade_info['symbol'], risk_capital_for_trade,
                      current_price_close, suggested_sl_pct
                 )

            # 4. Execute Action & Calculate Step PnL
            is_opening_trade = False
            current_pos_before_action = self.current_position # Store state before action

            if self.current_position == 0: # Currently Flat
                if action == 1 and position_size_notional > 0: # Buy Signal
                    self.current_position = 1
                    self.entry_price = next_price_open * (1 + self.slippage_pct)
                    self.current_notional_size = position_size_notional
                    cost = self._apply_costs(self.current_notional_size, True)
                    self.capital_manager.update_capital(-cost)
                    self.steps_in_trade = 1
                    self.stop_loss_level = self.entry_price * (1 - suggested_sl_pct)
                    self.take_profit_level = self.entry_price * (1 + suggested_tp_pct)
                    is_opening_trade = True
                elif action == 2 and position_size_notional > 0: # Sell Signal
                    self.current_position = -1
                    self.entry_price = next_price_open * (1 - self.slippage_pct)
                    self.current_notional_size = position_size_notional
                    cost = self._apply_costs(self.current_notional_size, True)
                    self.capital_manager.update_capital(-cost)
                    self.steps_in_trade = 1
                    self.stop_loss_level = self.entry_price * (1 + suggested_sl_pct)
                    self.take_profit_level = self.entry_price * (1 - suggested_tp_pct)
                    is_opening_trade = True

            elif self.current_position == 1: # Currently Long
                self.steps_in_trade += 1
                if action == 2: # Sell Signal (Close Long)
                    exit_price = next_price_open * (1 - self.slippage_pct)
                    pnl = self._calculate_pnl(exit_price)
                    cost = self._apply_costs(self.current_notional_size, False)
                    self.capital_manager.update_capital(pnl - cost)
                    step_pnl = pnl - cost # Realized PnL for this step
                    closed_trade = True
                else: # Hold Signal or Buy again
                    # Calculate unrealized PnL based on price change using next open
                    step_pnl = self._calculate_pnl(next_price_open) - self._calculate_pnl(current_bar['open'])


            elif self.current_position == -1: # Currently Short
                self.steps_in_trade += 1
                if action == 1: # Buy Signal (Close Short)
                    exit_price = next_price_open * (1 + self.slippage_pct)
                    pnl = self._calculate_pnl(exit_price)
                    cost = self._apply_costs(self.current_notional_size, False)
                    self.capital_manager.update_capital(pnl - cost)
                    step_pnl = pnl - cost # Realized PnL for this step
                    closed_trade = True
                else: # Hold Signal or Sell again
                    # Calculate unrealized PnL based on price change using next open
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
        # Use PnL generated during this step (either realized from SL/TP/Action or unrealized change)
        reward = 0.0
        log_pnl = pnl if hit_sl_tp else step_pnl # PnL to log and use for reward

        # Only calculate reward and store transition if agent actually provided an action
        if state is not None and log_prob is not None and value is not None:
            environment_info = {
                'is_trade_closed': closed_trade,
                'hit_stop_loss': hit_sl_tp and log_pnl < 0, # True if SL specifically was hit
                'hit_take_profit': hit_sl_tp and log_pnl > 0, # True if TP specifically was hit
                'steps_in_trade': self.steps_in_trade if self.current_position != 0 else 0,
                'current_price': current_price_close, # Use close price for state context
                'entry_price': self.entry_price,
                'is_in_trade': self.current_position != 0,
            }
            # Use the calculated log_pnl for reward calculation
            reward = self.agent.calculate_reward(log_pnl, action, model_outputs, environment_info)

            # --- Store Transition ---
            terminal = (self.current_step == self.total_steps - 2) or (self.capital_manager.get_current_capital() <= 0)
            self.agent.store_transition(state, action, reward, log_prob, value, terminal, model_outputs)


        # --- Log History ---
        self._log_history(current_bar.name, action, reward, log_pnl, cost, closed_trade, current_price_close, hit_sl_tp)


        # --- Update Agent (Periodically) ---
        if (self.current_step % self.sim_config['update_agent_every'] == 0 and
            len(self.agent.memory.get('rewards', [])) >= self.agent.batch_size):
             print(f"\n--- Step {self.current_step}: Updating Agent ({len(self.agent.memory['rewards'])} transitions) ---")
             try:
                  self.agent.update()
                  print(f"--- Agent Update Complete ---")
             except Exception as e:
                  print(f"Error during agent update: {e}")


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
            terminal = True
            return None, True # End simulation

        terminal = (self.current_step >= self.total_steps - 1)
        return reward, terminal

    # _log_history, run, get_results, plot_results remain the same as previous version
    # ... (Keep the rest of the methods from simulator_py_update) ...
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
        # A trade is closed if hit_sl_tp is True OR if action caused position to go to 0 or flip
        closed_trade_mask = history_df['hit_sl_tp'] | \
                            ((history_df['position']==0) & (history_df['prev_position']!=0)) | \
                            ((history_df['position'] * history_df['prev_position']) < 0)

        closed_trades_pnl = history_df.loc[closed_trade_mask, 'pnl']
        # Filter out potential zero PnL entries if needed (e.g., cost deductions caused 0 PnL)
        # closed_trades_pnl = closed_trades_pnl[closed_trades_pnl != 0]

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
        returns = capital_series.pct_change().dropna()
        sharpe_ratio = 0
        if not returns.empty and returns.std() != 0:
             # Infer frequency for annualization factor (basic example)
             time_diff = history_df.index.to_series().diff().median()
             periods_per_year = 0
             if time_diff <= pd.Timedelta(minutes=1): periods_per_year = 252*24*60 # Approx minutes in trading year
             elif time_diff <= pd.Timedelta(hours=1): periods_per_year = 252*24 # Approx hours
             elif time_diff <= pd.Timedelta(days=1): periods_per_year = 252 # Approx days
             else: periods_per_year = 1 # Default if frequency is low

             annualization_factor = np.sqrt(periods_per_year) if periods_per_year > 0 else 1
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
        print(f"Sharpe Ratio:    {results['sharpe_ratio']:.3f} (Annualized)")
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
        results_dir = self.config.get('results_dir', 'results/')
        os.makedirs(results_dir, exist_ok=True) # Ensure dir exists
        plot_path = os.path.join(results_dir, f"{self.config.get('symbol', 'Unknown')}_results.png")
        try:
             plt.savefig(plot_path)
             print(f"Plot saved to {plot_path}")
        except Exception as e:
             print(f"Error saving plot: {e}")
        plt.show()


# --- Factory Function ---
def create_simulator(data_ohlcv, data_features, config):
    """
    Factory function to create the Trading Simulator using pre-calculated features.

    Args:
        data_ohlcv (pd.DataFrame): Historical OHLCV data.
        data_features (pd.DataFrame): Pre-calculated and scaled feature data.
        config (dict): Full configuration dictionary.

    Returns:
        TradingSimulator: The configured simulator instance.
    """
    # Need to import torch and os here if model loading requires it
    import torch
    import os
    import logging # Import logger
    global logger # Access logger defined in main
    logger = logging.getLogger("genovo_traderv2")
    return TradingSimulator(data_ohlcv=data_ohlcv, data_features=data_features, config=config)

