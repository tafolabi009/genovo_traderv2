# core/simulator.py

import pandas as pd
import numpy as np
from tqdm import tqdm # Progress bar
import matplotlib.pyplot as plt

# Import necessary components from other modules
# (Assuming they are structured correctly in the 'core' directory)
from core.features import create_feature_pipeline # Using the factory function
from core.model import create_model # Using the factory function
from core.strategy import create_strategy # Using the factory function
from core.portfolio_compound import create_capital_manager # Using the factory function

class TradingSimulator:
    """
    Simulates the trading environment for backtesting and RL agent training.
    Integrates feature extraction, model prediction, strategy execution,
    and capital management.
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
        model_path = config.get('model_config', {}).get('load_path')
        if model_path:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device)
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights from {model_path}. Error: {e}")
        self.device = next(self.model.parameters()).device # Get device model is on

        self.agent = create_strategy(self.model, config.get('strategy_config', {}))
        self.capital_manager = create_capital_manager(config.get('capital_config', {}))
        print("Components initialized.")

        # --- Simulation State ---
        self.current_step = self.seq_length
        self.current_position = 0 # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.current_notional_size = 0.0 # Size of the current position in notional value
        self.steps_in_trade = 0
        self.last_action = 0 # Hold

        # --- Simulation Parameters ---
        self.commission_pct = self.sim_config.get('commission_pct', 0.0005) # 0.05% per trade
        self.slippage_pct = self.sim_config.get('slippage_pct', 0.0002) # 0.02% slippage

        # --- Logging ---
        self.history = {
            'timestamp': [],
            'capital': [],
            'position': [],
            'action': [],
            'reward': [],
            'pnl': [],
            'notional_size': [],
            'entry_price': [],
            'close_price': []
        }

    def _get_default_sim_config(self):
        """Provides default simulator configuration."""
        return {
            'commission_pct': 0.0005,
            'slippage_pct': 0.0002,
            'update_agent_every': 128, # Steps between PPO updates
            'print_every': 1000 # Steps between status prints
        }

    def _prepare_data(self, data):
        """Ensures data has correct format and timestamp index."""
        df = data.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
             raise ValueError("Data must have a datetime index or a 'timestamp' column.")
        df = df.sort_index()
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        # Handle potential NaNs (forward fill is common)
        df.ffill(inplace=True)
        df.bfill(inplace=True) # Fill any remaining NaNs at the beginning
        return df

    def _get_state(self):
        """Extracts the current state (feature matrix) for the agent."""
        if self.current_step < self.seq_length:
             # Should not happen with proper initialization check
             raise IndexError("Not enough data points for sequence length.")

        # Get data window [current_step - seq_length : current_step]
        market_window = self.data.iloc[self.current_step - self.seq_length : self.current_step]

        # Extract features - Assuming transform returns a DataFrame
        # In a real scenario, fit() should be called beforehand
        if not self.feature_pipeline.feature_extractor.is_fitted:
             print("Warning: Feature extractor not fitted. Fitting on initial data.")
             # Fit on a larger initial chunk to get better stats
             fit_end_idx = min(self.seq_length * 5, len(self.data))
             self.feature_pipeline.fit(self.data.iloc[:fit_end_idx])

        features_df = self.feature_pipeline.transform(market_window)

        # Convert features to numpy array or tensor as required by the model
        state_array = features_df.values

        # Add position context if model uses it (optional)
        # state_array = np.hstack([state_array, np.full((len(state_array), 1), self.current_position)])

        return state_array # Shape: [seq_length, num_features]

    def _calculate_pnl(self, current_price):
        """Calculates PnL for the current open position."""
        if self.current_position == 0:
            return 0.0

        price_diff = current_price - self.entry_price
        # PnL = (Price Diff / Entry Price) * Notional Size * Position Direction
        # Avoid division by zero if entry price is somehow zero
        if abs(self.entry_price) < 1e-9:
            return 0.0
        pnl = (price_diff / self.entry_price) * self.current_notional_size * self.current_position
        return pnl

    def _apply_costs(self, trade_value, is_opening_trade):
        """Applies commission and slippage costs."""
        commission = trade_value * self.commission_pct
        slippage_cost = 0
        if is_opening_trade: # Apply slippage only when opening/modifying significantly
            slippage_cost = trade_value * self.slippage_pct

        return commission + slippage_cost

    def step(self):
        """Performs one simulation step."""
        if self.current_step >= self.total_steps -1: # Need one more step for reward calc
            return None, True # End of simulation

        # 1. Get current state (features)
        state = self._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim

        # 2. Get action from agent
        # select_action returns: (action, log_prob, value, pos_size_sugg, uncertainty, state_info)
        action, log_prob, value, _, _, model_outputs = self.agent.select_action(state_tensor)
        # Action: 0=Hold, 1=Buy, 2=Sell (adjust if agent uses different mapping)

        # 3. Get current price data for execution simulation
        current_bar = self.data.iloc[self.current_step]
        current_price = current_bar['close'] # Execute based on closing price of the current bar

        # 4. Determine Position Size
        # Pass model outputs needed by capital manager
        position_size_notional = self.capital_manager.get_position_size(current_price, model_outputs)

        # 5. Execute Action & Calculate PnL
        pnl = 0.0
        cost = 0.0
        closed_trade = False
        hit_stop_loss = False # Placeholder for SL logic

        # --- Trade Execution Logic ---
        if self.current_position == 0: # Currently Flat
            if action == 1 and position_size_notional > 0: # Buy Signal
                self.current_position = 1
                self.entry_price = current_price * (1 + self.slippage_pct) # Account for slippage on entry
                self.current_notional_size = position_size_notional
                cost = self._apply_costs(self.current_notional_size, True)
                self.capital_manager.update_capital(-cost) # Deduct cost immediately
                self.steps_in_trade = 1
            elif action == 2 and position_size_notional > 0: # Sell Signal
                self.current_position = -1
                self.entry_price = current_price * (1 - self.slippage_pct) # Account for slippage on entry
                self.current_notional_size = position_size_notional
                cost = self._apply_costs(self.current_notional_size, True)
                self.capital_manager.update_capital(-cost) # Deduct cost immediately
                self.steps_in_trade = 1
            # Else (Hold signal or zero size): Do nothing

        elif self.current_position == 1: # Currently Long
            self.steps_in_trade += 1
            if action == 2: # Sell Signal (Close Long)
                pnl = self._calculate_pnl(current_price)
                cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                self.capital_manager.update_capital(pnl - cost)
                self.current_position = 0
                self.entry_price = 0.0
                self.current_notional_size = 0.0
                closed_trade = True
            elif action == 0: # Hold Signal
                pnl = self._calculate_pnl(current_price) # Unrealized PnL for reward shaping
            # elif action == 1: # Hold/Add to Long (not implemented here, simple flip/close)
            #     pnl = self._calculate_pnl(current_price) # Unrealized PnL

        elif self.current_position == -1: # Currently Short
            self.steps_in_trade += 1
            if action == 1: # Buy Signal (Close Short)
                pnl = self._calculate_pnl(current_price)
                cost = self._apply_costs(self.current_notional_size, False) # Commission on close
                self.capital_manager.update_capital(pnl - cost)
                self.current_position = 0
                self.entry_price = 0.0
                self.current_notional_size = 0.0
                closed_trade = True
            elif action == 0: # Hold Signal
                pnl = self._calculate_pnl(current_price) # Unrealized PnL for reward shaping
            # elif action == 2: # Hold/Add to Short (not implemented here)
            #     pnl = self._calculate_pnl(current_price) # Unrealized PnL

        # 6. Calculate Reward
        # Environment info needed by agent's reward function
        environment_info = {
            'is_trade_closed': closed_trade,
            'hit_stop_loss': hit_stop_loss, # TODO: Implement SL check based on model output/config
            'steps_in_trade': self.steps_in_trade if self.current_position != 0 else 0,
            'current_price': current_price,
            'entry_price': self.entry_price,
            'is_in_trade': self.current_position != 0,
            # Add other relevant info like trend direction if calculated
            # 'trend_direction': self._calculate_trend_direction(...)
        }
        # Use PnL from the *change* over the step, could be unrealized or realized
        step_pnl = pnl if not closed_trade else (pnl - cost) # Use realized PnL if trade closed
        reward = self.agent.calculate_reward(step_pnl, action, model_outputs, environment_info)

        # 7. Store Transition
        terminal = (self.current_step == self.total_steps - 2) or (self.capital_manager.get_current_capital() <= 0)
        self.agent.store_transition(state, action, reward, log_prob, value, terminal, model_outputs)

        # 8. Log History
        self.history['timestamp'].append(current_bar.name)
        self.history['capital'].append(self.capital_manager.get_current_capital())
        self.history['position'].append(self.current_position)
        self.history['action'].append(action)
        self.history['reward'].append(reward)
        self.history['pnl'].append(step_pnl) # Log step PnL (realized or unrealized change)
        self.history['notional_size'].append(self.current_notional_size)
        self.history['entry_price'].append(self.entry_price if self.current_position != 0 else np.nan)
        self.history['close_price'].append(current_price)


        # 9. Update Agent (Periodically)
        if (self.current_step % self.sim_config['update_agent_every'] == 0 and len(self.agent.memory['rewards']) > self.agent.batch_size):
             print(f"\n--- Step {self.current_step}: Updating Agent ---")
             self.agent.update()
             print(f"--- Agent Update Complete ---")


        # 10. Print Status (Periodically)
        if self.current_step % self.sim_config['print_every'] == 0:
             print(f"Step: {self.current_step}/{self.total_steps} | "
                   f"Capital: ${self.capital_manager.get_current_capital():.2f} | "
                   f"Position: {self.current_position} | "
                   f"Action: {action} | "
                   f"Reward: {reward:.4f} | "
                   f"Step PnL: {step_pnl:.4f}")

        # 11. Advance Time
        self.current_step += 1
        self.last_action = action

        # Check for bankruptcy
        if self.capital_manager.get_current_capital() <= 0:
            print(f"Simulation stopped at step {self.current_step}: Capital depleted.")
            return None, True # End simulation

        return reward, terminal


    def run(self):
        """Runs the entire simulation."""
        print(f"Starting simulation for {self.total_steps - self.seq_length} steps...")
        for step_num in tqdm(range(self.total_steps - self.seq_length -1), desc="Simulating"):
            _, terminal = self.step()
            if terminal:
                print(f"Simulation ended early at step {self.current_step}.")
                break
        print("Simulation finished.")
        return self.get_results()

    def get_results(self):
        """Returns the simulation history and performance metrics."""
        history_df = pd.DataFrame(self.history)
        history_df.set_index('timestamp', inplace=True)

        # --- Calculate Performance Metrics ---
        initial_capital = self.capital_manager.initial_capital
        final_capital = self.capital_manager.get_current_capital()
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100 if initial_capital else 0
        num_trades = history_df[history_df['action'].isin([1, 2]) & (history_df['position'].shift(1) == 0)].shape[0] # Count entries from flat

        # More detailed metrics can be added (Sharpe, Sortino, Max Drawdown, etc.)
        # Requires calculating daily/periodic returns from the capital curve

        results = {
            'history': history_df,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            # Add more metrics here
        }
        print("\n--- Simulation Results ---")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital:   ${results['final_capital']:.2f}")
        print(f"Total Return:    {results['total_return_pct']:.2f}%")
        print(f"Number of Trades:{results['num_trades']}")
        print("--------------------------")

        return results

    def plot_results(self, results=None):
        """Plots the capital curve."""
        if results is None:
            results = self.get_results()

        history_df = results['history']

        if history_df.empty:
            print("No history data to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(history_df.index, history_df['capital'])
        plt.title('Capital Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital ($)')
        plt.grid(True)
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
    # Need to import torch here if model loading requires it
    import torch
    return TradingSimulator(data=data, config=config)
