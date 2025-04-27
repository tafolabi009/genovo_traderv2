# core/strategy.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import math

# Assuming EnhancedTradingModel is in core.model and can be imported
# from core.model import EnhancedTradingModel
# For now, let's assume the model is passed during initialization

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent incorporating synthesized
    strategies inspired by successful trading algorithms.
    Leverages outputs from EnhancedTradingModel for sophisticated reward shaping.
    """
    def __init__(self, model, learning_rate=1e-4, gamma=0.99, lambda_gae=0.95,
                 ppo_epsilon=0.2, ppo_epochs=10, batch_size=64,
                 entropy_coef=0.01, value_loss_coef=0.5, config=None):
        """
        Initializes the PPO agent.

        Args:
            model (nn.Module): The policy and value network (EnhancedTradingModel).
            learning_rate (float): Optimizer learning rate.
            gamma (float): Discount factor for future rewards.
            lambda_gae (float): Factor for Generalized Advantage Estimation (GAE).
            ppo_epsilon (float): Clipping parameter for PPO.
            ppo_epochs (int): Number of epochs to train on a collected batch.
            batch_size (int): Size of minibatches for training.
            entropy_coef (float): Coefficient for entropy bonus in loss.
            value_loss_coef (float): Coefficient for value loss.
            config (dict): Configuration for reward shaping and agent behavior.
                           Example keys: 'reward_config', 'catastrophic_memory_size'.
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.config = config or {}
        self.reward_config = self.config.get('reward_config', self._get_default_reward_config())

        # Memory for PPO updates
        self.memory = {
            'states': [], 'actions': [], 'rewards': [], 'log_probs': [],
            'values': [], 'terminals': [], 'advantages': [], 'returns': [],
            # Store model outputs used in reward calculation for potential analysis
            'state_infos': []
        }

        # Catastrophic memory (stores recent significant loss events)
        # Stores tuples: (reward, action_taken, state_info_snapshot)
        self.catastrophic_memory = deque(maxlen=self.config.get('catastrophic_memory_size', 20))
        self.recent_rewards = deque(maxlen=100) # For calculating rolling Sharpe/Sortino proxies

    def _get_default_reward_config(self):
        """Provides default parameters for the reward function."""
        return {
            # PnL Scaling & Asymmetry
            'pnl_scale': 100.0,           # Multiplier for raw PnL
            'profit_bonus_factor': 1.2,   # Extra bonus for positive PnL
            'loss_penalty_factor': 1.5,   # Extra penalty for negative PnL (loss aversion)

            # Risk Management Penalties
            'max_loss_penalty': -50.0,    # Large penalty if a trade hits max stop loss %
            'uncertainty_penalty_factor': 0.5, # Penalty scaled by model uncertainty
            'high_volatility_penalty_factor': 0.3, # Penalty scaled by model volatility prediction

            # Regime-Based Rewards/Penalties (Assuming 0: Trend, 1: MeanRev, 2: Uncertain)
            'trend_follow_bonus': 0.1,    # Bonus for acting WITH predicted trend
            'mean_revert_bonus': 0.1,     # Bonus for acting counter-trend in MeanRev regime
            'trend_against_penalty': -0.2,# Penalty for acting AGAINST predicted trend
            'mean_revert_against_penalty': -0.2, # Penalty for acting trend-like in MeanRev regime
            'uncertain_regime_penalty': -0.1, # Penalty for taking aggressive actions in uncertain regime

            # Trade Duration & Efficiency
            'holding_penalty_factor': -0.001, # Small penalty per step holding a position
            'profit_capture_bonus': 0.5,  # Bonus for closing a profitable trade
            'quick_profit_bonus_factor': 0.2, # Extra bonus if profit target hit quickly (vs. model horizon)

            # Consistency (Proxy)
            'sharpe_ratio_bonus_factor': 0.05, # Bonus scaled by recent reward std dev (proxy)

            # Catastrophic Event Handling
            'catastrophic_threshold': -20.0, # Reward level considered catastrophic
            'catastrophic_repeat_penalty': -10.0, # Penalty for taking same action after recent catastrophe

            # Action Penalties (e.g., for over-trading)
            'transaction_cost_penalty': -0.01 # Simple penalty per Buy/Sell action
        }

    def select_action(self, state):
        """
        Selects an action based on the current state using the policy network.

        Args:
            state (torch.Tensor): The current market state representation (features).
                                  Shape: [seq_len, num_features] or [batch_size, seq_len, num_features]

        Returns:
            tuple: (action, log_prob, value, position_size, uncertainty, state_info)
                   state_info contains all relevant model outputs for reward calculation.
        """
        if not isinstance(state, torch.Tensor):
            # Assuming state is a numpy array [seq_len, num_features]
            state = torch.FloatTensor(state).unsqueeze(0) # Add batch dimension
        elif state.dim() == 2:
            # Add batch dimension if it's [seq_len, num_features]
             state = state.unsqueeze(0)

        # Ensure model is in evaluation mode for action selection
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(state) # Get all outputs from EnhancedTradingModel

        # Extract necessary outputs
        policy_logits = outputs['policy_logits']
        value = outputs['value']
        position_size = outputs['position_size'] # Model suggests position size
        uncertainty = outputs['uncertainty']
        regime_logits = outputs['regime_logits']
        volatility = outputs['volatility']
        sl_suggestion = outputs['stop_loss']
        tp_suggestion = outputs['take_profit']
        horizon_suggestion = outputs['trade_horizon']

        # Get action probabilities and sample action
        action_probs = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Package state info for reward calculation
        state_info = {
            'value': value.item(),
            'position_size_suggestion': position_size.item(),
            'uncertainty': uncertainty.item(),
            'regime_logits': regime_logits.cpu().numpy(), # Store numpy array
            'volatility': volatility.item(),
            'stop_loss_suggestion': sl_suggestion.item(),
            'take_profit_suggestion': tp_suggestion.item(),
            'trade_horizon_suggestion': horizon_suggestion.item(),
            # Add raw policy logits/probs if needed for analysis
            'policy_probs': action_probs.cpu().numpy()
        }

        # Note: Actual position size might be adjusted by compounding/risk logic later
        # The action here is Buy/Sell/Hold (e.g., 1/2/0)
        return (action.item(), log_prob.item(), value.item(),
                position_size.item(), uncertainty.item(), state_info)

    def store_transition(self, state, action, reward, log_prob, value, terminal, state_info):
        """Stores a transition in memory for PPO update."""
        # Ensure tensors are on CPU and detached
        self.memory['states'].append(torch.FloatTensor(state).cpu().detach())
        self.memory['actions'].append(torch.tensor(action, dtype=torch.long).cpu().detach())
        self.memory['rewards'].append(torch.tensor(reward, dtype=torch.float32).cpu().detach())
        self.memory['log_probs'].append(torch.tensor(log_prob, dtype=torch.float32).cpu().detach())
        self.memory['values'].append(torch.tensor(value, dtype=torch.float32).cpu().detach())
        self.memory['terminals'].append(torch.tensor(terminal, dtype=torch.bool).cpu().detach())
        self.memory['state_infos'].append(state_info) # Store the dictionary

        # Update recent rewards for consistency metrics
        self.recent_rewards.append(reward)

        # Check for catastrophic event
        if reward < self.reward_config['catastrophic_threshold']:
            self.catastrophic_memory.append((reward, action, state_info))

    def _calculate_advantages_returns(self):
        """Calculates GAE advantages and returns for PPO update."""
        # Detach tensors and ensure they are on CPU
        values = torch.stack(self.memory['values']).cpu().detach().view(-1, 1)
        rewards = torch.stack(self.memory['rewards']).cpu().detach().view(-1, 1)
        terminals = torch.stack(self.memory['terminals']).cpu().detach().view(-1, 1)

        num_steps = len(values)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        # Need value of the next state for the last step calculation
        if not terminals[-1]:
            # Get the last state from memory
            last_state = self.memory['states'][-1].unsqueeze(0) # Add batch dim
            self.model.eval() # Ensure eval mode
            with torch.no_grad():
                 # Pass state to model to get value prediction
                 # Ensure state is on the correct device if using GPU
                 last_state = last_state.to(next(self.model.parameters()).device)
                 next_value = self.model(last_state)['value'].cpu().detach()
        else:
            next_value = torch.zeros(1, 1) # Terminal state has zero value

        # Calculate advantages and returns backwards
        for t in reversed(range(num_steps)):
            # Determine the value of the state that follows state t
            if t == num_steps - 1:
                next_non_terminal = 1.0 - terminals[t].float() # Is state t+1 terminal?
                next_val = next_value # Use the bootstrapped value calculated above
            else:
                next_non_terminal = 1.0 - terminals[t+1].float() # Is state t+2 terminal?
                next_val = values[t+1] # Use the stored value of state t+1

            # Calculate the TD error (delta)
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            # Calculate the advantage using GAE formula
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam

        # Calculate returns by adding advantages to values
        returns = advantages + values

        # Normalize advantages (optional but often recommended)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Store calculated advantages and returns back into memory
        self.memory['advantages'] = advantages
        self.memory['returns'] = returns


    def update(self):
        """Performs the PPO update step."""
        if not self.memory['rewards']: # Check if memory is empty
             print("Memory is empty, skipping PPO update.")
             return

        # Calculate advantages and returns first
        self._calculate_advantages_returns()

        # Prepare data for training - ensure tensors are correctly shaped and detached
        states = torch.stack(self.memory['states']).cpu().detach()
        # Ensure actions are Long type for Categorical distribution
        actions = torch.stack(self.memory['actions']).cpu().detach().view(-1, 1).long()
        old_log_probs = torch.stack(self.memory['log_probs']).cpu().detach().view(-1, 1)
        advantages = self.memory['advantages'].cpu().detach()
        returns = self.memory['returns'].cpu().detach()

        n_samples = len(states)
        indices = np.arange(n_samples)

        # Ensure model is in training mode
        self.model.train()
        # Move model to appropriate device (e.g., GPU if available)
        device = next(self.model.parameters()).device

        # PPO training loop
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                if end > n_samples: # Avoid partial batch if smaller than batch_size
                    continue
                batch_indices = indices[start:end]

                # Get minibatch data and move to device
                batch_states = states[batch_indices].to(device)
                batch_actions = actions[batch_indices].to(device)
                batch_old_log_probs = old_log_probs[batch_indices].to(device)
                batch_advantages = advantages[batch_indices].to(device)
                batch_returns = returns[batch_indices].to(device)

                # Get current policy predictions for the batch
                outputs = self.model(batch_states)
                policy_logits = outputs['policy_logits']
                values = outputs['value'] # Critic's value estimate for the batch states

                # Calculate new log probabilities and entropy
                action_probs = torch.softmax(policy_logits, dim=-1)
                dist = Categorical(action_probs)
                # Ensure batch_actions is squeezed correctly for log_prob
                new_log_probs = dist.log_prob(batch_actions.squeeze(-1)).view(-1, 1)
                entropy = dist.entropy().mean()

                # Calculate PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate surrogate objectives (Policy Loss)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss (MSE between predicted value and actual returns)
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Total loss
                loss = (policy_loss
                        - self.entropy_coef * entropy          # Encourage exploration
                        + self.value_loss_coef * value_loss)   # Improve value estimates

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        # Clear memory after update
        self._clear_memory()


    def _clear_memory(self):
        """Clears the memory buffers."""
        for key in self.memory:
            self.memory[key] = []

    def calculate_reward(self, pnl, action, state_info, environment_info):
        """
        Calculates the reward based on PnL and synthesized strategy elements.

        Args:
            pnl (float): Profit and loss since the last step.
            action (int): Action taken (e.g., 0:Hold, 1:Buy, 2:Sell).
            state_info (dict): Outputs from the EnhancedTradingModel for the current state.
            environment_info (dict): Additional info from the environment/simulator.
                                     Expected keys: 'is_trade_closed', 'hit_stop_loss',
                                     'steps_in_trade', 'current_price', 'entry_price'.

        Returns:
            float: The calculated reward.
        """
        cfg = self.reward_config
        reward = 0.0

        # --- 1. Core PnL Reward (with asymmetry) ---
        if pnl > 0:
            reward += pnl * cfg['pnl_scale'] * cfg['profit_bonus_factor']
        else:
            # Penalize losses more heavily
            reward += pnl * cfg['pnl_scale'] * cfg['loss_penalty_factor']

        # --- 2. Risk Management Penalties ---
        if environment_info.get('hit_stop_loss', False):
            reward += cfg['max_loss_penalty'] # Significant penalty for hitting SL

        # Penalize based on model's uncertainty and volatility prediction
        reward -= state_info.get('uncertainty', 0) * cfg['uncertainty_penalty_factor']
        reward -= state_info.get('volatility', 0) * cfg['high_volatility_penalty_factor']

        # --- 3. Regime-Aware Rewards/Penalties ---
        regime_probs = torch.softmax(torch.tensor(state_info.get('regime_logits', [0,0,0])), dim=-1)
        predicted_regime = torch.argmax(regime_probs).item() # 0:Trend, 1:MeanRev, 2:Uncertain

        # Assuming action 1=Buy, 2=Sell, 0=Hold
        is_buy = action == 1
        is_sell = action == 2
        is_aggressive = is_buy or is_sell

        # Get simple trend direction (e.g., from a moving average feature if available,
        # or potentially infer from model if trained for it - simplified here)
        # Placeholder: Assume a simple trend indicator is available or calculated
        trend_direction = environment_info.get('trend_direction', 0) # 1=Up, -1=Down, 0=Neutral

        if predicted_regime == 0: # Trending
            if (is_buy and trend_direction == 1) or (is_sell and trend_direction == -1):
                reward += cfg['trend_follow_bonus']
            elif (is_buy and trend_direction == -1) or (is_sell and trend_direction == 1):
                reward += cfg['trend_against_penalty']
        elif predicted_regime == 1: # Mean Reverting
             # Reward fading moves (selling highs, buying lows - needs price context)
             # Simplified: Reward acting against recent trend
            if (is_buy and trend_direction == -1) or (is_sell and trend_direction == 1):
                 reward += cfg['mean_revert_bonus']
            elif (is_buy and trend_direction == 1) or (is_sell and trend_direction == -1):
                 reward += cfg['mean_revert_against_penalty']
        elif predicted_regime == 2: # Uncertain Regime
            if is_aggressive:
                reward += cfg['uncertain_regime_penalty']

        # --- 4. Trade Duration & Efficiency ---
        if environment_info.get('is_in_trade', False):
            reward += cfg['holding_penalty_factor'] * environment_info.get('steps_in_trade', 0)

        if environment_info.get('is_trade_closed', False) and pnl > 0:
            reward += cfg['profit_capture_bonus']
            # Optional: Bonus for hitting profit target faster than model predicted
            # horizon = state_info.get('trade_horizon_suggestion', 100)
            # steps = environment_info.get('steps_in_trade', 100)
            # if steps < horizon:
            #     reward += cfg['quick_profit_bonus_factor'] * (1 - steps/horizon)

        # --- 5. Consistency (Proxy using recent reward volatility) ---
        if len(self.recent_rewards) > 10: # Need enough samples
             reward_std = np.std(list(self.recent_rewards))
             # Penalize high reward volatility (encourage smooth equity curve)
             # Avoid division by zero
             if reward_std > 1e-6:
                  # Simple Sharpe proxy: reward mean / reward std
                  # We use the inverse of std dev as a bonus factor if positive mean reward
                  mean_reward = np.mean(list(self.recent_rewards))
                  if mean_reward > 0:
                       reward += (mean_reward / reward_std) * cfg['sharpe_ratio_bonus_factor']


        # --- 6. Catastrophic Event Handling ---
        # Check if the current action matches an action that recently led to a catastrophe
        for past_reward, past_action, past_state_info in self.catastrophic_memory:
             if action == past_action and is_aggressive: # Penalize repeating aggressive actions
                  # Optional: Add more complex check (e.g., if state is similar)
                  reward += cfg['catastrophic_repeat_penalty']
                  break # Apply penalty only once

        # --- 7. Action Penalties ---
        if is_aggressive:
            reward += cfg['transaction_cost_penalty']

        # --- Final Reward Clipping (Optional) ---
        # reward = np.clip(reward, -100, 100) # Clip rewards to a range

        return reward


    def save_model(self, path):
        """Saves the model state."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cpu'):
        """Loads the model state."""
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device) # Ensure model is on the correct device
        print(f"Model loaded from {path} to device {device}")

# --- Helper Function ---
def create_strategy(model, config):
    """
    Factory function to create the PPO agent with synthesized strategy logic.

    Args:
        model (nn.Module): The trading model (EnhancedTradingModel).
        config (dict): Configuration dictionary for the agent and reward shaping.

    Returns:
        PPOAgent: The configured PPO agent.
    """
    # Ensure reward config is nested correctly if passed within main config
    agent_config = config.get('agent_config', {})
    reward_config = config.get('reward_config', {})
    # Merge if reward_config is provided separately at top level
    agent_config['reward_config'] = {**agent_config.get('reward_config', {}), **reward_config}


    return PPOAgent(
        model=model,
        learning_rate=agent_config.get('learning_rate', 1e-4),
        gamma=agent_config.get('gamma', 0.99),
        lambda_gae=agent_config.get('lambda_gae', 0.95),
        ppo_epsilon=agent_config.get('ppo_epsilon', 0.2),
        ppo_epochs=agent_config.get('ppo_epochs', 10),
        batch_size=agent_config.get('batch_size', 64),
        entropy_coef=agent_config.get('entropy_coef', 0.01),
        value_loss_coef=agent_config.get('value_loss_coef', 0.5),
        config=agent_config # Pass the combined agent configuration
    )
