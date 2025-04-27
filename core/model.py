# core/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network Block with causal convolutions,
    weight normalization, ReLU activation, and dropout.
    Includes residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        # Calculate padding for causal convolution (output has same length as input)
        padding = (kernel_size - 1) * dilation

        # First convolutional layer + normalization + activation + dropout
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # LayerNorm applied channel-wise after transposition
        self.norm1 = nn.LayerNorm(out_channels)

        # Second convolutional layer + normalization + activation + dropout
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(out_channels)

        # Residual connection handling: if input/output channels differ, use a 1x1 conv
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.output_relu = nn.ReLU() # Final activation for the block output

    def forward(self, x):
        # x shape: [batch_size, in_channels, seq_len]
        res = self.residual(x) # Calculate residual connection path

        # First conv block
        out = self.conv1(x)
        out = out[..., :x.size(-1)] # Trim padding to maintain sequence length
        # Apply LayerNorm on the feature dimension (transpose needed)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = out[..., :x.size(-1)] # Trim padding
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Add residual and apply final activation
        return self.output_relu(out + res)


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention mechanism."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Query, Key, Value, and final output projection
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]
        batch_size = x.size(0)

        # Project and reshape for multi-head attention
        # q, k, v shape: [batch_size, num_heads, seq_len, head_dim]
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores (scaled dot-product)
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask if provided (for preventing attention to future positions)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # context shape: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, v)

        # Concatenate heads and apply final linear layer
        # context shape: [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)

        return output


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer Encoder Block."""
    def __init__(self, d_model, num_heads, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]
        # Self-attention sublayer
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output) # Add & Norm (residual connection)
        x = self.norm1(x)

        # Feed-forward sublayer
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output) # Add & Norm (residual connection)
        x = self.norm2(x)

        return x


class EnhancedTradingModel(nn.Module):
    """
    Combined TCN + GRU + Attention + Transformer model for HFT.
    Includes multiple output heads for sophisticated trading decisions.
    """
    def __init__(self,
                 num_features,
                 seq_length=100, # Used mainly for positional encoding if added
                 hidden_size=128, # Dimension for GRU, Attention, Transformer
                 num_layers=2, # Number of GRU layers
                 tcn_channels=[64, 128, 128], # Output channels for TCN blocks
                 tcn_kernel_size=3,
                 num_heads=4, # Number of attention heads
                 num_transformer_layers=2, # Number of Transformer encoder layers
                 dropout=0.2):
        super(EnhancedTradingModel, self).__init__()

        self.hidden_size = hidden_size

        # --- Input Layer ---
        # Optional: Linear layer to project input features to hidden_size
        # self.input_proj = nn.Linear(num_features, hidden_size)
        # self.layer_norm_input = nn.LayerNorm(hidden_size)
        # Or directly use LayerNorm on input features if TCN handles projection
        self.layer_norm_input = nn.LayerNorm(num_features)
        tcn_input_channels = num_features # Input to TCN is original features

        # --- Temporal Convolutional Network (TCN) ---
        self.tcn_layers = nn.ModuleList()
        in_channels = tcn_input_channels
        tcn_output_channels = 0
        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i # Exponentially increasing dilation
            self.tcn_layers.append(TCNBlock(in_channels, out_channels, tcn_kernel_size, dilation, dropout))
            in_channels = out_channels
        tcn_output_channels = in_channels # Output channels from the last TCN block

        # --- GRU Layer ---
        # GRU input size should match TCN output channels
        self.gru = nn.GRU(tcn_output_channels, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=False) # Keep unidirectional for causality

        # --- Transformer Encoder Layers ---
        # Applied after GRU to capture deeper context over the sequence
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size) # LayerNorm after transformers

        # --- Output Heads ---
        # Use the output of the *last* time step from the final layer for predictions

        # Policy head (Action prediction: Buy, Sell, Hold)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # Logits for 3 actions
        )

        # Value head (Critic's state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Position size suggestion head (e.g., fraction of max capital)
        self.position_size_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Uncertainty prediction head (e.g., predicting variance or confidence)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure output is positive
        )

        # Stop loss suggestion head (e.g., percentage distance from entry)
        self.stop_loss_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output between 0 and 1 (representing fraction, e.g., 0.01 = 1%)
        )

        # Take profit suggestion head (e.g., percentage distance from entry)
        self.take_profit_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output between 0 and 1 (e.g., 0.02 = 2%)
        )

        # Trade horizon prediction head (e.g., number of steps to hold)
        self.trade_horizon_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus() # Ensure positive output
        )

        # Market regime detection head (e.g., Trending, Mean-Reverting, Uncertain)
        self.regime_detection_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # Logits for 3 regimes
        )

        # Volatility prediction head (e.g., predicted ATR or std dev)
        self.volatility_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus() # Ensure positive output
        )


    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]

        # 1. Input Layer Normalization
        x = self.layer_norm_input(x)

        # 2. TCN Layers
        # Reshape for Conv1d: [batch_size, num_features, seq_len]
        x = x.transpose(1, 2)
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        # Reshape back: [batch_size, seq_len, tcn_output_channels]
        x = x.transpose(1, 2)

        # 3. GRU Layer
        # gru_output shape: [batch_size, seq_len, hidden_size]
        # hidden shape: [num_layers, batch_size, hidden_size]
        gru_output, hidden = self.gru(x)

        # 4. Transformer Encoder Layers
        transformer_output = gru_output # Input to transformer is GRU output sequence
        for transformer_layer in self.transformer_layers:
            # Masking usually not needed for encoder-only structure on full sequence
            transformer_output = transformer_layer(transformer_output)
        transformer_output = self.final_norm(transformer_output)

        # 5. Extract Last Time Step Output for Heads
        # Use the output from the transformer's last time step
        last_step_output = transformer_output[:, -1, :] # Shape: [batch_size, hidden_size]

        # 6. Calculate Outputs from Heads
        policy_logits = self.policy_head(last_step_output)
        value = self.value_head(last_step_output)
        position_size = self.position_size_head(last_step_output)
        uncertainty = self.uncertainty_head(last_step_output)
        stop_loss = self.stop_loss_head(last_step_output)
        take_profit = self.take_profit_head(last_step_output)
        trade_horizon = self.trade_horizon_head(last_step_output)
        regime_logits = self.regime_detection_head(last_step_output)
        volatility = self.volatility_prediction_head(last_step_output)

        # Return all outputs in a dictionary
        return {
            'policy_logits': policy_logits,
            'value': value,
            'position_size': position_size,
            'uncertainty': uncertainty,
            'stop_loss': stop_loss,       # Predicted SL percentage (0 to 1)
            'take_profit': take_profit,   # Predicted TP percentage (0 to 1)
            'trade_horizon': trade_horizon, # Predicted holding steps
            'regime_logits': regime_logits, # Logits for market regimes
            'volatility': volatility,     # Predicted volatility measure
            # Include intermediate outputs if needed for analysis/debugging
            'gru_last_hidden': hidden[-1], # Last hidden state from GRU (last layer)
            'transformer_output': transformer_output # Full output sequence from transformer
        }


# --- Factory Function ---
def create_model(config):
    """
    Factory function to create the EnhancedTradingModel based on configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.
                       Expected keys: 'num_features', 'seq_length', 'hidden_size',
                       'num_layers', 'tcn_channels', 'tcn_kernel_size',
                       'num_heads', 'num_transformer_layers', 'dropout'.

    Returns:
        EnhancedTradingModel: The instantiated trading model.
    """
    model_config = config.get('model_config', {}) # Expect nested config
    return EnhancedTradingModel(
        num_features=model_config.get('num_features', 50), # Default based on feature config
        seq_length=model_config.get('seq_length', 100),
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        tcn_channels=model_config.get('tcn_channels', [64, 128, 128]),
        tcn_kernel_size=model_config.get('tcn_kernel_size', 3),
        num_heads=model_config.get('num_heads', 4),
        num_transformer_layers=model_config.get('num_transformer_layers', 2),
        dropout=model_config.get('dropout', 0.2)
    )

