# core/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging # Import logging

logger = logging.getLogger("genovo_traderv2") # Get logger

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network Block with causal convolutions,
    weight normalization, ReLU activation, and dropout.
    Includes residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.output_relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        out = self.conv1(x)
        out = out[..., :x.size(-1)]
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out[..., :x.size(-1)]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)
        return self.output_relu(out + res)


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention mechanism."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
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
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
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

        logger.info(f"Initializing EnhancedTradingModel with num_features = {num_features}")
        if not isinstance(num_features, int) or num_features <= 0:
             raise ValueError(f"num_features must be a positive integer, got: {num_features}")

        self.hidden_size = hidden_size
        self.num_features = num_features # Store num_features

        # --- Input Layer ---
        # --- !! FIX HERE: Ensure LayerNorm uses num_features !! ---
        self.layer_norm_input = nn.LayerNorm(self.num_features)
        tcn_input_channels = self.num_features # Input to TCN is original features

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
        # (Output heads remain the same)
        self.policy_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 3))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.position_size_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Softplus())
        self.stop_loss_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())
        self.take_profit_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())
        self.trade_horizon_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Softplus())
        self.regime_detection_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 3))
        self.volatility_prediction_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Softplus())


    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]
        if x.shape[-1] != self.num_features:
             logger.error(f"Model forward pass input feature dimension mismatch! Expected {self.num_features}, got {x.shape[-1]}")
             # Handle error appropriately, e.g., raise ValueError
             raise ValueError(f"Input feature dimension mismatch: Expected {self.num_features}, got {x.shape[-1]}")


        # 1. Input Layer Normalization
        try:
            # --- Ensure this uses the correct dimension ---
            x = self.layer_norm_input(x)
        except Exception as e:
             logger.error(f"Error in layer_norm_input: {e}. Input shape: {x.shape}", exc_info=True)
             raise

        # 2. TCN Layers
        x = x.transpose(1, 2)
        for i, tcn_layer in enumerate(self.tcn_layers):
            try: x = tcn_layer(x)
            except Exception as e: logger.error(f"Error in TCN layer {i}: {e}. Input shape: {x.shape}", exc_info=True); raise
        x = x.transpose(1, 2)

        # 3. GRU Layer
        try: gru_output, hidden = self.gru(x)
        except Exception as e: logger.error(f"Error in GRU layer: {e}. Input shape: {x.shape}", exc_info=True); raise

        # 4. Transformer Encoder Layers
        transformer_output = gru_output
        for i, transformer_layer in enumerate(self.transformer_layers):
            try: transformer_output = transformer_layer(transformer_output)
            except Exception as e: logger.error(f"Error in Transformer layer {i}: {e}. Input shape: {transformer_output.shape}", exc_info=True); raise
        try: transformer_output = self.final_norm(transformer_output)
        except Exception as e: logger.error(f"Error in final_norm: {e}. Input shape: {transformer_output.shape}", exc_info=True); raise

        # 5. Extract Last Time Step Output for Heads
        last_step_output = transformer_output[:, -1, :]

        # 6. Calculate Outputs from Heads
        try:
            policy_logits = self.policy_head(last_step_output)
            value = self.value_head(last_step_output)
            position_size = self.position_size_head(last_step_output)
            uncertainty = self.uncertainty_head(last_step_output)
            stop_loss = self.stop_loss_head(last_step_output)
            take_profit = self.take_profit_head(last_step_output)
            trade_horizon = self.trade_horizon_head(last_step_output)
            regime_logits = self.regime_detection_head(last_step_output)
            volatility = self.volatility_prediction_head(last_step_output)
        except Exception as e:
             logger.error(f"Error calculating output heads: {e}. Input shape: {last_step_output.shape}", exc_info=True)
             raise

        # Return all outputs in a dictionary
        return {
            'policy_logits': policy_logits, 'value': value, 'position_size': position_size,
            'uncertainty': uncertainty, 'stop_loss': stop_loss, 'take_profit': take_profit,
            'trade_horizon': trade_horizon, 'regime_logits': regime_logits, 'volatility': volatility,
            'gru_last_hidden': hidden[-1], 'transformer_output': transformer_output
        }


# --- Factory Function ---
def create_model(config):
    """
    Factory function to create the EnhancedTradingModel based on configuration.
    Ensures 'num_features' is correctly passed.
    """
    model_config = config # Expect model_config directly
    num_features = model_config.get('num_features')
    if num_features is None:
         logger.error("num_features is missing in model_config during model creation.")
         raise ValueError("num_features is missing in model_config.")
    if not isinstance(num_features, int) or num_features <= 0:
        raise ValueError(f"Invalid num_features in model_config: {num_features}")


    return EnhancedTradingModel(
        num_features=num_features, # Use the validated num_features
        seq_length=model_config.get('seq_length', 201),
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        tcn_channels=model_config.get('tcn_channels', [64, 128, 128]),
        tcn_kernel_size=model_config.get('tcn_kernel_size', 3),
        num_heads=model_config.get('num_heads', 4),
        num_transformer_layers=model_config.get('num_transformer_layers', 2),
        dropout=model_config.get('dropout', 0.2)
    )
