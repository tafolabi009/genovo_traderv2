# core/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging # Import logging
import math
from torch.nn import LayerNorm
from einops import rearrange, repeat

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


class PositionalEncoding(nn.Module):
    """Improved positional encoding with learnable parameters"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # Learnable scaling factor
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for better feature processing"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 3, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context=None):
        residual = x
        x = self.layer_norm(x)
        parallel_1 = self.fc1(x)
        parallel_2 = self.fc2(x)
        if context is not None:
            gate_input = torch.cat([parallel_1, parallel_2, context], dim=-1)
        else:
            gate_input = torch.cat([parallel_1, parallel_2, x], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return residual + self.dropout(gate * parallel_1 + (1 - gate) * parallel_2)


class VariableSelectionNetwork(nn.Module):
    """Network for dynamic variable selection"""
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        
        self.flattened_grn = GatedResidualNetwork(hidden_size)
        self.grn_vec = nn.ModuleList([GatedResidualNetwork(hidden_size) for _ in range(num_features)])
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, num_features, hidden_size]
        flat_x = x.reshape(-1, self.num_features, self.hidden_size)
        
        # Process each feature independently
        processed_features = []
        for i, grn in enumerate(self.grn_vec):
            processed_features.append(grn(flat_x[:, i:i+1, :]))
        processed_features = torch.cat(processed_features, dim=1)
        
        # Compute attention weights
        combined = torch.mean(processed_features, dim=1, keepdim=True)
        weights = self.softmax(combined)
        
        # Weight features
        weighted_features = weights * processed_features
        return weighted_features.reshape(x.shape)


class EnhancedTradingModel(nn.Module):
    """Enhanced Trading Model with advanced architecture"""
    def __init__(self,
                 num_features,
                 seq_length=100,
                 hidden_size=256,  # Increased from 128
                 num_layers=3,     # Increased from 2
                 tcn_channels=[128, 256, 256],  # Increased channel sizes
                 tcn_kernel_size=5,  # Increased from 3
                 num_heads=8,      # Increased from 4
                 num_transformer_layers=4,  # Increased from 2
                 dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_features = num_features
        
        # Input embedding and normalization
        self.feature_projection = nn.Linear(num_features, hidden_size)
        self.layer_norm_input = nn.LayerNorm(hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, seq_length)
        
        # Variable selection network
        self.variable_selection = VariableSelectionNetwork(num_features, hidden_size)
        
        # Enhanced Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu'  # Changed from ReLU
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_transformer_layers)
        
        # Temporal Convolution layers with skip connections
        self.tcn_layers = nn.ModuleList()
        in_channels = hidden_size
        for out_channels in tcn_channels:
            self.tcn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, tcn_kernel_size, padding=tcn_kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # GRU with bidirectional processing
        self.gru = nn.GRU(
            tcn_channels[-1],
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output heads with enhanced architecture
        output_size = hidden_size * 2  # Due to bidirectional GRU
        
        self.policy_head = nn.Sequential(
            GatedResidualNetwork(output_size),
            nn.Linear(output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        self.value_head = nn.Sequential(
            GatedResidualNetwork(output_size),
            nn.Linear(output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.position_size_head = nn.Sequential(
            GatedResidualNetwork(output_size),
            nn.Linear(output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_head = nn.Sequential(
            GatedResidualNetwork(output_size),
            nn.Linear(output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        # Additional specialized heads
        self.market_regime_head = nn.Sequential(
            GatedResidualNetwork(output_size),
            nn.Linear(output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)  # Multiple market regime categories
        )
        
        self.risk_assessment_head = nn.Sequential(
            GatedResidualNetwork(output_size),
            nn.Linear(output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # Risk levels (low, medium, high)
        )

    def forward(self, x):
        # Input shape: [batch_size, seq_len, num_features]
        batch_size, seq_len, _ = x.shape
        
        # Project features and apply positional encoding
        x = self.feature_projection(x)
        x = self.layer_norm_input(x)
        x = self.positional_encoding(x)
        
        # Apply variable selection
        x = x.reshape(batch_size, seq_len, self.num_features, -1)
        x = self.variable_selection(x)
        x = x.reshape(batch_size, seq_len, -1)
        
        # Transformer processing
        transformer_output = self.transformer_encoder(x)
        
        # TCN processing
        tcn_input = transformer_output.transpose(1, 2)
        for tcn_layer in self.tcn_layers:
            tcn_input = tcn_layer(tcn_input) + tcn_input  # Skip connection
        tcn_output = tcn_input.transpose(1, 2)
        
        # GRU processing
        gru_output, hidden = self.gru(tcn_output)
        
        # Get final output (concatenate forward and backward states)
        final_output = torch.cat([gru_output[:, -1, :self.hidden_size],
                                gru_output[:, 0, self.hidden_size:]], dim=1)
        
        # Calculate outputs from all heads
        return {
            'policy_logits': self.policy_head(final_output),
            'value': self.value_head(final_output),
            'position_size': self.position_size_head(final_output),
            'uncertainty': self.uncertainty_head(final_output),
            'market_regime': self.market_regime_head(final_output),
            'risk_assessment': self.risk_assessment_head(final_output),
            'gru_last_hidden': hidden,
            'transformer_output': transformer_output
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
        hidden_size=model_config.get('hidden_size', 256),
        num_layers=model_config.get('num_layers', 3),
        tcn_channels=model_config.get('tcn_channels', [128, 256, 256]),
        tcn_kernel_size=model_config.get('tcn_kernel_size', 5),
        num_heads=model_config.get('num_heads', 8),
        num_transformer_layers=model_config.get('num_transformer_layers', 4),
        dropout=model_config.get('dropout', 0.2)
    )

class SelfAttentionPooling(nn.Module):
    """Advanced self-attention pooling with learnable temperature"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights / self.temperature, dim=1)
        attended = torch.sum(x * attn_weights, dim=1)
        return attended, attn_weights

class MultiScaleConvolution(nn.Module):
    """Multi-scale convolution block for capturing patterns at different frequencies"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7, 15, 31]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), k, padding=k//2)
            for k in kernel_sizes
        ])
        self.norm = LayerNorm(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: [batch, channels, time]
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        x = torch.cat(outputs, dim=1)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return self.activation(x)

class AdaptiveAttention(nn.Module):
    """Attention mechanism with adaptive temperature and sparse attention"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, sparsity=0.9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.sparsity = sparsity
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable temperature per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with learnable temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores * self.temperature
        
        # Apply sparse attention
        if self.training:
            mask = torch.rand_like(scores) > self.sparsity
            scores = scores.masked_fill(~mask, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(out)

class QuantileTransformer(nn.Module):
    """Transformer block with quantile regression heads"""
    def __init__(self, hidden_size, num_heads, dropout=0.1, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.attention = AdaptiveAttention(hidden_size, num_heads, dropout)
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.quantiles = quantiles
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in quantiles
        ])
        
    def forward(self, x):
        # Self-attention with residual
        attended = self.attention(x)
        x = self.norm1(x + attended)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Quantile predictions
        quantile_preds = [head(x) for head in self.quantile_heads]
        return x, torch.cat(quantile_preds, dim=-1)

class UltraHighFrequencyModel(nn.Module):
    """State-of-the-art model for high-frequency trading with enhanced risk management"""
    def __init__(self,
                 num_features,
                 seq_length=100,
                 hidden_size=768,  # Increased from 512
                 num_heads=32,     # Doubled from 16
                 num_layers=8,     # Increased from 6
                 dropout=0.1,
                 quantiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],  # More granular
                 use_regime_detection=True,
                 use_neural_flow=True,
                 use_hierarchical_attention=True):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        
        # Advanced embedding with frequency and regime encoding
        self.freq_encodings = nn.Parameter(torch.randn(1, seq_length, hidden_size // 2))
        self.regime_encodings = nn.Parameter(torch.randn(5, hidden_size // 2))  # 5 regime types
        self.feature_proj = nn.Linear(num_features, hidden_size // 2)
        
        # Hierarchical attention for multi-timeframe analysis
        if use_hierarchical_attention:
            self.hierarchical_attention = nn.ModuleList([
                AdaptiveAttention(hidden_size, num_heads // 2, dropout)
                for _ in range(3)  # 3 timeframe levels
            ])
        
        # Neural flow model for price dynamics
        if use_neural_flow:
            self.flow_model = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size)
                ) for _ in range(3)
            ])
        
        # Multi-scale convolution with quantum-inspired layers
        self.temporal_conv = nn.ModuleList([
            MultiScaleConvolution(hidden_size, hidden_size, 
                kernel_sizes=[3, 7, 15, 31, 63])  # Added larger kernel
            for _ in range(4)  # Increased from 3
        ])
        
        # Enhanced transformer layers with regime detection
        self.transformer_layers = nn.ModuleList([
            QuantileTransformer(hidden_size, num_heads, dropout, quantiles)
            for _ in range(num_layers)
        ])
        
        if use_regime_detection:
            self.regime_detector = nn.ModuleList([
                TransformerEncoderBlock(hidden_size, num_heads)
                for _ in range(2)
            ])
        
        # Advanced pooling with learnable temperature
        self.global_pool = SelfAttentionPooling(hidden_size)
        
        # Enhanced output heads
        self.policy_head = nn.Sequential(
            GatedResidualNetwork(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)
        )
        
        # Value prediction with uncertainty
        self.value_head = nn.Sequential(
            GatedResidualNetwork(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(quantiles))
        )
        
        # Enhanced risk management heads
        self.risk_heads = nn.ModuleDict({
            'var': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            ),
            'expected_shortfall': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            ),
            'drawdown_risk': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            ),
            'liquidity_risk': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            ),
            'correlation_risk': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, hidden_size)
            )
        })
        
        # Market impact and execution risk
        self.execution_heads = nn.ModuleDict({
            'market_impact': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            ),
            'execution_risk': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            ),
            'spread_risk': nn.Sequential(
                GatedResidualNetwork(hidden_size),
                nn.Linear(hidden_size, len(quantiles))
            )
        })
        
        # Adaptive position sizing
        self.position_size_head = nn.Sequential(
            GatedResidualNetwork(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Enhanced feature embedding
        x = self.feature_proj(x)
        freq_enc = repeat(self.freq_encodings, '1 n d -> b n d', b=batch_size)
        x = torch.cat([x, freq_enc], dim=-1)
        
        # Hierarchical attention processing
        if hasattr(self, 'hierarchical_attention'):
            for attention in self.hierarchical_attention:
                x = x + attention(x)
        
        # Neural flow processing
        if hasattr(self, 'flow_model'):
            for flow in self.flow_model:
                x = x + flow(x)
        
        # Enhanced temporal convolution
        x = x.transpose(1, 2)
        for conv in self.temporal_conv:
            x = x + conv(x)  # Residual connection
        x = x.transpose(1, 2)
        
        # Regime detection and transformer processing
        if hasattr(self, 'regime_detector'):
            regime_features = x
            for regime_layer in self.regime_detector:
                regime_features = regime_layer(regime_features)
            regime_logits = regime_features.mean(dim=1)
            regime_probs = F.softmax(regime_logits, dim=-1)
            regime_embedding = torch.matmul(regime_probs, self.regime_encodings)
            x = x + regime_embedding.unsqueeze(1)
        
        # Quantile transformer processing
        all_quantiles = []
        for transformer in self.transformer_layers:
            x, quantile_preds = transformer(x)
            all_quantiles.append(quantile_preds)
        
        # Advanced pooling
        pooled, attention_weights = self.global_pool(x)
        
        # Calculate all outputs
        risk_metrics = {name: head(pooled) for name, head in self.risk_heads.items()}
        execution_metrics = {name: head(pooled) for name, head in self.execution_heads.items()}
        
        return {
            'policy_logits': self.policy_head(pooled),
            'value_quantiles': self.value_head(pooled),
            'position_size': self.position_size_head(pooled),
            'risk_metrics': risk_metrics,
            'execution_metrics': execution_metrics,
            'regime_probabilities': regime_probs if hasattr(self, 'regime_detector') else None,
            'attention_weights': attention_weights,
            'quantile_predictions': torch.stack(all_quantiles, dim=1),
            'feature_importance': attention_weights.mean(dim=0)
        }

    def _init_weights(self, module):
        """Enhanced weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

def create_model(config):
    """Factory function to create the UltraHighFrequencyModel"""
    num_features = config.get('num_features')
    if num_features is None:
        raise ValueError("num_features is required in config")
        
    return UltraHighFrequencyModel(
        num_features=num_features,
        seq_length=config.get('seq_length', 201),
        hidden_size=config.get('hidden_size', 768),
        num_heads=config.get('num_heads', 32),
        num_layers=config.get('num_layers', 8),
        dropout=config.get('dropout', 0.1),
        quantiles=config.get('quantiles', [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
        use_regime_detection=config.get('use_regime_detection', True),
        use_neural_flow=config.get('use_neural_flow', True),
        use_hierarchical_attention=config.get('use_hierarchical_attention', True)
    )
