# core/model_enhancements.py

"""
Advanced model enhancements for HFT trading
Implements cutting-edge techniques from research and industry best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AdaptiveFourierFeatures(nn.Module):
    """Learnable Fourier features for capturing cyclical patterns in financial data"""
    
    def __init__(self, input_dim, fourier_dim=64, scale=1.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        # Learnable frequency components
        self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * scale)
        
    def forward(self, x):
        # x: [batch, seq, features]
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MultiScaleAttention(nn.Module):
    """Multi-scale attention for different time horizons"""
    
    def __init__(self, d_model, num_heads, scales=[1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads // len(scales), batch_first=True)
            for _ in scales
        ])
        self.combine = nn.Linear(d_model * len(scales), d_model)
        
    def forward(self, x):
        outputs = []
        for scale, attn in zip(self.scales, self.attentions):
            # Downsample by scale
            if scale > 1:
                x_scaled = F.adaptive_avg_pool1d(
                    x.transpose(1, 2), 
                    x.size(1) // scale
                ).transpose(1, 2)
            else:
                x_scaled = x
            
            attn_out, _ = attn(x_scaled, x_scaled, x_scaled)
            
            # Upsample back
            if scale > 1:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2),
                    size=x.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            outputs.append(attn_out)
        
        combined = torch.cat(outputs, dim=-1)
        return self.combine(combined)


class AdaptiveRiskModule(nn.Module):
    """Adaptive risk assessment based on market regime"""
    
    def __init__(self, d_model, num_regimes=5):
        super().__init__()
        self.num_regimes = num_regimes
        
        # Regime classification
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_regimes)
        )
        
        # Risk parameters per regime
        self.risk_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 3),  # volatility, skewness, kurtosis
                nn.Softplus()
            )
            for _ in range(num_regimes)
        ])
        
    def forward(self, x):
        # x: [batch, seq, features]
        x_pooled = x.mean(dim=1)  # [batch, features]
        
        # Classify regime
        regime_logits = self.regime_classifier(x_pooled)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Estimate risk for each regime
        risk_estimates = torch.stack([
            est(x_pooled) for est in self.risk_estimators
        ], dim=1)  # [batch, num_regimes, 3]
        
        # Weighted combination based on regime probabilities
        risk = torch.einsum('bn,bnr->br', regime_probs, risk_estimates)
        
        return risk, regime_probs


class MarketMicrostructureModule(nn.Module):
    """Models market microstructure effects for HFT"""
    
    def __init__(self, d_model):
        super().__init__()
        
        # Order flow imbalance modeling
        self.ofi_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Spread dynamics
        self.spread_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Market impact
        self.impact_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [batch, seq, features]
        x_pooled = x[:, -1, :]  # Use last timestep
        
        ofi = self.ofi_encoder(x_pooled)
        spread = self.spread_encoder(x_pooled)
        
        micro_features = torch.cat([ofi, spread], dim=-1)
        impact = self.impact_predictor(micro_features)
        
        return micro_features, impact


class QuantileHead(nn.Module):
    """Quantile regression head for uncertainty-aware predictions"""
    
    def __init__(self, d_model, quantiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, self.num_quantiles)
        )
        
    def forward(self, x):
        return self.head(x)
    
    def quantile_loss(self, predictions, targets):
        """Pinball loss for quantile regression"""
        targets = targets.unsqueeze(-1).expand_as(predictions)
        errors = targets - predictions
        
        quantiles = torch.tensor(self.quantiles, device=predictions.device)
        loss = torch.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        )
        return loss.mean()


class EnhancedEmbedding(nn.Module):
    """Enhanced embedding with multiple encoding schemes"""
    
    def __init__(self, num_features, d_model, seq_length, dropout=0.1):
        super().__init__()
        
        # Feature embedding
        self.feature_embed = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        
        # Time-of-day encoding (for intraday patterns)
        self.time_embed = nn.Embedding(288, d_model)  # 5-minute intervals in a day
        
        # Day-of-week encoding
        self.dow_embed = nn.Embedding(5, d_model)  # Trading days
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, time_idx=None, dow_idx=None):
        # x: [batch, seq, features]
        embedded = self.feature_embed(x)
        embedded = embedded + self.pos_encoding[:, :x.size(1), :]
        
        if time_idx is not None:
            embedded = embedded + self.time_embed(time_idx)
        
        if dow_idx is not None:
            embedded = embedded + self.dow_embed(dow_idx)
        
        return self.dropout(self.layer_norm(embedded))


class EnsembleHead(nn.Module):
    """Ensemble multiple prediction heads for robustness"""
    
    def __init__(self, d_model, num_actions, num_heads=5):
        super().__init__()
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, num_actions)
            )
            for _ in range(num_heads)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
    def forward(self, x):
        # Get predictions from all heads
        predictions = torch.stack([head(x) for head in self.heads], dim=0)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        ensemble_pred = (predictions * weights).sum(dim=0)
        
        # Also return variance across heads for uncertainty
        uncertainty = predictions.var(dim=0)
        
        return ensemble_pred, uncertainty


def create_enhanced_model_components(config):
    """Factory function to create enhanced model components"""
    d_model = config.get('hidden_size', 768)
    num_features = config.get('num_features', 512)
    seq_length = config.get('seq_length', 201)
    
    components = {
        'fourier_features': AdaptiveFourierFeatures(num_features, fourier_dim=64),
        'multi_scale_attention': MultiScaleAttention(d_model, num_heads=8),
        'risk_module': AdaptiveRiskModule(d_model, num_regimes=5),
        'microstructure': MarketMicrostructureModule(d_model),
        'embedding': EnhancedEmbedding(num_features, d_model, seq_length),
        'quantile_head': QuantileHead(d_model),
        'ensemble_head': EnsembleHead(d_model, num_actions=3)
    }
    
    return components
