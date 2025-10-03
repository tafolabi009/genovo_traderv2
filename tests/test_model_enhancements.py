# tests/test_model_enhancements.py

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_enhancements import (
    AdaptiveFourierFeatures,
    MultiScaleAttention,
    AdaptiveRiskModule,
    QuantileHead
)


def test_fourier_features():
    """Test Adaptive Fourier Features"""
    batch_size, seq_len, features = 4, 50, 10
    x = torch.randn(batch_size, seq_len, features)
    
    fourier = AdaptiveFourierFeatures(features, fourier_dim=32)
    output = fourier(x)
    
    assert output.shape == (batch_size, seq_len, 64)  # 32 * 2 (sin + cos)


def test_multi_scale_attention():
    """Test Multi-Scale Attention"""
    batch_size, seq_len, d_model = 4, 100, 256
    x = torch.randn(batch_size, seq_len, d_model)
    
    msa = MultiScaleAttention(d_model, num_heads=8, scales=[1, 2, 4])
    output = msa(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_risk_module():
    """Test Adaptive Risk Module"""
    batch_size, seq_len, d_model = 4, 50, 256
    x = torch.randn(batch_size, seq_len, d_model)
    
    risk_module = AdaptiveRiskModule(d_model, num_regimes=5)
    risk, regime_probs = risk_module(x)
    
    assert risk.shape == (batch_size, 3)  # volatility, skewness, kurtosis
    assert regime_probs.shape == (batch_size, 5)
    assert torch.allclose(regime_probs.sum(dim=1), torch.ones(batch_size))


def test_quantile_head():
    """Test Quantile Head"""
    batch_size, d_model = 4, 256
    x = torch.randn(batch_size, d_model)
    
    quantile_head = QuantileHead(d_model, quantiles=[0.1, 0.5, 0.9])
    output = quantile_head(x)
    
    assert output.shape == (batch_size, 3)


def test_quantile_loss():
    """Test quantile loss calculation"""
    quantile_head = QuantileHead(256, quantiles=[0.1, 0.5, 0.9])
    
    predictions = torch.randn(4, 3)
    targets = torch.randn(4)
    
    loss = quantile_head.quantile_loss(predictions, targets)
    assert loss.item() >= 0  # Loss should be non-negative


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
