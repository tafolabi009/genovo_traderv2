# tests/test_broker_factory.py

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.broker_factory import BrokerFactory
from broker.base_broker import BaseBroker


def test_broker_factory_registration():
    """Test broker registration"""
    # Should have at least mock broker registered
    brokers = BrokerFactory.list_brokers()
    assert len(brokers) > 0
    assert 'mock' in brokers or 'test' in brokers


def test_create_mock_broker():
    """Test creating mock broker"""
    config = {
        'broker': {
            'type': 'mock',
            'initial_balance': 10000
        }
    }
    
    broker = BrokerFactory.create_broker('mock', config)
    assert broker is not None
    assert isinstance(broker, BaseBroker)


def test_invalid_broker_type():
    """Test error handling for invalid broker type"""
    with pytest.raises(ValueError):
        BrokerFactory.create_broker('invalid_broker', {})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
