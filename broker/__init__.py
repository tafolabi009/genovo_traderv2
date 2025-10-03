# broker/__init__.py

"""
Broker package - Multi-broker support for trading
"""

from broker.base_broker import BaseBroker, OrderType, OrderStatus
from broker.broker_factory import BrokerFactory, create_broker_from_config

# For backwards compatibility
try:
    from broker.metatrader_interface import MetatraderInterface, create_mt5_interface
except ImportError:
    MetatraderInterface = None
    create_mt5_interface = None

__all__ = [
    'BaseBroker',
    'OrderType',
    'OrderStatus',
    'BrokerFactory',
    'create_broker_from_config',
    'MetatraderInterface',
    'create_mt5_interface',
]
