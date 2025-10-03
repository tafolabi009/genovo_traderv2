# broker/broker_factory.py

"""
Broker Factory - Easy plug-and-play broker switching
"""

from typing import Dict, Any
import logging

logger = logging.getLogger("genovo_traderv2")


class BrokerFactory:
    """Factory class for creating broker instances"""
    
    # Registry of available brokers
    _brokers = {}
    
    @classmethod
    def register_broker(cls, name: str, broker_class):
        """
        Register a new broker implementation
        
        Args:
            name: Broker identifier (e.g., 'mt5', 'ibkr', 'alpaca')
            broker_class: Broker class implementing BaseBroker
        """
        cls._brokers[name.lower()] = broker_class
        logger.info(f"Registered broker: {name}")
    
    @classmethod
    def create_broker(cls, broker_type: str, config: Dict[str, Any]):
        """
        Create a broker instance
        
        Args:
            broker_type: Type of broker (e.g., 'mt5', 'ibkr')
            config: Configuration dictionary
            
        Returns:
            Broker instance
            
        Raises:
            ValueError: If broker type not registered
        """
        broker_type = broker_type.lower()
        
        if broker_type not in cls._brokers:
            available = ', '.join(cls._brokers.keys())
            raise ValueError(
                f"Unknown broker type: {broker_type}. "
                f"Available brokers: {available}"
            )
        
        broker_class = cls._brokers[broker_type]
        logger.info(f"Creating {broker_type} broker instance")
        
        try:
            broker = broker_class(config)
            return broker
        except Exception as e:
            logger.error(f"Failed to create {broker_type} broker: {e}")
            raise
    
    @classmethod
    def list_brokers(cls):
        """List all registered brokers"""
        return list(cls._brokers.keys())


# Auto-register available brokers
def _register_default_brokers():
    """Register all available broker implementations"""
    
    # MetaTrader 5
    try:
        from broker.metatrader_interface import MetatraderInterface
        BrokerFactory.register_broker('mt5', MetatraderInterface)
        BrokerFactory.register_broker('metatrader5', MetatraderInterface)
    except ImportError as e:
        logger.warning(f"MetaTrader 5 broker not available: {e}")
    
    # Mock broker (for testing)
    try:
        from broker.mock import MockBroker
        BrokerFactory.register_broker('mock', MockBroker)
        BrokerFactory.register_broker('test', MockBroker)
    except ImportError as e:
        logger.warning(f"Mock broker not available: {e}")
    
    # TODO: Add more brokers
    # Interactive Brokers
    # try:
    #     from broker.ibkr_interface import IBKRInterface
    #     BrokerFactory.register_broker('ibkr', IBKRInterface)
    # except ImportError:
    #     pass
    
    # Alpaca
    # try:
    #     from broker.alpaca_interface import AlpacaInterface
    #     BrokerFactory.register_broker('alpaca', AlpacaInterface)
    # except ImportError:
    #     pass


# Initialize default brokers
_register_default_brokers()


def create_broker_from_config(config: Dict[str, Any]):
    """
    Convenience function to create broker from config
    
    Args:
        config: Configuration dictionary with 'broker' section
        
    Returns:
        Configured broker instance
    """
    broker_config = config.get('broker', {})
    broker_type = broker_config.get('type', 'mt5')  # Default to MT5
    
    return BrokerFactory.create_broker(broker_type, config)
