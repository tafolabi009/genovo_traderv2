# broker/base_broker.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger("genovo_traderv2")


class OrderType(Enum):
    """Standardized order types across brokers"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Standardized order status"""
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BaseBroker(ABC):
    """
    Abstract base class for all broker interfaces.
    Provides a unified API for trading operations across different brokers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker with configuration
        
        Args:
            config: Broker-specific configuration dictionary
        """
        self.config = config
        self.connected = False
        self.account_currency = None
        self.logger = logging.getLogger(f"genovo_traderv2.{self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the broker
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to the broker"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dict containing account details (balance, equity, margin, etc.)
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """
        Get current account balance
        
        Returns:
            float: Account balance
        """
        pass
    
    @abstractmethod
    def get_account_equity(self) -> float:
        """
        Get current account equity
        
        Returns:
            float: Account equity
        """
        pass
    
    @abstractmethod
    def place_order(self, 
                   symbol: str,
                   order_type: OrderType,
                   volume: float,
                   price: Optional[float] = None,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Place an order
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (from OrderType enum)
            volume: Order volume/size
            price: Limit price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            **kwargs: Additional broker-specific parameters
            
        Returns:
            Dict containing order result with keys:
                - success: bool
                - order_id: str/int
                - message: str
                - details: Any
        """
        pass
    
    @abstractmethod
    def close_position(self, position_id: Any, volume: Optional[float] = None) -> Dict[str, Any]:
        """
        Close an open position
        
        Args:
            position_id: Position identifier
            volume: Partial close volume (None for full close)
            
        Returns:
            Dict containing close result
        """
        pass
    
    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open positions
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        pass
    
    @abstractmethod
    def get_historical_data(self,
                           symbol: str,
                           timeframe: str,
                           start_date: Any,
                           end_date: Any) -> Any:
        """
        Get historical price data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', 'D')
            start_date: Start date/time
            end_date: End date/time
            
        Returns:
            Historical data (pandas DataFrame or similar)
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get current bid/ask prices
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with 'bid' and 'ask' prices
        """
        pass
    
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict containing symbol details (tick size, contract size, etc.)
        """
        pass
    
    def ensure_connection(self) -> bool:
        """
        Ensure broker connection is active, reconnect if needed
        
        Returns:
            bool: True if connected
        """
        if not self.connected:
            self.logger.warning("Connection lost, attempting to reconnect...")
            return self.connect()
        return True
    
    def calculate_position_size(self,
                               symbol: str,
                               risk_amount: float,
                               entry_price: float,
                               stop_loss_price: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Trading symbol
            risk_amount: Amount to risk (in account currency)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            float: Position size (volume/lots)
        """
        symbol_info = self.get_symbol_info(symbol)
        
        # Calculate price risk per unit
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            self.logger.error("Stop loss price equals entry price")
            return 0
        
        # Get contract size
        contract_size = symbol_info.get('contract_size', 100000)
        point = symbol_info.get('point', 0.00001)
        
        # Calculate position size
        pip_risk = price_risk / point
        value_per_pip = contract_size * point
        position_size = risk_amount / (pip_risk * value_per_pip)
        
        # Apply min/max constraints
        min_volume = symbol_info.get('volume_min', 0.01)
        max_volume = symbol_info.get('volume_max', 100.0)
        volume_step = symbol_info.get('volume_step', 0.01)
        
        # Round to volume step
        position_size = round(position_size / volume_step) * volume_step
        position_size = max(min_volume, min(position_size, max_volume))
        
        return position_size
