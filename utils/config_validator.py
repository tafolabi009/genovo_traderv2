# utils/config_validator.py

"""
Configuration validation and schema checking
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("genovo_traderv2")


class ConfigValidator:
    """Validates configuration files for correctness and completeness"""
    
    REQUIRED_FIELDS = {
        'mode': ['simulation', 'live'],
        'symbols': list,
        'results_dir': str,
    }
    
    REQUIRED_BROKER_FIELDS = {
        'account_id': (int, str),
        'password': str,
        'server': str,
        'mt5_path': str,
    }
    
    REQUIRED_MODEL_FIELDS = {
        'num_features': int,
        'seq_length': int,
        'hidden_size': int,
        'num_layers': int,
        'num_heads': int,
    }
    
    REQUIRED_RISK_FIELDS = {
        'max_total_risk_pct': float,
        'max_allocation_per_trade_pct': float,
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required top-level fields
        for field, expected_type in cls.REQUIRED_FIELDS.items():
            if field not in config:
                errors.append(f"Missing required field: {field}")
            elif expected_type == list and not isinstance(config[field], list):
                errors.append(f"Field '{field}' must be a list")
            elif expected_type == str and not isinstance(config[field], str):
                errors.append(f"Field '{field}' must be a string")
        
        # Validate mode
        if 'mode' in config:
            if config['mode'] not in ['simulation', 'live']:
                errors.append(f"Invalid mode: {config['mode']}. Must be 'simulation' or 'live'")
        
        # Validate symbols
        if 'symbols' in config:
            if not config['symbols']:
                errors.append("Symbols list cannot be empty")
            elif not all(isinstance(s, str) for s in config['symbols']):
                errors.append("All symbols must be strings")
        
        # Validate broker config (for live mode)
        if config.get('mode') == 'live':
            broker_config = config.get('broker', {})
            for field, expected_type in cls.REQUIRED_BROKER_FIELDS.items():
                if field not in broker_config:
                    errors.append(f"Missing broker field: {field}")
                elif not isinstance(broker_config[field], expected_type):
                    errors.append(f"Broker field '{field}' has wrong type")
        
        # Validate model config
        model_config = config.get('model_config', {})
        for field, expected_type in cls.REQUIRED_MODEL_FIELDS.items():
            if field not in model_config:
                errors.append(f"Missing model_config field: {field}")
            elif not isinstance(model_config[field], expected_type):
                errors.append(f"model_config field '{field}' must be {expected_type.__name__}")
        
        # Validate portfolio capital config
        portfolio_config = config.get('portfolio_capital_config', {})
        for field, expected_type in cls.REQUIRED_RISK_FIELDS.items():
            if field not in portfolio_config:
                errors.append(f"Missing portfolio_capital_config field: {field}")
            elif not isinstance(portfolio_config[field], expected_type):
                errors.append(f"portfolio_capital_config field '{field}' must be {expected_type.__name__}")
        
        # Validate risk percentages are in valid range
        if 'portfolio_capital_config' in config:
            pc = config['portfolio_capital_config']
            if 'max_total_risk_pct' in pc:
                if not (0 < pc['max_total_risk_pct'] <= 1):
                    errors.append("max_total_risk_pct must be between 0 and 1")
            if 'max_allocation_per_trade_pct' in pc:
                if not (0 < pc['max_allocation_per_trade_pct'] <= 1):
                    errors.append("max_allocation_per_trade_pct must be between 0 and 1")
        
        # Check model architecture constraints
        if 'model_config' in config:
            mc = config['model_config']
            if 'hidden_size' in mc and 'num_heads' in mc:
                if mc['hidden_size'] % mc['num_heads'] != 0:
                    errors.append("hidden_size must be divisible by num_heads")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @classmethod
    def validate_and_log(cls, config: Dict[str, Any]) -> bool:
        """
        Validate config and log errors
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if valid
        """
        is_valid, errors = cls.validate_config(config)
        
        if is_valid:
            logger.info("✓ Configuration validation passed")
        else:
            logger.error("✗ Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid
    
    @classmethod
    def get_config_summary(cls, config: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of the configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            str: Summary text
        """
        summary = []
        summary.append("=== Configuration Summary ===")
        summary.append(f"Mode: {config.get('mode', 'unknown')}")
        summary.append(f"Symbols: {', '.join(config.get('symbols', []))}")
        
        if 'model_config' in config:
            mc = config['model_config']
            summary.append(f"Model: {mc.get('num_layers', 0)} layers, "
                         f"{mc.get('hidden_size', 0)} hidden size, "
                         f"{mc.get('num_heads', 0)} heads")
        
        if 'portfolio_capital_config' in config:
            pc = config['portfolio_capital_config']
            summary.append(f"Capital: ${pc.get('initial_capital', 0):.2f}, "
                         f"Max Risk: {pc.get('max_total_risk_pct', 0)*100:.1f}%")
        
        summary.append("=" * 30)
        return "\n".join(summary)


def validate_config_file(config: Dict[str, Any]) -> bool:
    """
    Convenience function to validate config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid
    """
    return ConfigValidator.validate_and_log(config)
