#!/usr/bin/env python3
"""
Genovo Trader V2 - Launcher Script
Provides an easy way to start the trading bot with various modes
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config_validator import ConfigValidator
from utils.logger import setup_logger


def print_banner():
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              🚀 GENOVO TRADER V2 🚀                          ║
║                                                               ║
║         Professional High-Frequency Trading System            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_and_validate_config(config_path):
    """Load and validate configuration"""
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate
        is_valid, errors = ConfigValidator.validate_config(config)
        if not is_valid:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
            return None
        
        print("✓ Configuration loaded and validated")
        print(ConfigValidator.get_config_summary(config))
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Genovo Trader V2 - Professional Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start in simulation mode (backtesting)
  python launcher.py --mode simulation
  
  # Start live trading
  python launcher.py --mode live
  
  # Start only the web UI
  python launcher.py --ui-only
  
  # Use custom config file
  python launcher.py --config myconfig.yaml --mode live
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['simulation', 'live'],
        help='Trading mode (simulation for backtesting, live for real trading)'
    )
    
    parser.add_argument(
        '--config',
        default='configs/params.yaml',
        help='Path to configuration file (default: configs/params.yaml)'
    )
    
    parser.add_argument(
        '--ui-only',
        action='store_true',
        help='Start only the web UI without trading'
    )
    
    parser.add_argument(
        '--ui-port',
        type=int,
        default=5000,
        help='Port for web UI (default: 5000)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Override symbols from config (e.g., --symbols EURUSD GBPUSD)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration and exit'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_and_validate_config(args.config)
    
    if config is None:
        return 1
    
    # Override mode if specified
    if args.mode:
        config['mode'] = args.mode
        print(f"Mode overridden to: {args.mode}")
    
    # Override symbols if specified
    if args.symbols:
        config['symbols'] = args.symbols
        print(f"Symbols overridden to: {', '.join(args.symbols)}")
    
    # If validate-only, exit here
    if args.validate_only:
        print("\n✓ Configuration is valid!")
        return 0
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        'genovo_traderv2',
        config.get('logging_config', {}).get('log_file', 'results/genovo_traderv2.log'),
        level=log_level
    )
    
    # Start UI only mode
    if args.ui_only:
        print(f"\n🌐 Starting Web UI on port {args.ui_port}...")
        print(f"   Access dashboard at: http://localhost:{args.ui_port}")
        print("\n   Press Ctrl+C to stop\n")
        
        from ui.app import run_ui_server
        try:
            run_ui_server(host='0.0.0.0', port=args.ui_port, debug=args.debug)
        except KeyboardInterrupt:
            print("\n\n👋 UI server stopped by user")
        return 0
    
    # Start main trading system
    print(f"\n🚀 Starting Genovo Trader V2 in {config['mode']} mode...")
    print(f"   Trading symbols: {', '.join(config.get('symbols', []))}")
    print(f"   Web UI will be available at: http://localhost:{args.ui_port}")
    print("\n   Press Ctrl+C to stop\n")
    
    try:
        # Import and run main
        from main import main as run_main
        
        # Start UI in background if not in ui-only mode
        if not args.ui_only:
            from ui.app import start_ui_thread
            start_ui_thread(host='0.0.0.0', port=args.ui_port)
        
        # Run main trading loop
        return run_main()
        
    except KeyboardInterrupt:
        print("\n\n👋 Trading bot stopped by user")
        logger.info("Trading bot stopped by user (Ctrl+C)")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
