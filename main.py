# main.py

import yaml
import pandas as pd
import numpy as np
import os
import argparse
import torch
import time
import mt5 as mt5 # Keep this specific import
from collections import defaultdict
import traceback
import joblib # Import joblib for scaler save/load
import logging # Use logging module
from datetime import datetime, timedelta # For scheduling

# Import necessary components from the project structure
from core.simulator import create_simulator
from utils.logger import setup_logger
from data.preprocessing import create_preprocessor
from broker.metatrader_interface import MetaTraderInterface, create_mt5_interface
from core.features import create_feature_pipeline
from core.model import create_model
from core.strategy import create_strategy
from core.portfolio_compound import PortfolioCapitalManager, create_portfolio_capital_manager
from utils.notifier import send_email_notification

# --- Configuration and Initialization Functions ---
def load_config(config_path='configs/params.yaml'):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")
    return config

def initialize_mt5_connection(config, logger):
    """
    Initializes and validates connection to the MetaTrader 5 terminal.
    Returns True on success, False on failure.
    """
    # Check if already initialized and connected
    term_info = mt5.terminal_info()
    if term_info and term_info.connected:
        # logger.info("MT5 connection already active.") # Reduce noise
        return True

    # If not connected, attempt initialization
    mt5_config = config.get('mt5_config', {})
    login = mt5_config.get('login')
    password = mt5_config.get('password')
    server = mt5_config.get('server')
    path = mt5_config.get('path')

    if not all([login, password, server, path]):
         logger.error("MT5 connection failed: Missing login, password, server, or path in config.")
         return False
    if not os.path.exists(path):
        logger.error(f"MT5 connection failed: Executable path not found: {path}")
        return False

    logger.info(f"Initializing MT5 connection from path: {path}...")
    try:
        # Shutdown previous connection if exists but not connected
        if term_info and not term_info.connected:
            logger.info("Shutting down previous inactive MT5 terminal instance.")
            mt5.shutdown()
            time.sleep(1) # Brief pause

        if not mt5.initialize(path=path, login=login, password=password, server=server, timeout=20000):
            logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False

        logger.info(f"MT5 initialized. Version: {mt5.version()}")
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connected to account: {account_info.login} on {account_info.server} ({account_info.company})")
            if not mt5.terminal_info().connected:
                 logger.warning("MT5 initialized but terminal is not connected to the trade server!")
                 # For live trading, connection is essential.
                 if config.get('mode') == 'live':
                      logger.error("Connection to trade server failed. Cannot proceed with live trading.")
                      mt5.shutdown()
                      return False
            return True
        else:
            logger.error(f"Failed to get account info after MT5 initialization, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False
    except Exception as e:
        logger.error(f"Exception during MT5 initialization: {e}", exc_info=True)
        try: mt5.shutdown()
        except Exception: pass
        return False

def load_and_clean_data_for_symbols(config, symbols, logger):
    """Loads and cleans historical data for a list of symbols from MT5."""
    all_data_cleaned = {}
    data_config = config.get('data_config', {})
    timeframe_str = data_config.get('timeframe', 'M1')
    num_bars = data_config.get('num_bars', 50000)
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
    }
    timeframe = timeframe_map.get(timeframe_str.upper())
    if timeframe is None: raise ValueError(f"Invalid timeframe '{timeframe_str}'.")

    # Ensure MT5 connection
    if not initialize_mt5_connection(config, logger): # Use the robust init function
         logger.error("MT5 connection failed. Cannot load historical data.")
         return {}

    try: # Log available symbols
        all_mt5_symbols = mt5.symbols_get(); available_names = [s.name for s in all_mt5_symbols] if all_mt5_symbols else []
        logger.info(f"Symbols available on server ({len(available_names)} total): {available_names[:20]}...")
    except Exception as e: logger.warning(f"Error retrieving symbol list from MT5: {e}")

    logger.info(f"Attempting to load {num_bars} bars of {timeframe_str} data for requested symbols: {symbols}")
    mt5_interface = create_mt5_interface(config); preprocessor = create_preprocessor(config)
    symbols_not_found = []
    for symbol in symbols:
        logger.debug(f"Loading data for {symbol}...")
        try:
            rates_df = mt5_interface.get_recent_bars(symbol, timeframe, num_bars)
            if rates_df is None or rates_df.empty:
                s_info = mt5_interface.get_symbol_info(symbol)
                if s_info is None: symbols_not_found.append(symbol)
                else: logger.warning(f"No bar data returned for {symbol}. Check history availability.")
                continue
            cleaned_df = preprocessor.clean_data(rates_df, symbol)
            if cleaned_df.empty: logger.warning(f"Data for {symbol} became empty after cleaning. Skipping."); continue
            all_data_cleaned[symbol] = cleaned_df; logger.info(f"Finished loading/cleaning for {symbol}. Shape: {cleaned_df.shape}")
        except Exception as e: logger.error(f"Failed to load/clean data for symbol {symbol}: {e}", exc_info=True)
    if symbols_not_found: logger.error(f"Requested symbols NOT FOUND on MT5 server: {symbols_not_found}.")
    if not all_data_cleaned: logger.warning("Failed to load/clean data for ANY requested symbols.")
    return all_data_cleaned

# --- Training Function ---
def run_training_cycle(config, logger):
    """
    Performs one full training cycle: Load data, pre-calculate features,
    run simulation/training for all symbols, save models/scalers.
    Returns True if successful for all processed symbols, False otherwise.
    """
    logger.info("--- Starting Training Cycle ---")
    symbols_requested = config.get('symbols', [])
    if not symbols_requested: logger.error("No symbols specified for training."); return False

    # 1. Initialize MT5 (needed for data loading)
    if not initialize_mt5_connection(config, logger):
        logger.error("MT5 connection failed. Cannot start training cycle."); return False

    # 2. Load Data
    all_historical_data = {}
    try:
        all_historical_data = load_and_clean_data_for_symbols(config, symbols_requested, logger)
        if not all_historical_data: logger.error("Failed to load any data for training cycle."); return False
    except Exception as e: logger.error(f"Critical error during data loading for training: {e}", exc_info=True); return False
    finally: # Shut down MT5 after data loading for the training cycle
        if mt5.terminal_info(): mt5.shutdown(); logger.info("MT5 connection shut down after training data loading.")

    # 3. Pre-calculate Features & Fit Scalers
    results_dir = config.get('results_dir', 'results/'); os.makedirs(results_dir, exist_ok=True)
    all_features_scaled = {}; all_feature_pipelines = {}; precalc_success = True
    logger.info("--- Pre-calculating Features for Training ---")
    for symbol, symbol_data in all_historical_data.items():
        logger.info(f"Calculating features for {symbol}...")
        try:
            fp = create_feature_pipeline(config.get('feature_config', {}))
            logger.info(f"Fitting new scaler for {symbol}.")
            fp.fit(symbol_data) # Always fit new scaler
            if not fp.feature_extractor.is_fitted: raise RuntimeError(f"Scaler for {symbol} could not be fitted.")
            raw_features = fp.feature_extractor._extract_all_features(symbol_data)
            scaled_features = fp.transform(raw_features)
            if scaled_features.empty or scaled_features.isnull().values.any():
                 logger.error(f"Scaled features empty/NaNs for {symbol}. Skipping."); precalc_success = False; continue
            all_features_scaled[symbol] = scaled_features; all_feature_pipelines[symbol] = fp
            logger.info(f"Features pre-calculated/scaled for {symbol}. Shape: {scaled_features.shape}")
        except Exception as e: logger.error(f"Error pre-calculating features for {symbol}: {e}", exc_info=True); precalc_success = False

    symbols_to_process = list(all_features_scaled.keys())
    if not symbols_to_process: logger.error("No symbols have pre-calculated features. Training cycle failed."); return False

    # 4. Run Simulation/Training Loop for each symbol
    logger.info(f"--- Running Simulation/Training for Symbols: {symbols_to_process} ---")
    cycle_errors = False
    for symbol in symbols_to_process:
        logger.info(f"--- Starting Simulation/Training Run for {symbol} ---")
        symbol_data_raw = all_historical_data[symbol]; symbol_features_scaled = all_features_scaled[symbol]
        symbol_config = config.copy(); symbol_config['symbol'] = symbol
        model_cfg = symbol_config.get('model_config', {})
        load_base = model_cfg.get('load_path_base', results_dir); save_base = model_cfg.get('save_path_base', results_dir)
        load_model_path = os.path.join(load_base, f"{symbol}_final_model.pth")
        save_model_path = os.path.join(save_base, f"{symbol}_final_model.pth")
        scaler_path = os.path.join(results_dir, f"{symbol}_scaler.joblib")
        symbol_config['model_config']['load_path'] = load_model_path if os.path.exists(load_model_path) else None
        symbol_config['model_config']['save_path'] = None # Checkpoints not used here
        if 'model_config' not in symbol_config: symbol_config['model_config'] = {}
        symbol_config['model_config']['num_features'] = symbol_features_scaled.shape[1]

        simulator = None; results = None; sim_success = False
        try:
            simulator = create_simulator(data_ohlcv=symbol_data_raw, data_features=symbol_features_scaled, config=symbol_config)
            results = simulator.run()
            logger.info(f"Simulation run completed for {symbol}.")
            sim_success = True
        except Exception as e:
            logger.error(f"Error creating/running TradingSimulator for {symbol}: {e}", exc_info=True)
            cycle_errors = True; continue # Skip saving

        if sim_success:
            scaler_saved = False; model_saved = False
            try: # Save Scaler
                if symbol in all_feature_pipelines: scaler_saved = all_feature_pipelines[symbol].save_scaler(scaler_path)
                else: logger.warning(f"Pipeline not found for {symbol}. Cannot save scaler.")
            except Exception as e: logger.error(f"Exception saving scaler state for {symbol}: {e}", exc_info=True)
            try: # Save Model
                if hasattr(simulator, 'agent') and simulator.agent:
                     simulator.agent.save_model(save_model_path); model_saved = True
                     logger.info(f"Trained model weights saved for {symbol} to {save_model_path}")
                else: logger.warning(f"Simulator/agent not available for {symbol}, cannot save model.")
            except Exception as e: logger.error(f"Could not save final model for {symbol}: {e}", exc_info=True)
            if not scaler_saved or not model_saved: cycle_errors = True; logger.error(f"Failed to save scaler or model for {symbol}.")
            # Save history/plot optionally...
            if sim_success and results and simulator.sim_config.get('plot_results', False):
                try:
                    simulator.plot_results(results)
                except Exception as e:
                    logger.error(f"Error plotting results for {symbol}: {e}")


        logger.info(f"--- Finished Simulation/Training Run for {symbol} ---")

    logger.info("--- Training Cycle Finished ---")
    # Return True only if precalc was successful AND no critical errors during simulation/saving
    return precalc_success and not cycle_errors

# --- Live Trading Function (Refactored with Fix) ---
def run_live_trading_loop(config, logger):
    """
    Initializes components and runs the main live trading loop indefinitely.
    Reloads components as needed (e.g., after retraining).
    """
    logger.info("--- Initializing Live Trading Components ---")
    symbols_requested = config.get('symbols', [])
    if not symbols_requested: logger.error("No symbols specified for live trading."); return

    mt5_interface = None; portfolio_manager = None
    try: # Init MT5 and managers
        if not initialize_mt5_connection(config, logger): raise ConnectionError("Failed to initialize MT5.")
        mt5_interface = create_mt5_interface(config)
        portfolio_manager = create_portfolio_capital_manager(config)
        account_summary = mt5_interface.get_account_summary()
        if account_summary and account_summary.get('balance') is not None:
            portfolio_manager.current_capital = account_summary['balance']
            logger.info(f"Portfolio manager initial capital synced: ${portfolio_manager.current_capital:.2f}")
        else: logger.warning("Could not get account summary. Using capital from config.")
    except Exception as e:
        logger.critical(f"Fatal error initializing core live components: {e}", exc_info=True)
        send_email_notification("GenovoTraderV2 CRITICAL ERROR", f"Live init failed: {e}", config)
        if mt5.terminal_info(): mt5.shutdown()
        return

    models = {}; agents = {}; feature_pipelines = {}; active_symbols = []
    last_component_load_time = None
    live_config = config.get('live_config', {}); loop_interval_sec = live_config.get('loop_interval_sec', 5)
    timeframe_str = config.get('data_config', {}).get('timeframe', 'M1')
    timeframe_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
    mt5_timeframe = timeframe_map.get(timeframe_str.upper())
    if not mt5_timeframe: logger.error(f"Invalid timeframe '{timeframe_str}'. Defaulting to M1."); mt5_timeframe = mt5.TIMEFRAME_M1

    # <<< FIX START >>>
    feature_cfg = config.get('feature_config', {})
    model_cfg = config.get('model_config', {})
    donchian_upper = feature_cfg.get('donchian_upper', 480) # Get Donchian lookback
    model_seq_length = model_cfg.get('seq_length', 201) # Get model sequence length
    # Calculate max bars needed for features + model sequence
    bars_to_fetch = max(model_seq_length, donchian_upper + 1) # Add buffer
    logger.info(f"Determined bars_to_fetch for live features: {bars_to_fetch}")
    # <<< FIX END >>>

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = config.get('results_dir', 'results/')
    load_base = config.get('model_config', {}).get('load_path_base', results_dir)
    strategy_cfg = config.get('strategy_config', {})
    last_print_time = time.time(); print_interval = 60

    def load_live_components(): # Function to load/reload
        nonlocal models, agents, feature_pipelines, active_symbols, last_component_load_time
        logger.info("Attempting to load/reload models, agents, and scalers...")
        _models = {}; _agents = {}; _feature_pipelines = {}; _active_symbols = []
        missing_components = defaultdict(list); load_successful = True
        for symbol in symbols_requested:
            try:
                symbol_info = mt5_interface.get_symbol_info(symbol)
                if not symbol_info: missing_components[symbol].append("SymbolNotFound"); continue
                fp = create_feature_pipeline(config.get('feature_config', {}))
                scaler_path = os.path.join(results_dir, f"{symbol}_scaler.joblib")
                fp.load_scaler(scaler_path)
                if not fp.feature_extractor.is_fitted: missing_components[symbol].append("ScalerLoadFail"); continue
                _feature_pipelines[symbol] = fp
                num_actual_features = len(fp.feature_extractor.feature_names)
                symbol_model_cfg = config.get('model_config', {}).copy(); symbol_model_cfg['num_features'] = num_actual_features
                model = create_model(symbol_model_cfg)
                model_path = os.path.join(load_base, f"{symbol}_final_model.pth")
                if model_path and os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device); model.eval(); _models[symbol] = model
                else: missing_components[symbol].append("ModelNotFound"); continue
                agent = create_strategy(model, strategy_cfg); _agents[symbol] = agent
                _active_symbols.append(symbol); logger.info(f"Successfully loaded components for {symbol}.")
            except Exception as e:
                logger.error(f"Error initializing components for symbol {symbol} during load: {e}", exc_info=True)
                missing_components[symbol].append(f"LoadError ({type(e).__name__})"); load_successful = False
        if not _active_symbols: logger.critical("Failed to load components for ANY symbols."); load_successful = False
        else: logger.info(f"Component loading complete. Active symbols: {_active_symbols}")
        if missing_components: logger.warning(f"Issues loading components for some symbols: {dict(missing_components)}")
        if load_successful: # Update main dictionaries only if successful
             models = _models; agents = _agents; feature_pipelines = _feature_pipelines; active_symbols = _active_symbols
             last_component_load_time = datetime.now()
        return load_successful, active_symbols

    # --- Initial Component Load ---
    load_success, active_symbols = load_live_components()
    if not load_success:
         logger.critical("Initial component load failed. Exiting live mode.")
         send_email_notification("GenovoTraderV2 CRITICAL ERROR", "Live init failed: Component load error.", config)
         if mt5.terminal_info(): mt5.shutdown(); return

    logger.info(f"Starting live trading loop. Interval: {loop_interval_sec}s")
    # --- Main Live Trading / Monitoring Loop ---
    try:
        while True:
            loop_start_time = time.time()
            # --- MT5 Connection Check ---
            if not mt5.terminal_info() or not mt5.terminal_info().connected:
                 logger.error("MT5 connection lost. Attempting to reconnect...")
                 send_email_notification("GenovoTraderV2 Warning", "MT5 connection lost. Attempting reconnect.", config)
                 if not initialize_mt5_connection(config, logger):
                      logger.warning("Reconnect failed. Will retry next cycle."); time.sleep(60); continue
                 else: logger.info("Successfully reconnected to MT5."); send_email_notification("GenovoTraderV2 Info", "Successfully reconnected to MT5.", config)

            # --- Check if components need reloading (e.g., based on file mod time or signal) ---
            # This logic needs to be added if the training happens in a separate process.
            # For now, reloading happens implicitly when this function is called after training.

            # --- Live Trading Logic ---
            potential_trades = []
            try: # Update Portfolio Risk
                all_open_positions = mt5_interface.get_positions(); open_positions_info = {}
                if all_open_positions:
                     for pos in all_open_positions:
                          symbol = pos['symbol']
                          if symbol not in active_symbols: continue
                          symbol_info = mt5_interface.get_symbol_info(symbol)
                          if symbol_info:
                               pos_details = pos.copy(); pos_details['volume_lots'] = pos['volume']; pos_details['entry_price'] = pos['price_open']
                               pos_details['sl_price'] = pos['sl']; pos_details['contract_size'] = symbol_info.get('trade_contract_size', 100000)
                               open_positions_info[symbol] = pos_details
                          else: logger.warning(f"Could not get symbol info for open position {symbol}")
                portfolio_manager.update_open_risk(open_positions_info)
            except Exception as e: logger.error(f"Error updating portfolio risk: {e}", exc_info=True)

            for symbol in active_symbols: # Iterate through successfully loaded symbols
                model = models[symbol]; agent = agents[symbol]; fp = feature_pipelines[symbol]

                # <<< FIX START >>> Fetch enough data
                current_state_df = mt5_interface.get_recent_bars(symbol, mt5_timeframe, bars_to_fetch)
                if current_state_df is None or len(current_state_df) < bars_to_fetch:
                    logger.warning(f"Not enough bars ({len(current_state_df) if current_state_df is not None else 0}/{bars_to_fetch}) for {symbol} feature calculation. Skipping.")
                    continue
                # <<< FIX END >>>

                try: # Feature processing
                    # <<< FIX START >>> Calculate on full, transform slice
                    # Calculate features on the full fetched history
                    raw_features_full = fp.feature_extractor._extract_all_features(current_state_df)

                    # Select the last 'model_seq_length' rows for scaling and model input
                    raw_features_for_transform = raw_features_full.tail(model_seq_length)

                    if len(raw_features_for_transform) < model_seq_length:
                         logger.warning(f"Could not get required sequence length ({model_seq_length}) after feature calculation for {symbol}. Skipping.")
                         continue

                    # Transform only the required sequence length
                    features_df = fp.transform(raw_features_for_transform)
                    # <<< FIX END >>>

                    if features_df.empty or features_df.isnull().values.any():
                        logger.warning(f"Features empty/NaN after transform for {symbol}. Skipping.")
                        continue
                    state_tensor = torch.FloatTensor(features_df.values).unsqueeze(0).to(device)
                except Exception as e:
                    logger.error(f"Error creating features for {symbol}: {e}", exc_info=True)
                    continue

                try: # Model inference
                    action, _, _, _, _, model_outputs = agent.select_action(state_tensor)
                    policy_logits = model_outputs.get('policy_logits', torch.zeros(1,3)); action_probs = torch.softmax(policy_logits, dim=-1)
                    confidence = action_probs.max().item(); stop_loss_pct = model_outputs.get('stop_loss', torch.tensor(0.01)).item()
                    take_profit_pct = model_outputs.get('take_profit', torch.tensor(0.02)).item()
                    logger.debug(f"{symbol}: Action={action}, Conf={confidence:.3f}, SL%={stop_loss_pct:.4f}, TP%={take_profit_pct:.4f}")
                    if (action == 1 or action == 2) and stop_loss_pct > 0:
                         potential_trades.append({'symbol': symbol, 'action': action, 'signal_strength': confidence, 'stop_loss_pct': stop_loss_pct, 'take_profit_pct': take_profit_pct})
                except Exception as e: logger.error(f"Error during inference for {symbol}: {e}", exc_info=True); continue

            allocated_risk = {}
            if potential_trades: # Risk Allocation
                 try:
                      current_capital_for_alloc = portfolio_manager.get_current_capital()
                      if current_capital_for_alloc > 0: allocated_risk = portfolio_manager.allocate_risk_capital(potential_trades)
                      if allocated_risk: logger.info(f"Risk Allocation: { {s: f'${rc:.2f}' for s, rc in allocated_risk.items()} }")
                 except Exception as e: logger.error(f"Error allocating risk: {e}", exc_info=True)

            if allocated_risk: # Trade Execution
                 account_info = mt5_interface.get_account_summary(); account_currency = account_info.get('currency', 'USD') if account_info else 'USD'
                 for symbol, risk_capital in allocated_risk.items():
                     if risk_capital <= 0: continue
                     trade_signal = next((t for t in potential_trades if t['symbol'] == symbol), None);
                     if not trade_signal: continue
                     current_tick = mt5_interface.get_tick(symbol); price_for_calc = current_tick.get('ask') if trade_signal['action'] == 1 else current_tick.get('bid')
                     if not current_tick or price_for_calc <= 0: logger.warning(f"No valid tick for {symbol}. Skipping trade."); continue
                     sl_pct = trade_signal['stop_loss_pct']; notional_size = portfolio_manager.calculate_position_size_notional(symbol, risk_capital, price_for_calc, sl_pct)
                     if notional_size < portfolio_manager.min_position_notional: logger.info(f"[{symbol}] Notional ${notional_size:.2f} < min ${portfolio_manager.min_position_notional}. No trade."); continue
                     position_size_lots = mt5_interface.calculate_lots(symbol, notional_size, account_currency)
                     if position_size_lots <= 0: logger.info(f"[{symbol}] Lot size zero. No trade."); continue
                     logger.info(f"Attempting Trade: {symbol} Action={trade_signal['action']}, Lots={position_size_lots:.2f}, Risk Cap=${risk_capital:.2f}")
                     current_position_info = mt5_interface.get_position_info(symbol); current_position_lots = current_position_info['volume'] if current_position_info else 0.0
                     current_position_type = current_position_info['type'] if current_position_info else None
                     try: # Actual order sending logic
                         action_type = 'BUY' if trade_signal['action'] == 1 else 'SELL'; tp_pct = trade_signal['take_profit_pct']
                         delay_after_close_ms = live_config.get('delay_after_close_ms', 500)
                         if action_type == 'BUY':
                             sl_price = price_for_calc * (1 - sl_pct); tp_price = price_for_calc * (1 + tp_pct)
                             if current_position_type == 1: # Close Short first
                                 logger.info(f"[{symbol}] Closing Short (Ticket: {current_position_info['ticket']}) before BUY.")
                                 close_result = mt5_interface.close_position(symbol, position_info=current_position_info)
                                 if close_result and close_result['retcode'] == mt5.TRADE_RETCODE_DONE: time.sleep(delay_after_close_ms / 1000.0); mt5_interface.open_position(symbol, 'BUY', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                 else: logger.error(f"[{symbol}] Failed to close Short. Cannot open Long.")
                             elif current_position_lots == 0: mt5_interface.open_position(symbol, 'BUY', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                             else: logger.info(f"[{symbol}] BUY signal, but already Long.")
                         elif action_type == 'SELL':
                             sl_price = price_for_calc * (1 + sl_pct); tp_price = price_for_calc * (1 - tp_pct)
                             if current_position_type == 0: # Close Long first
                                 logger.info(f"[{symbol}] Closing Long (Ticket: {current_position_info['ticket']}) before SELL.")
                                 close_result = mt5_interface.close_position(symbol, position_info=current_position_info)
                                 if close_result and close_result['retcode'] == mt5.TRADE_RETCODE_DONE: time.sleep(delay_after_close_ms / 1000.0); mt5_interface.open_position(symbol, 'SELL', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                 else: logger.error(f"[{symbol}] Failed to close Long. Cannot open Short.")
                             elif current_position_lots == 0: mt5_interface.open_position(symbol, 'SELL', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                             else: logger.info(f"[{symbol}] SELL signal, but already Short.")
                     except Exception as e:
                         logger.error(f"Error during trade execution logic for {symbol}: {e}", exc_info=True)
                         send_email_notification(f"GenovoTraderV2 Trade Error ({symbol})", f"Error: {e}\n{traceback.format_exc()}", config)

            # --- Loop Timing & Summary Print ---
            loop_end_time = time.time(); elapsed_time = loop_end_time - loop_start_time
            sleep_time = max(0, loop_interval_sec - elapsed_time)
            if loop_end_time - last_print_time >= print_interval:
                 account_summary = mt5_interface.get_account_summary()
                 if account_summary and account_summary.get('balance') is not None: portfolio_manager.current_capital = account_summary['balance']
                 current_capital = portfolio_manager.get_current_capital(); current_risk_pct = portfolio_manager.get_total_risk_pct() * 100
                 logger.info(f"Live Summary: Capital=${current_capital:.2f}, Equity=${account_summary.get('equity', 0.0):.2f}, Risk={current_risk_pct:.2f}%, Loop Time={elapsed_time:.3f}s, Sleep={sleep_time:.3f}s")
                 last_print_time = loop_end_time
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt: logger.info("Live trading loop interrupted by user (Ctrl+C).")
    except Exception as e: logger.critical(f"Fatal error in live trading loop: {e}", exc_info=True); send_email_notification("GenovoTraderV2 FATAL ERROR", f"Fatal error in live loop:\n{traceback.format_exc()}", config)
    finally: logger.info("Shutting down live trading..."); send_email_notification("GenovoTraderV2 Stopped", "Live trading loop stopping.", config)


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Genovo Trader v2 - Multi Symbol")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, choices=['live', 'train'], default=None, help='Run mode: train (train then exit) or live (run continuously with weekly retrain). Overrides config file.')
    args = parser.parse_args()
    config, logger = None, None

    try:
        config = load_config(args.config)
        logger = setup_logger(config.get('logging_config', {}))
        logger.info(f"--- Starting Genovo Trader v2 (Multi-Symbol) ---")

        # Determine mode: command line overrides config
        run_mode = args.mode if args.mode else config.get('mode', 'live') # Default to live if not specified
        logger.info(f"Run Mode: {run_mode}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        send_email_notification("GenovoTraderV2 Started", f"Process started in '{run_mode}' mode.", config)

        # --- Mode Execution ---
        if run_mode == 'train':
            logger.info("Running in Training-Only mode.")
            run_training_cycle(config, logger)
            logger.info("Training-Only mode finished.")

        elif run_mode == 'live':
            logger.info("Running in Continuous Live Trading mode with Weekly Retraining.")
            last_train_time_file = os.path.join(config.get('results_dir', 'results/'), "last_train_time.txt")
            last_training_time = None

            # Load last training time
            try:
                if os.path.exists(last_train_time_file):
                    with open(last_train_time_file, 'r') as f:
                        timestamp_str = f.read().strip()
                        last_training_time = datetime.fromisoformat(timestamp_str)
                        logger.info(f"Loaded last training time: {last_training_time}")
                else:
                    logger.info("Last training time file not found. Will trigger initial training.")
                    last_training_time = datetime.now() - timedelta(days=8) # Force initial train
            except Exception as e:
                logger.error(f"Error loading last training time: {e}. Triggering initial training.", exc_info=True)
                last_training_time = datetime.now() - timedelta(days=8) # Force initial train

            # --- Main Control Loop ---
            while True:
                # Check if retraining is needed
                time_since_last_train = datetime.now() - last_training_time
                retraining_needed = time_since_last_train >= timedelta(weeks=1)

                if retraining_needed:
                    logger.info(f"Retraining triggered. Time since last train: {time_since_last_train}")
                    send_email_notification("GenovoTraderV2 Retraining", "Starting weekly retraining cycle.", config)

                    # Run the training cycle
                    training_success = run_training_cycle(config, logger)

                    if training_success:
                        logger.info("Weekly retraining cycle completed successfully.")
                        last_training_time = datetime.now()
                        try: # Save the new training time
                            with open(last_train_time_file, 'w') as f: f.write(last_training_time.isoformat())
                            logger.info(f"Updated last training time to: {last_training_time}")
                        except Exception as e: logger.error(f"Failed to save last training time: {e}", exc_info=True)
                        send_email_notification("GenovoTraderV2 Retraining", "Weekly retraining cycle finished successfully.", config)
                        # Reload components in the live loop after successful training
                        # The live loop needs to handle this reload internally now.
                        # We can signal the live loop or rely on it restarting/reloading.
                        # For simplicity, we'll let the live loop reload on its next cycle start.
                        logger.info("Live loop will reload components on its next iteration.")

                    else:
                        logger.error("Weekly retraining cycle failed or had errors. Live trading will continue with previous models/scalers.")
                        send_email_notification("GenovoTraderV2 Retraining ERROR", "Weekly retraining cycle failed. Check logs.", config)
                        # Do NOT update last_training_time, will retry next week

                    # Short pause after training attempt before resuming live check
                    logger.info("Pausing briefly after training attempt...")
                    time.sleep(30)
                    # Continue the outer loop to potentially start live trading now
                    continue

                else:
                    # If not retraining, run the live trading logic
                    logger.info("Starting/Resuming Live Trading Loop...")
                    run_live_trading_loop(config, logger)
                    # If run_live_trading_loop exits (e.g., fatal error), break the outer loop
                    logger.info("Live trading loop function has exited.")
                    break

        else:
            logger.error(f"Invalid mode '{run_mode}' specified.")
            send_email_notification("GenovoTraderV2 Error", f"Startup failed: Invalid mode '{run_mode}'.", config)

    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found. {e}")
        if logger: logger.critical(f"Configuration file not found: {e}")
    except KeyboardInterrupt:
         if logger: logger.info("Main process interrupted by user (Ctrl+C). Shutting down.")
         if config: send_email_notification("GenovoTraderV2 Stopped", "Main process stopped by user.", config)
    except Exception as e:
        print(f"FATAL ERROR during execution: {e}")
        print(traceback.format_exc())
        if logger: logger.critical(f"FATAL ERROR during execution: {e}", exc_info=True)
        if config:
            error_details = f"Fatal error during execution:\n{traceback.format_exc()}"
            send_email_notification("GenovoTraderV2 FATAL ERROR", error_details, config)
    finally:
        if logger: logger.info("--- Genovo Trader v2 finished ---")
        else: print("--- Genovo Trader v2 finished ---")
        if mt5.terminal_info():
            print("Final MT5 shutdown check.")
            mt5.shutdown()

if __name__ == "__main__":
    main()