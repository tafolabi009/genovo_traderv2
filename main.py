# main.py

import yaml
import pandas as pd
import numpy as np
import os
import argparse
import torch
import time
import mt5 as mt5
from collections import defaultdict
import traceback
import joblib # Import joblib for scaler save/load
import logging # Use logging module

# Import necessary components from the project structure
from core.simulator import create_simulator
from utils.logger import setup_logger
from data.preprocessing import create_preprocessor # Import preprocessor
from broker.metatrader_interface import MetaTraderInterface, create_mt5_interface
from core.features import create_feature_pipeline # FeaturePipeline now handles scaler save/load via extractor
from core.model import create_model
from core.strategy import create_strategy
from core.portfolio_compound import PortfolioCapitalManager, create_portfolio_capital_manager
from utils.notifier import send_email_notification


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
    if mt5.terminal_info() and mt5.terminal_info().connected:
        logger.info("MT5 connection already active.")
        return True

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
        if not mt5.initialize(path=path, login=login, password=password, server=server, timeout=20000): # Increased timeout
            logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False

        logger.info(f"MT5 initialized. Version: {mt5.version()}")
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connected to account: {account_info.login} on {account_info.server} ({account_info.company})")
            if not mt5.terminal_info().connected:
                 logger.warning("MT5 initialized but terminal is not connected to the trade server!")
                 # Decide if this is critical - for live trading, it is.
                 # return False # Uncomment if connection is mandatory
            return True
        else:
            logger.error(f"Failed to get account info after MT5 initialization, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False
    except Exception as e:
        logger.error(f"Exception during MT5 initialization: {e}", exc_info=True)
        try:
             mt5.shutdown()
        except Exception: pass # Ignore shutdown errors if init failed badly
        return False

def load_and_clean_data_for_symbols(config, symbols, logger):
    """
    Loads and cleans historical data for a list of symbols from MT5.
    Uses the DataPreprocessor for cleaning. Logs available symbols.

    Returns:
        dict: Dictionary mapping symbol -> cleaned DataFrame. Returns empty dict on critical failure.
    """
    all_data_raw = {}
    all_data_cleaned = {}
    data_config = config.get('data_config', {})
    timeframe_str = data_config.get('timeframe', 'M1')
    num_bars = data_config.get('num_bars', 100000) # Default to 100k bars

    # Map timeframe string to MT5 constant
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
    }
    timeframe = timeframe_map.get(timeframe_str.upper())
    if timeframe is None:
        logger.error(f"Invalid timeframe '{timeframe_str}' in config.")
        raise ValueError(f"Invalid timeframe '{timeframe_str}'.")

    # Ensure MT5 is connected (should be called after initialize_mt5_connection)
    if not mt5.terminal_info() or not mt5.terminal_info().connected:
         logger.error("MT5 not connected. Cannot load historical data.")
         return {} # Return empty dict

    # --- Log available symbols ---
    try:
        all_mt5_symbols = mt5.symbols_get()
        if all_mt5_symbols:
            available_names = [s.name for s in all_mt5_symbols]
            logger.info(f"Symbols available on server ({len(available_names)} total): {available_names[:20]}...") # Log first 20
        else:
            logger.warning("Could not retrieve symbol list from MT5.")
    except Exception as e:
        logger.warning(f"Error retrieving symbol list from MT5: {e}")
    # --- End log available symbols ---


    logger.info(f"Attempting to load {num_bars} bars of {timeframe_str} data for requested symbols: {symbols}")
    mt5_interface = create_mt5_interface(config) # Use interface helper methods
    preprocessor = create_preprocessor(config) # Create preprocessor for cleaning

    symbols_not_found = []
    for symbol in symbols:
        logger.debug(f"Loading data for {symbol}...")
        try:
            # Use interface method which includes ensuring visibility
            # This now logs errors internally if symbol not found
            rates_df = mt5_interface.get_recent_bars(symbol, timeframe, num_bars)

            if rates_df is None or rates_df.empty:
                # Check if the symbol was actually found by get_symbol_info inside get_recent_bars
                s_info = mt5_interface.get_symbol_info(symbol)
                if s_info is None:
                    # Error "Symbol not found" was already logged by get_symbol_info/ensure_visible
                    symbols_not_found.append(symbol)
                else:
                    # Symbol exists but no data returned (maybe insufficient history?)
                    logger.warning(f"No bar data returned for {symbol} (it exists on server). Check history availability.")
                continue # Skip this symbol

            logger.info(f"Loaded {len(rates_df)} raw bars for {symbol}.")
            all_data_raw[symbol] = rates_df

            # Clean the loaded data
            cleaned_df = preprocessor.clean_data(rates_df, symbol)
            if cleaned_df.empty:
                 logger.warning(f"Data for {symbol} became empty after cleaning. Skipping.")
                 continue

            all_data_cleaned[symbol] = cleaned_df
            logger.info(f"Finished loading and cleaning for {symbol}. Shape: {cleaned_df.shape}")

        except Exception as e:
            logger.error(f"Failed to load or clean data for symbol {symbol}: {e}", exc_info=True)
            # Decide whether to continue with other symbols or stop
            # For now, continue

    if symbols_not_found:
         logger.error(f"The following requested symbols were NOT FOUND on the MT5 server: {symbols_not_found}. Please check names in config against available symbols.")

    if not all_data_cleaned:
         logger.warning("Failed to load or clean data for ANY requested symbols.")

    return all_data_cleaned


def run_simulation_or_training(config, logger):
    """
    Runs the trading simulation/training independently for each configured symbol.
    Saves the fitted scaler state for each symbol after training.
    """
    logger.info("--- Starting Multi-Symbol Simulation/Training Mode ---")
    symbols_requested = config.get('symbols', [])
    if not symbols_requested:
        logger.error("No symbols specified in configuration ('symbols' list). Exiting.")
        send_email_notification("GenovoTraderV2 Error", "Startup failed: No symbols specified in config.", config)
        return

    # Ensure MT5 connection for data loading
    if not initialize_mt5_connection(config, logger):
        logger.error("MT5 connection failed. Cannot proceed with simulation/training.")
        send_email_notification("GenovoTraderV2 Error", "Startup failed: Could not initialize MT5 for data loading.", config)
        return

    try:
        # Load and clean data using the dedicated function
        # This now logs available symbols and specific "not found" errors
        all_historical_data = load_and_clean_data_for_symbols(config, symbols_requested, logger)
        if not all_historical_data:
            logger.error("Failed to load data for any requested symbols. Exiting.")
            send_email_notification("GenovoTraderV2 Error", "Startup failed: Could not load/clean data for any requested symbols.", config)
            return
    except Exception as e:
        logger.error(f"Critical error during multi-symbol data loading/cleaning: {e}", exc_info=True)
        error_details = f"Critical error during data loading/cleaning:\n{traceback.format_exc()}"
        send_email_notification("GenovoTraderV2 Error", f"Startup failed: {error_details}", config)
        return
    finally:
        # Shutdown MT5 if it's only needed for data loading in this mode
        if config.get('mode', 'simulation') != 'live':
             if mt5.terminal_info(): mt5.shutdown(); logger.info("MT5 connection shut down after data loading.")


    results_dir = config.get('results_dir', 'results/')
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")

    all_results = {}
    errors_occurred = False
    feature_pipelines = {} # Store fitted pipelines to save scalers

    # Iterate only over symbols for which data was successfully loaded
    symbols_to_process = list(all_historical_data.keys())
    logger.info(f"Proceeding with simulation/training for symbols: {symbols_to_process}")

    for symbol in symbols_to_process:
        logger.info(f"--- Starting Simulation/Training for {symbol} ---")
        symbol_data = all_historical_data[symbol]
        # Create a copy of config for this symbol to avoid cross-contamination of paths etc.
        symbol_config = config.copy()
        symbol_config['symbol'] = symbol # Add symbol identifier to config

        # --- Configure Paths ---
        model_cfg = symbol_config.get('model_config', {})
        load_base = model_cfg.get('load_path_base', results_dir)
        save_base = model_cfg.get('save_path_base', results_dir)
        # Define paths for loading existing model/scaler and saving new ones
        load_model_path = os.path.join(load_base, f"{symbol}_final_model.pth") # Path to load previous final model
        save_model_path = os.path.join(save_base, f"{symbol}_final_model.pth") # Path to save final model
        save_checkpoint_path = os.path.join(save_base, f"{symbol}_checkpoint.pth") # Path for intermediate checkpoints (optional)
        scaler_path = os.path.join(results_dir, f"{symbol}_scaler.joblib") # Path for scaler state

        # Update config for the simulator instance
        symbol_config['model_config']['load_path'] = load_model_path if os.path.exists(load_model_path) else None
        symbol_config['model_config']['save_path'] = save_checkpoint_path # Simulator might save checkpoints

        simulator = None # Define simulator outside try block
        try:
            # Create simulator (which initializes features, model, agent, capital manager)
            simulator = create_simulator(data=symbol_data, config=symbol_config)

            # Load scaler state if it exists
            if os.path.exists(scaler_path):
                 simulator.feature_pipeline.load_scaler(scaler_path)
                 if not simulator.feature_pipeline.feature_extractor.is_fitted:
                      logger.warning(f"Scaler file found for {symbol} but failed to load correctly. Fitting new scaler.")
                      # Fit scaler if loading failed
                      simulator.feature_pipeline.fit(symbol_data)
            else:
                 logger.info(f"No existing scaler found for {symbol}. Fitting new scaler.")
                 # Fit scaler if file doesn't exist
                 simulator.feature_pipeline.fit(symbol_data)

            # Ensure scaler is fitted before running
            if not simulator.feature_pipeline.feature_extractor.is_fitted:
                 raise RuntimeError(f"Scaler for {symbol} could not be fitted.")

            # Store the pipeline instance for saving later
            feature_pipelines[symbol] = simulator.feature_pipeline
            logger.info(f"TradingSimulator and FeaturePipeline prepared for {symbol}.")

        except Exception as e:
            logger.error(f"Error creating/preparing TradingSimulator for {symbol}: {e}", exc_info=True)
            errors_occurred = True
            continue # Skip to next symbol

        # --- Run Simulation ---
        try:
            results = simulator.run() # run() now returns results dictionary
            all_results[symbol] = results
            logger.info(f"Simulation run completed for {symbol}.")
        except Exception as e:
            logger.error(f"Error during simulation run for {symbol}: {e}", exc_info=True)
            # Try to get partial results if run failed mid-way
            all_results[symbol] = simulator.get_results()
            errors_occurred = True

        # --- Save Scaler State ---
        try:
            if symbol in feature_pipelines and feature_pipelines[symbol].feature_extractor.is_fitted:
                 feature_pipelines[symbol].save_scaler(scaler_path) # Use pipeline's save method
            else:
                 logger.warning(f"Scaler for {symbol} was not fitted or pipeline not found. Cannot save scaler state.")
        except Exception as e:
             logger.error(f"Error saving scaler state for {symbol}: {e}", exc_info=True)
             errors_occurred = True

        # --- Save Results History ---
        try:
            history_df = results.get('history')
            if history_df is not None and not history_df.empty:
                history_path = os.path.join(results_dir, f"{symbol}_history.csv")
                history_df.to_csv(history_path)
                logger.info(f"Simulation history saved for {symbol} to {history_path}")
            elif not errors_occurred: # Don't warn if run failed anyway
                 logger.warning(f"Simulation history empty/missing for {symbol}.")
        except Exception as e:
            logger.error(f"Error saving history for {symbol}: {e}", exc_info=True); errors_occurred = True

        # --- Save Final Model Weights ---
        try:
            if hasattr(simulator, 'agent') and simulator.agent:
                 simulator.agent.save_model(save_model_path) # Use the final model save path
                 logger.info(f"Final model weights saved for {symbol} to {save_model_path}")
            elif not errors_occurred:
                 logger.warning(f"Simulator/agent not available for {symbol}, cannot save model.")
        except Exception as e:
            logger.error(f"Could not save final model for {symbol}: {e}", exc_info=True); errors_occurred = True

        # --- Plot Results ---
        if symbol_config.get('simulator_config', {}).get('plot_results', False):
            try:
                if hasattr(simulator, 'plot_results') and results.get('history') is not None and not results['history'].empty:
                     simulator.plot_results(results); logger.info(f"Results plotted for {symbol}.")
                elif not errors_occurred:
                     logger.warning(f"Cannot plot results for {symbol} - simulator or history missing/empty.")
            except Exception as e:
                 logger.error(f"Could not plot results for {symbol}: {e}", exc_info=True)

        logger.info(f"--- Finished Simulation/Training for {symbol} ---")

    # --- Aggregate Results & Final Notification ---
    summary_lines = ["--- Overall Simulation Summary ---"]
    valid_results_count = len(all_results)
    processed_symbols_list = list(all_results.keys())
    summary_lines.append(f"Successfully Processed Symbols: {processed_symbols_list}")
    symbols_with_data_issues = [s for s in symbols_requested if s not in processed_symbols_list]
    if symbols_with_data_issues:
        summary_lines.append(f"Symbols Skipped (Data Issues): {symbols_with_data_issues}")


    if all_results:
        # Use initial capital from portfolio config (assumed same start for all sims here)
        initial_capital_per_sim = config.get('portfolio_capital_config',{}).get('initial_capital', 10000.0)
        total_initial_capital = initial_capital_per_sim * valid_results_count
        total_final_capital = sum(res.get('final_capital', 0) for res in all_results.values())

        summary_lines.append(f"Total Initial Capital (Summed): ${total_initial_capital:.2f} ({valid_results_count} sims)")
        summary_lines.append(f"Total Final Capital (Summed):   ${total_final_capital:.2f}")
        if total_initial_capital > 0:
             total_return = ((total_final_capital - total_initial_capital) / total_initial_capital) * 100
             summary_lines.append(f"Overall Return (Simple Sum):    {total_return:.2f}%")

        # Average metrics
        avg_win_rate = np.mean([res.get('win_rate', 0) for res in all_results.values()])
        avg_max_drawdown = np.mean([res.get('max_drawdown', 0) for res in all_results.values()])
        avg_sharpe = np.mean([res.get('sharpe_ratio', 0) for res in all_results.values()])
        summary_lines.append(f"Average Win Rate:        {avg_win_rate:.2f}%")
        summary_lines.append(f"Average Max Drawdown:    {avg_max_drawdown:.2f}%")
        summary_lines.append(f"Average Sharpe Ratio:    {avg_sharpe:.3f}")

    summary_lines.append("---------------------------------")
    summary_message = "\n".join(summary_lines)
    logger.info(summary_message)
    email_subject = f"GenovoTraderV2 {config.get('mode','Simulation').capitalize()} Complete" # Use mode in subject
    email_body = f"{config.get('mode','Simulation').capitalize()} process finished.\n\n{summary_message}"
    if errors_occurred:
        email_subject += " (with errors)"
        email_body += "\n\nNOTE: Errors occurred during the process. Please check the logs."
    send_email_notification(email_subject, email_body, config)
    logger.info(f"--- Multi-Symbol {config.get('mode','Simulation').capitalize()} Finished ---")


def run_live(config, logger):
    """Runs the live trading bot for multiple symbols, loading scaler state."""
    logger.info("--- Starting Multi-Symbol Live Trading Mode ---")
    symbols_requested = config.get('symbols', [])
    if not symbols_requested:
        logger.error("No symbols specified. Exiting."); send_email_notification("GenovoTraderV2 Error", "Live Startup failed: No symbols specified.", config); return

    # Initialize MT5 connection ONCE
    if not initialize_mt5_connection(config, logger):
        logger.error("Failed to initialize MT5. Exiting live mode."); send_email_notification("GenovoTraderV2 Error", "Live Startup failed: Could not initialize MT5.", config); return

    # --- Create Core Components ---
    try:
        mt5_interface = create_mt5_interface(config); logger.info("MetaTraderInterface created.")
        portfolio_manager = create_portfolio_capital_manager(config)
        # Sync initial capital from account
        account_summary = mt5_interface.get_account_summary()
        if account_summary and account_summary.get('balance') is not None:
            portfolio_manager.current_capital = account_summary['balance']
            logger.info(f"Portfolio manager initial capital synced from account: ${portfolio_manager.current_capital:.2f}")
        else:
            logger.warning("Could not get account summary to sync initial capital. Using value from config.")
            # Keep initial capital from config if summary fails
            logger.info(f"Portfolio manager initial capital: ${portfolio_manager.initial_capital:.2f} (from config)")

    except Exception as e:
        logger.error(f"Error creating core components (MT5 Interface, Portfolio Manager): {e}", exc_info=True)
        error_details = f"Live Startup failed: Error creating core components:\n{traceback.format_exc()}"
        send_email_notification("GenovoTraderV2 Error", error_details, config)
        if mt5.terminal_info(): mt5.shutdown()
        return

    # --- Initialize components FOR EACH SYMBOL ---
    models = {}
    agents = {} # Store agent instances per symbol
    feature_pipelines = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cfg = config.get('model_config', {})
    strategy_cfg = config.get('strategy_config', {}) # Get strategy config
    results_dir = config.get('results_dir', 'results/') # Use results_dir for loading scaler/model
    load_base = model_cfg.get('load_path_base', results_dir)
    logger.info(f"Initializing models, agents, and feature pipelines for symbols: {symbols_requested}")
    initialization_successful = True
    missing_components = defaultdict(list) # Store missing components per symbol
    active_symbols = [] # List of symbols successfully initialized

    for symbol in symbols_requested:
        logger.info(f"Initializing for {symbol}...")
        try:
            # 1. Check if symbol exists on server (using MT5 interface which caches)
            symbol_info = mt5_interface.get_symbol_info(symbol)
            if not symbol_info:
                 logger.error(f"Symbol {symbol} not found or unavailable on MT5 server. Skipping.")
                 missing_components[symbol].append("Symbol Not Found")
                 initialization_successful = False
                 continue

            # 2. Feature Pipeline & Load Scaler
            fp = create_feature_pipeline(config.get('feature_config', {}))
            scaler_path = os.path.join(results_dir, f"{symbol}_scaler.joblib")
            fp.load_scaler(scaler_path) # Attempt to load scaler state
            if not fp.feature_extractor.is_fitted:
                 logger.error(f"Scaler state not loaded successfully for {symbol} from {scaler_path}. Cannot trade this symbol.")
                 missing_components[symbol].append("Scaler Load Failed")
                 initialization_successful = False
                 continue # Skip this symbol if scaler is missing/failed to load
            feature_pipelines[symbol] = fp
            logger.debug(f"Feature pipeline loaded/checked for {symbol}.")

            # 3. Model
            model = create_model(model_cfg)
            model_path = os.path.join(load_base, f"{symbol}_final_model.pth") # Load the FINAL model
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval() # Set model to evaluation mode
                models[symbol] = model
                logger.debug(f"Model loaded for {symbol} from {model_path}")
            else:
                logger.error(f"Model file not found for {symbol} at {model_path}. Cannot trade this symbol.")
                missing_components[symbol].append("Model Not Found")
                if symbol in feature_pipelines: del feature_pipelines[symbol] # Clean up pipeline if model missing
                initialization_successful = False
                continue # Skip this symbol if model missing

            # 4. Strategy Agent (not used for updates in live mode, but needed for select_action)
            agent = create_strategy(model, strategy_cfg) # Pass the loaded model here
            agents[symbol] = agent
            logger.debug(f"Strategy agent created for {symbol}.")

            # If all steps successful, add to active list
            active_symbols.append(symbol)
            logger.info(f"Successfully initialized components for {symbol}.")

        except Exception as e:
            logger.error(f"Error initializing components for symbol {symbol}: {e}", exc_info=True)
            missing_components[symbol].append(f"Initialization Error ({type(e).__name__})")
            initialization_successful = False # Treat init error as potentially critical
            # Clean up any partially loaded components for this symbol
            if symbol in models: del models[symbol]
            if symbol in feature_pipelines: del feature_pipelines[symbol]
            if symbol in agents: del agents[symbol]
            if symbol in active_symbols: active_symbols.remove(symbol) # Remove if partially added
            # Break if you want to stop entirely on first error, or continue to log all issues
            # break

    # --- Final Check and Start Notification ---
    if not active_symbols:
        logger.critical("Failed to initialize components for ANY symbols. Exiting live mode.")
        error_msg = "Live Startup failed: No symbols were successfully initialized."
        if missing_components: error_msg += f" Issues: {dict(missing_components)}"
        send_email_notification("GenovoTraderV2 CRITICAL ERROR", error_msg, config)
        if mt5.terminal_info(): mt5.shutdown();
        return

    logger.info(f"Live trading components initialized. Active symbols: {active_symbols}")
    start_notification_body = f"GenovoTraderV2 live trading started.\nTrading Symbols: {', '.join(active_symbols)}\nAccount: {config.get('mt5_config',{}).get('login')}"
    if missing_components:
        missing_str = "; ".join([f"{s}: {', '.join(c)}" for s, c in missing_components.items()])
        start_notification_body += f"\nNote: Issues prevented trading for other symbols: {missing_str}"
    send_email_notification("GenovoTraderV2 Live Trading Started", start_notification_body, config)


    # --- Live Trading Loop ---
    seq_length = config.get('model_config', {}).get('seq_length', 100)
    live_config = config.get('live_config', {})
    loop_interval_sec = live_config.get('loop_interval_sec', 5) # Default 5 seconds
    timeframe_str = config.get('data_config', {}).get('timeframe', 'M1')
    timeframe_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
    mt5_timeframe = timeframe_map.get(timeframe_str.upper())
    if not mt5_timeframe:
         logger.error(f"Invalid timeframe '{timeframe_str}' for live trading. Defaulting to M1.")
         mt5_timeframe = mt5.TIMEFRAME_M1

    logger.info(f"Starting multi-symbol live trading loop. Interval: {loop_interval_sec}s, Timeframe: {timeframe_str}")
    last_print_time = time.time()
    print_interval = 60 # Print summary every 60 seconds

    try:
        while True:
            loop_start_time = time.time()

            # Check MT5 connection status periodically
            if not mt5.terminal_info() or not mt5.terminal_info().connected:
                 logger.error("MT5 connection lost. Attempting to reconnect...")
                 send_email_notification("GenovoTraderV2 Warning", "MT5 connection lost. Attempting reconnect.", config)
                 if not initialize_mt5_connection(config, logger):
                      logger.critical("Failed to re-establish MT5 connection. Stopping live trading.")
                      send_email_notification("GenovoTraderV2 CRITICAL ERROR", "Failed to reconnect to MT5. Live trading stopped.", config)
                      break # Exit the loop
                 else:
                      logger.info("Successfully reconnected to MT5.")
                      send_email_notification("GenovoTraderV2 Info", "Successfully reconnected to MT5.", config)


            potential_trades = [] # Reset potential trades each loop

            # --- Update Portfolio Risk ---
            try:
                all_open_positions = mt5_interface.get_positions() # Get all positions
                open_positions_info = {}
                if all_open_positions:
                     # Convert position info for portfolio manager format
                     for pos in all_open_positions:
                          symbol = pos['symbol']
                          # Only process positions for symbols we are actively trading
                          if symbol not in active_symbols: continue
                          symbol_info = mt5_interface.get_symbol_info(symbol) # Uses cache
                          if symbol_info:
                               pos_details = pos.copy()
                               pos_details['volume_lots'] = pos['volume']
                               pos_details['entry_price'] = pos['price_open']
                               pos_details['sl_price'] = pos['sl'] # SL price level
                               pos_details['contract_size'] = symbol_info.get('trade_contract_size', 100000)
                               open_positions_info[symbol] = pos_details
                          else:
                               # Should not happen if symbol is active, but log just in case
                               logger.warning(f"Could not get symbol info for open position {symbol} (Ticket: {pos['ticket']}) even though it's active.")
                # Update manager with risk info for *currently active* symbols
                portfolio_manager.update_open_risk(open_positions_info)
                # logger.debug(f"Current total portfolio risk: {portfolio_manager.get_total_risk_pct()*100:.2f}%")
            except Exception as e:
                logger.error(f"Error updating portfolio risk: {e}", exc_info=True)


            # --- Iterate through symbols for signals ---
            for symbol in active_symbols:
                # Get necessary components for this symbol
                model = models[symbol]
                agent = agents[symbol]
                fp = feature_pipelines[symbol]

                # 1. Get Market Data
                current_state_df = mt5_interface.get_recent_bars(symbol, mt5_timeframe, seq_length)
                if current_state_df is None or len(current_state_df) < seq_length:
                    logger.warning(f"Not enough bars received for {symbol} ({len(current_state_df) if current_state_df is not None else 0} < {seq_length}). Skipping signal generation.")
                    continue

                # 2. Create State Features
                try:
                    features_df = fp.transform(current_state_df) # Use the loaded pipeline/scaler
                    if features_df.empty or features_df.isnull().values.any():
                         logger.warning(f"Feature transformation returned empty or NaN for {symbol}. Skipping.")
                         continue
                    state_array = features_df.values
                    state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(device)
                except Exception as e:
                    logger.error(f"Error creating features/state for {symbol}: {e}", exc_info=True)
                    continue # Skip this symbol on error

                # 3. Get Action/Signals from Model/Agent
                try:
                    # Use agent's select_action which uses model internally
                    action, _, _, _, _, model_outputs = agent.select_action(state_tensor)
                    # Action: 0=Hold, 1=Buy, 2=Sell

                    # Extract necessary info from model outputs for decision making
                    policy_logits = model_outputs.get('policy_logits', torch.zeros(1,3))
                    action_probs = torch.softmax(policy_logits, dim=-1)
                    confidence = action_probs.max().item()
                    stop_loss_pct = model_outputs.get('stop_loss', torch.tensor(0.01)).item() # Default 1% SL
                    take_profit_pct = model_outputs.get('take_profit', torch.tensor(0.02)).item() # Default 2% TP

                    logger.debug(f"{symbol}: Action={action}, Confidence={confidence:.3f}, SL%={stop_loss_pct:.4f}, TP%={take_profit_pct:.4f}")

                    # Add to potential trades if action is Buy/Sell and SL is valid
                    if (action == 1 or action == 2) and stop_loss_pct > 0:
                         potential_trades.append({
                              'symbol': symbol,
                              'action': action, # 1=Buy, 2=Sell
                              'signal_strength': confidence,
                              'stop_loss_pct': stop_loss_pct,
                              'take_profit_pct': take_profit_pct,
                              'model_outputs': model_outputs # Pass full outputs if needed later
                         })
                except Exception as e:
                    logger.error(f"Error during model inference/action selection for {symbol}: {e}", exc_info=True)
                    continue # Skip this symbol on error


            # --- Allocate Risk Capital ---
            allocated_risk = {}
            if potential_trades: # Only allocate if there are signals
                 try:
                      # Get current capital from portfolio manager for allocation calculation
                      current_capital_for_alloc = portfolio_manager.get_current_capital()
                      if current_capital_for_alloc <= 0:
                           logger.warning("Cannot allocate risk, current capital is zero or negative.")
                      else:
                           allocated_risk = portfolio_manager.allocate_risk_capital(potential_trades)
                           if allocated_risk: logger.info(f"Risk Allocation: { {s: f'${rc:.2f}' for s, rc in allocated_risk.items()} }")
                 except Exception as e:
                      logger.error(f"Error allocating risk capital: {e}", exc_info=True)


            # --- Execute Trades based on Allocation ---
            if allocated_risk:
                 account_info = mt5_interface.get_account_summary()
                 account_currency = account_info.get('currency', 'USD') if account_info else 'USD'

                 for symbol, risk_capital in allocated_risk.items():
                     if risk_capital <= 0: continue # Skip if no capital allocated

                     # Find the corresponding signal
                     trade_signal = next((t for t in potential_trades if t['symbol'] == symbol), None)
                     if not trade_signal: continue # Should not happen if allocation exists

                     # Get current price for execution sizing and SL/TP calculation
                     current_tick = mt5_interface.get_tick(symbol)
                     # Use Ask for Buy SL/TP calc, Bid for Sell SL/TP calc
                     price_for_calc = current_tick.get('ask') if trade_signal['action'] == 1 else current_tick.get('bid')
                     if not current_tick or price_for_calc <= 0:
                          logger.warning(f"No valid tick/price ({price_for_calc}) for {symbol}. Skipping trade execution.")
                          continue

                     # Calculate Notional Size and Lots
                     sl_pct = trade_signal['stop_loss_pct']
                     notional_size = portfolio_manager.calculate_position_size_notional(
                          symbol, risk_capital, price_for_calc, sl_pct
                     )
                     if notional_size < portfolio_manager.min_position_notional: # Double check min notional
                          logger.info(f"[{symbol}] Calculated notional size ${notional_size:.2f} below minimum ${portfolio_manager.min_position_notional}. No trade.")
                          continue

                     # Use account currency for lot calculation
                     position_size_lots = mt5_interface.calculate_lots(symbol, notional_size, account_currency)
                     if position_size_lots <= 0:
                          logger.info(f"[{symbol}] Calculated lot size is zero or negative ({position_size_lots}). No trade.")
                          continue

                     logger.info(f"Attempting Trade Execution for {symbol}: Action={trade_signal['action']}, Lots={position_size_lots:.2f}, Risk Capital=${risk_capital:.2f}, Notional=${notional_size:.2f}")

                     # Check current position for the symbol
                     current_position_info = mt5_interface.get_position_info(symbol)
                     current_position_lots = current_position_info['volume'] if current_position_info else 0.0
                     current_position_type = current_position_info['type'] if current_position_info else None # 0:Buy, 1:Sell

                     # --- Execute Buy/Sell Logic ---
                     try:
                         action_type = 'BUY' if trade_signal['action'] == 1 else 'SELL'
                         tp_pct = trade_signal['take_profit_pct']
                         delay_after_close_ms = live_config.get('delay_after_close_ms', 500) # Get delay from config

                         # Calculate SL/TP levels based on current price
                         if action_type == 'BUY':
                             sl_price = price_for_calc * (1 - sl_pct)
                             tp_price = price_for_calc * (1 + tp_pct)
                             # Check if currently Short
                             if current_position_type == 1: # Position type 1 is Sell
                                 logger.info(f"[{symbol}] Closing existing Short (Ticket: {current_position_info['ticket']}) before opening Long.")
                                 close_result = mt5_interface.close_position(symbol, position_info=current_position_info)
                                 # Wait briefly only if close was successful before opening new
                                 if close_result and close_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                                      time.sleep(delay_after_close_ms / 1000.0) # Use configured delay
                                      logger.info(f"[{symbol}] Opening new Long ({position_size_lots:.2f} lots). SL={sl_price:.5f}, TP={tp_price:.5f}")
                                      open_result = mt5_interface.open_position(symbol, 'BUY', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                      if open_result and open_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                                           # Update capital manager after successful open (deduct estimated costs if needed)
                                           # cost = mt5_interface.calculate_commission(...) + mt5_interface.calculate_slippage(...)
                                           # portfolio_manager.update_capital(-cost)
                                           pass # Capital updated via PnL from broker ideally
                                 else:
                                      logger.error(f"[{symbol}] Failed to close existing Short position (RetCode: {close_result.get('retcode') if close_result else 'N/A'}). Cannot open Long.")
                             # Check if currently Flat
                             elif current_position_lots == 0:
                                 logger.info(f"[{symbol}] Opening new Long ({position_size_lots:.2f} lots). SL={sl_price:.5f}, TP={tp_price:.5f}")
                                 open_result = mt5_interface.open_position(symbol, 'BUY', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                 if open_result and open_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                                      # Update capital manager if needed
                                      pass
                             # Else: Already Long
                             else:
                                 logger.info(f"[{symbol}] Received Buy signal, but already Long (Ticket: {current_position_info['ticket']}). No action.")

                         elif action_type == 'SELL':
                             sl_price = price_for_calc * (1 + sl_pct)
                             tp_price = price_for_calc * (1 - tp_pct)
                             # Check if currently Long
                             if current_position_type == 0: # Position type 0 is Buy
                                 logger.info(f"[{symbol}] Closing existing Long (Ticket: {current_position_info['ticket']}) before opening Short.")
                                 close_result = mt5_interface.close_position(symbol, position_info=current_position_info)
                                 # Wait briefly only if close was successful
                                 if close_result and close_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                                      time.sleep(delay_after_close_ms / 1000.0)
                                      logger.info(f"[{symbol}] Opening new Short ({position_size_lots:.2f} lots). SL={sl_price:.5f}, TP={tp_price:.5f}")
                                      open_result = mt5_interface.open_position(symbol, 'SELL', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                      if open_result and open_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                                           # Update capital manager if needed
                                           pass
                                 else:
                                      logger.error(f"[{symbol}] Failed to close existing Long position (RetCode: {close_result.get('retcode') if close_result else 'N/A'}). Cannot open Short.")
                             # Check if currently Flat
                             elif current_position_lots == 0:
                                 logger.info(f"[{symbol}] Opening new Short ({position_size_lots:.2f} lots). SL={sl_price:.5f}, TP={tp_price:.5f}")
                                 open_result = mt5_interface.open_position(symbol, 'SELL', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                 if open_result and open_result['retcode'] == mt5.TRADE_RETCODE_DONE:
                                      # Update capital manager if needed
                                      pass
                             # Else: Already Short
                             else:
                                 logger.info(f"[{symbol}] Received Sell signal, but already Short (Ticket: {current_position_info['ticket']}). No action.")

                     except Exception as e:
                         logger.error(f"Error during trade execution logic for {symbol}: {e}", exc_info=True)
                         error_details = f"Error during trade execution for {symbol}:\n{traceback.format_exc()}"
                         # Send notification for trade execution errors
                         send_email_notification(f"GenovoTraderV2 Trade Error ({symbol})", error_details, config)


            # --- Loop Timing ---
            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            sleep_time = max(0, loop_interval_sec - elapsed_time)

            # Print summary periodically
            if loop_end_time - last_print_time >= print_interval:
                 # Update capital from account before printing summary
                 account_summary = mt5_interface.get_account_summary()
                 if account_summary and account_summary.get('balance') is not None:
                      portfolio_manager.current_capital = account_summary['balance']

                 current_capital = portfolio_manager.get_current_capital()
                 current_risk_pct = portfolio_manager.get_total_risk_pct() * 100
                 logger.info(f"Live Summary: Capital=${current_capital:.2f}, Equity=${account_summary.get('equity', 0.0):.2f}, Risk={current_risk_pct:.2f}%, Loop Time={elapsed_time:.3f}s, Sleep={sleep_time:.3f}s")
                 last_print_time = loop_end_time


            if sleep_time > 0:
                 time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Live trading loop interrupted by user (Ctrl+C).")
        send_email_notification("GenovoTraderV2 Stopped", "Live trading loop stopped by user.", config)
    except Exception as e:
        logger.critical(f"Fatal error in live trading loop: {e}", exc_info=True)
        error_details = f"Fatal error in live trading loop:\n{traceback.format_exc()}"
        send_email_notification("GenovoTraderV2 FATAL ERROR", error_details, config)
    finally:
        logger.info("Shutting down live trading...")
        # Optional: Attempt to close all open positions on shutdown?
        # close_all_positions(mt5_interface, active_symbols, logger)
        if mt5.terminal_info():
            mt5.shutdown()
            logger.info("MetaTrader 5 connection shut down.")
        send_email_notification("GenovoTraderV2 Stopped", "Live trading process has shut down.", config)
        logger.info("--- Live Trading Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Genovo Trader v2 - Multi Symbol")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, choices=['simulation', 'live', 'train'], help='Override the mode from the config file.')
    args = parser.parse_args()
    config, logger = None, None # Initialize

    try:
        config = load_config(args.config)
        # Setup logger first
        logger = setup_logger(config.get('logging_config', {}))
        logger.info(f"--- Starting Genovo Trader v2 (Multi-Symbol) ---")

        # Determine mode
        mode = args.mode if args.mode else config.get('mode', 'simulation')
        config['mode'] = mode # Ensure mode is set in config dict for other modules
        logger.info(f"Mode selected: {mode}")

        # Log device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Send startup notification
        send_email_notification("GenovoTraderV2 Started", f"Process started in '{mode}' mode.", config)

        # Execute selected mode
        if mode == 'simulation':
            run_simulation_or_training(config, logger)
        elif mode == 'train':
            logger.info("Running Training mode (equivalent to simulation for now).")
            run_simulation_or_training(config, logger)
        elif mode == 'live':
            run_live(config, logger)
        else:
            logger.error(f"Invalid mode '{mode}' specified.")
            send_email_notification("GenovoTraderV2 Error", f"Startup failed: Invalid mode '{mode}'.", config)

    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found. {e}")
        # Logger might not be initialized yet
        if logger: logger.critical(f"Configuration file not found: {e}")
    except Exception as e:
        print(f"FATAL ERROR during execution: {e}")
        print(traceback.format_exc())
        if logger: logger.critical(f"FATAL ERROR during execution: {e}", exc_info=True)
        # Send notification if config was loaded
        if config:
            error_details = f"Fatal error during execution:\n{traceback.format_exc()}"
            send_email_notification("GenovoTraderV2 FATAL ERROR", error_details, config)
    finally:
        if logger: logger.info("--- Genovo Trader v2 finished ---")
        else: print("--- Genovo Trader v2 finished ---")
        # Final check for MT5 shutdown, especially if live mode failed early
        if mt5.terminal_info():
            print("Final MT5 shutdown check.")
            mt5.shutdown()

if __name__ == "__main__":
    main()
