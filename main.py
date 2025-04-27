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

# Import necessary components from the project structure
from core.simulator import create_simulator
from utils.logger import setup_logger
from broker.metatrader_interface import MetaTraderInterface, create_mt5_interface
from core.features import create_feature_pipeline # FeaturePipeline now handles scaler save/load via extractor
from core.model import create_model
from core.strategy import create_strategy
from core.portfolio_compound import PortfolioCapitalManager, create_portfolio_capital_manager
from utils.notifier import send_email_notification


# --- load_config, initialize_mt5, load_data_for_symbols remain the same ---
# (Code omitted for brevity - use version from genovo_main_v4)
def load_config(config_path='configs/params.yaml'):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")
    return config

def initialize_mt5(config):
    """Initializes connection to the MetaTrader 5 terminal."""
    if mt5.terminal_info(): return True # Already initialized
    mt5_config = config.get('mt5_config', {})
    login, password, server, path = mt5_config.get('login'), mt5_config.get('password'), mt5_config.get('server'), mt5_config.get('path')
    if not path or not os.path.exists(path):
        print(f"Error: MT5 path not found/specified: {path}"); return False
    print(f"Initializing MT5 from path: {path}...")
    if not mt5.initialize(path=path, login=login, password=password, server=server, timeout=10000):
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}"); mt5.shutdown(); return False
    print(f"MT5 initialized successfully. Version: {mt5.version()}")
    account_info = mt5.account_info()
    if account_info: print(f"Connected to account: {account_info.login} on {account_info.server}")
    else: print(f"Failed to get account info, error code = {mt5.last_error()}"); mt5.shutdown(); return False
    return True

def load_data_for_symbols(config, symbols):
    """Loads historical data for a list of symbols from MT5."""
    all_data = {}
    data_config = config.get('data_config', {})
    timeframe_str = data_config.get('timeframe', 'M1')
    num_bars = data_config.get('num_bars', 100000)
    timeframe_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1}
    timeframe = timeframe_map.get(timeframe_str.upper())
    if timeframe is None: raise ValueError(f"Invalid timeframe '{timeframe_str}'.")
    if not mt5.terminal_info():
        if not initialize_mt5(config): raise ConnectionError("Failed to initialize MT5.")
    for symbol in symbols:
        print(f"Loading {num_bars} bars of {symbol} {timeframe_str}...")
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None: print(f"Symbol {symbol} not found. Skipping."); continue
        if not symbol_info.visible:
            print(f"Symbol {symbol} not visible, enabling...");
            if not mt5.symbol_select(symbol, True): print(f"mt5.symbol_select failed. Skipping."); continue
            time.sleep(1)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0: print(f"No data for {symbol}. Skipping."); continue
        data = pd.DataFrame(rates); data['time'] = pd.to_datetime(data['time'], unit='s'); data = data.set_index('time')
        data.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'}, inplace=True)
        all_data[symbol] = data[['open', 'high', 'low', 'close', 'volume']]; print(f"Loaded {len(data)} bars for {symbol}.")
    return all_data
# --- End of omitted section ---


def run_simulation_or_training(config, logger):
    """
    Runs the trading simulation/training independently for each configured symbol.
    Saves the fitted scaler state for each symbol after training.
    """
    logger.info("--- Starting Multi-Symbol Simulation/Training Mode ---")
    symbols = config.get('symbols', [])
    if not symbols:
        logger.error("No symbols specified in configuration ('symbols' list). Exiting.")
        send_email_notification("GenovoTraderV2 Error", "Startup failed: No symbols specified in config.", config)
        return

    try:
        all_historical_data = load_data_for_symbols(config, symbols)
        if not all_historical_data:
            logger.error("Failed to load data for any symbols. Exiting.")
            send_email_notification("GenovoTraderV2 Error", "Startup failed: Could not load data for any symbols from MT5.", config)
            return
    except Exception as e:
        logger.error(f"Error during multi-symbol data loading: {e}", exc_info=True)
        error_details = f"Error during data loading:\n{traceback.format_exc()}"
        send_email_notification("GenovoTraderV2 Error", f"Startup failed: {error_details}", config)
        return

    results_dir = config.get('results_dir', 'results/')
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")

    all_results = {}
    errors_occurred = False
    feature_pipelines = {} # Store fitted pipelines to save scalers

    for symbol in symbols:
        if symbol not in all_historical_data:
            logger.warning(f"Skipping simulation for {symbol} due to missing data.")
            continue

        logger.info(f"--- Starting Simulation/Training for {symbol} ---")
        symbol_data = all_historical_data[symbol]
        symbol_config = config.copy()
        symbol_config['symbol'] = symbol

        model_cfg = symbol_config.get('model_config', {})
        load_base = model_cfg.get('load_path_base', results_dir)
        save_base = model_cfg.get('save_path_base', results_dir)
        model_cfg['load_path'] = os.path.join(load_base, f"{symbol}_model.pth") if load_base else None
        model_cfg['save_path'] = os.path.join(save_base, f"{symbol}_checkpoint.pth")

        simulator = None # Define simulator outside try block
        try:
            symbol_config['capital_config'] = config.get('capital_config', {})
            symbol_config['capital_config']['initial_capital'] = config.get('portfolio_capital_config',{}).get('initial_capital', 20.0)

            simulator = create_simulator(data=symbol_data, config=symbol_config)
            # Store the feature pipeline instance from the simulator after creation
            if hasattr(simulator, 'feature_pipeline'):
                 feature_pipelines[symbol] = simulator.feature_pipeline
                 logger.info(f"TradingSimulator and FeaturePipeline created for {symbol}.")
            else:
                 logger.error(f"Simulator for {symbol} does not have 'feature_pipeline' attribute.")
                 errors_occurred = True
                 continue

        except Exception as e:
            logger.error(f"Error creating TradingSimulator for {symbol}: {e}", exc_info=True)
            errors_occurred = True
            continue

        try:
            results = simulator.run()
            all_results[symbol] = results
            logger.info(f"Simulation run completed for {symbol}.")
        except Exception as e:
            logger.error(f"Error during simulation run for {symbol}: {e}", exc_info=True)
            all_results[symbol] = simulator.get_results()
            errors_occurred = True

        # --- Save Scaler State ---
        try:
            if symbol in feature_pipelines and feature_pipelines[symbol].feature_extractor.is_fitted:
                 scaler_path = os.path.join(results_dir, f"{symbol}_scaler.joblib")
                 feature_pipelines[symbol].save_scaler(scaler_path) # Use pipeline's save method
            else:
                 logger.warning(f"Scaler for {symbol} was not fitted or pipeline not found. Cannot save scaler state.")
        except Exception as e:
             logger.error(f"Error saving scaler state for {symbol}: {e}", exc_info=True)
             errors_occurred = True
        # --- End Save Scaler State ---


        # Save results history
        try:
            history_df = results.get('history')
            if history_df is not None and not history_df.empty:
                history_path = os.path.join(results_dir, f"{symbol}_history.csv")
                history_df.to_csv(history_path)
                logger.info(f"Simulation history saved for {symbol} to {history_path}")
            else: logger.warning(f"Simulation history empty/missing for {symbol}.")
        except Exception as e:
            logger.error(f"Error saving history for {symbol}: {e}", exc_info=True); errors_occurred = True

        # Save final model weights
        final_model_path = os.path.join(save_base, f"{symbol}_final_model.pth")
        try:
            if hasattr(simulator, 'agent') and simulator.agent:
                 simulator.agent.save_model(final_model_path)
                 logger.info(f"Final model weights saved for {symbol} to {final_model_path}")
            else: logger.warning(f"Simulator/agent not available for {symbol}, cannot save model.")
        except Exception as e:
            logger.error(f"Could not save final model for {symbol}: {e}", exc_info=True); errors_occurred = True

        # Plot results
        if symbol_config.get('simulator_config', {}).get('plot_results', False):
            try:
                if hasattr(simulator, 'plot_results') and results.get('history') is not None:
                     simulator.plot_results(results); logger.info(f"Results plotted for {symbol}.")
                else: logger.warning(f"Cannot plot results for {symbol} - simulator or history missing.")
            except Exception as e: logger.error(f"Could not plot results for {symbol}: {e}", exc_info=True)

        logger.info(f"--- Finished Simulation/Training for {symbol} ---")

    # --- Aggregate Results & Final Notification ---
    # (Summary logic remains the same)
    summary_lines = ["--- Overall Simulation Summary ---"]
    summary_lines.append(f"Simulated Symbols: {list(all_results.keys())}")
    if all_results:
        total_initial_capital = config.get('portfolio_capital_config',{}).get('initial_capital', 20.0) * len(all_results)
        total_final_capital = sum(res.get('final_capital', 0) for res in all_results.values())
        summary_lines.append(f"Total Initial Capital (Summed): ${total_initial_capital:.2f}")
        summary_lines.append(f"Total Final Capital (Summed):   ${total_final_capital:.2f}")
        if total_initial_capital > 0:
             total_return = ((total_final_capital - total_initial_capital) / total_initial_capital) * 100
             summary_lines.append(f"Overall Return (Simple Sum):    {total_return:.2f}%")
    summary_lines.append("---------------------------------")
    summary_message = "\n".join(summary_lines)
    logger.info(summary_message)
    email_subject = "GenovoTraderV2 Simulation/Training Complete"
    email_body = f"Simulation/Training process finished.\n\n{summary_message}"
    if errors_occurred: email_subject += " (with errors)"; email_body += "\n\nNOTE: Errors occurred during the process. Please check the logs."
    send_email_notification(email_subject, email_body, config)
    logger.info("--- Multi-Symbol Simulation/Training Finished ---")


def run_live(config, logger):
    """Runs the live trading bot for multiple symbols, loading scaler state."""
    logger.info("--- Starting Multi-Symbol Live Trading Mode ---")
    symbols = config.get('symbols', [])
    if not symbols:
        logger.error("No symbols specified. Exiting."); send_email_notification("GenovoTraderV2 Error", "Live Startup failed: No symbols specified.", config); return

    if not initialize_mt5(config):
        logger.error("Failed to initialize MT5. Exiting."); send_email_notification("GenovoTraderV2 Error", "Live Startup failed: Could not initialize MT5.", config); return

    try:
        mt5_interface = create_mt5_interface(config); logger.info("MetaTraderInterface created.")
    except Exception as e:
        logger.error(f"Error creating MT5 interface: {e}", exc_info=True); send_email_notification("GenovoTraderV2 Error", f"Live Startup failed: Error creating MT5 interface:\n{traceback.format_exc()}", config); mt5.shutdown(); return

    try:
        portfolio_manager = create_portfolio_capital_manager(config)
        account_summary = mt5_interface.get_account_summary()
        if account_summary: portfolio_manager.current_capital = account_summary['balance']; logger.info(f"Portfolio manager synced: ${portfolio_manager.current_capital:.2f}")
        else: logger.warning("Could not get account summary.")
    except Exception as e:
        logger.error(f"Error creating Portfolio Manager: {e}", exc_info=True); send_email_notification("GenovoTraderV2 Error", f"Live Startup failed: Error creating Portfolio Manager:\n{traceback.format_exc()}", config); mt5.shutdown(); return

    # --- Initialize components FOR EACH SYMBOL ---
    models = {}
    feature_pipelines = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cfg = config.get('model_config', {})
    results_dir = config.get('results_dir', 'results/') # Use results_dir for loading scaler/model
    load_base = model_cfg.get('load_path_base', results_dir)
    logger.info("Initializing models and feature pipelines...")
    initialization_successful = True
    missing_components = defaultdict(list) # Store missing components per symbol

    for symbol in symbols:
        try:
            # Feature Pipeline & Load Scaler
            fp = create_feature_pipeline(config.get('feature_config', {}))
            scaler_path = os.path.join(results_dir, f"{symbol}_scaler.joblib")
            fp.load_scaler(scaler_path) # Attempt to load scaler state
            if not fp.feature_extractor.is_fitted:
                 logger.warning(f"Scaler state not loaded successfully for {symbol}. Features will be unscaled.")
                 missing_components[symbol].append("Scaler")
                 # Decide if you want to continue without scaler or skip symbol
                 # continue # Option: Skip symbol if scaler is missing
            feature_pipelines[symbol] = fp

            # Model
            model = create_model(model_cfg)
            model_path = os.path.join(load_base, f"{symbol}_final_model.pth")
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models[symbol] = model
                logger.info(f"Loaded model for {symbol} from {model_path}")
            else:
                logger.warning(f"Model file not found for {symbol} at {model_path}. This symbol will not be traded.")
                missing_components[symbol].append("Model")
                if symbol in feature_pipelines: del feature_pipelines[symbol] # Remove pipeline if model is missing
                continue # Skip this symbol if model missing

        except Exception as e:
            logger.error(f"Error initializing components for symbol {symbol}: {e}", exc_info=True)
            missing_components[symbol].append("Initialization Error")
            initialization_successful = False # Treat init error as potentially critical
            # Break if you want to stop entirely on first error, or continue to log all issues
            # break

    active_symbols = list(models.keys())
    if not active_symbols: # Check if any models were loaded successfully
        logger.error("Failed to initialize components for ANY symbols. Exiting live mode.")
        error_msg = "Live Startup failed: No models loaded."
        if missing_components: error_msg += f" Issues: {dict(missing_components)}"
        send_email_notification("GenovoTraderV2 Error", error_msg, config)
        mt5.shutdown(); return

    logger.info(f"Live trading components initialized for active symbols: {active_symbols}")
    start_notification_body = f"GenovoTraderV2 live trading started.\nTrading Symbols: {', '.join(active_symbols)}\nAccount: {config.get('mt5_config',{}).get('login')}"
    if missing_components:
        missing_str = "; ".join([f"{s}: {', '.join(c)}" for s, c in missing_components.items()])
        start_notification_body += f"\nNote: Issues for non-active symbols: {missing_str}"
    send_email_notification("GenovoTraderV2 Live Trading Started", start_notification_body, config)


    # --- Live Trading Loop ---
    # (Loop logic remains the same as genovo_main_v4 - ensure feature_pipelines[symbol] is used)
    # ... (rest of run_live loop code from genovo_main_v4) ...
    seq_length = config.get('model_config', {}).get('seq_length', 100)
    live_config = config.get('live_config', {})
    loop_interval_sec = live_config.get('loop_interval_sec', 5)
    timeframe_str = config.get('data_config', {}).get('timeframe', 'M1')
    timeframe_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'H1': mt5.TIMEFRAME_H1}
    mt5_timeframe = timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M1)

    logger.info(f"Starting multi-symbol live trading loop. Interval: {loop_interval_sec}s")
    try:
        while True:
            start_time = time.time()
            potential_trades = []

            # Update Portfolio Risk
            try:
                all_open_positions = mt5_interface.get_positions()
                open_positions_info = {}
                if all_open_positions:
                     for pos in all_open_positions:
                          symbol = pos['symbol']; symbol_info = mt5_interface.get_symbol_info(symbol)
                          if symbol_info:
                               pos_details = pos.copy(); pos_details['volume_lots'] = pos['volume']; pos_details['entry_price'] = pos['price_open']
                               pos_details['sl_price'] = pos['sl']; pos_details['contract_size'] = symbol_info.get('trade_contract_size', 100000)
                               open_positions_info[symbol] = pos_details
                          else: logger.warning(f"Could not get symbol info for open position {symbol}")
                portfolio_manager.update_open_risk(open_positions_info)
                logger.debug(f"Current total portfolio risk: {portfolio_manager.get_total_risk_pct()*100:.2f}%")
            except Exception as e: logger.error(f"Error updating portfolio risk: {e}", exc_info=True)

            # Iterate through symbols for signals
            for symbol in active_symbols:
                model = models[symbol]
                fp = feature_pipelines[symbol] # Use the loaded pipeline

                current_state_df = mt5_interface.get_recent_bars(symbol, mt5_timeframe, seq_length)
                if current_state_df is None or len(current_state_df) < seq_length: logger.warning(f"Not enough bars for {symbol}. Skipping."); continue

                try: # Features/State Tensor
                    features_df = fp.transform(current_state_df) # Use the loaded pipeline/scaler
                    if features_df.empty: logger.warning(f"Feature transformation returned empty for {symbol}. Skipping."); continue
                    state_array = features_df.values
                    state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(device)
                except Exception as e: logger.error(f"Error creating features/state for {symbol}: {e}", exc_info=True); continue

                try: # Model Prediction
                    with torch.no_grad(): model_outputs = model(state_tensor)
                    action_probs = torch.softmax(model_outputs['policy_logits'], dim=-1); action = torch.argmax(action_probs).item()
                    confidence = action_probs.max().item(); stop_loss_pct = model_outputs.get('stop_loss', torch.tensor(0.01)).item()
                    logger.debug(f"{symbol}: Action={action}, Confidence={confidence:.3f}, SL%={stop_loss_pct:.4f}")
                    if (action == 1 or action == 2) and stop_loss_pct > 0:
                         potential_trades.append({'symbol': symbol, 'action': action, 'signal_strength': confidence,'stop_loss_pct': stop_loss_pct,'take_profit_pct': model_outputs.get('take_profit', torch.tensor(0.02)).item(),'model_outputs': model_outputs})
                except Exception as e: logger.error(f"Error during model inference for {symbol}: {e}", exc_info=True); continue

            # Allocate Risk Capital
            try: allocated_risk = portfolio_manager.allocate_risk_capital(potential_trades);
            except Exception as e: logger.error(f"Error allocating risk capital: {e}", exc_info=True); allocated_risk = {}
            if allocated_risk: logger.info(f"Risk Allocation: {allocated_risk}")

            # Execute Trades
            if allocated_risk:
                 for symbol, risk_capital in allocated_risk.items():
                     # ... (rest of trade execution logic from genovo_main_v4, ensuring correct indentation) ...
                     if risk_capital <= 0: continue
                     trade_signal = next((t for t in potential_trades if t['symbol'] == symbol), None)
                     if not trade_signal: continue
                     current_tick = mt5_interface.get_tick(symbol)
                     if not current_tick or current_tick.get('last', 0) <= 0: logger.warning(f"No valid tick for {symbol}. Skipping trade."); continue
                     current_price = current_tick['last']
                     sl_pct = trade_signal['stop_loss_pct']
                     notional_size = portfolio_manager.calculate_position_size_notional(symbol, risk_capital, current_price, sl_pct)
                     if notional_size <= 0: continue
                     position_size_lots = mt5_interface.calculate_lots(symbol, notional_size, portfolio_manager.get_current_capital())
                     if position_size_lots <= 0: logger.info(f"Lot size zero for {symbol}. No trade."); continue
                     logger.info(f"Trade Execution for {symbol}: Action={trade_signal['action']}, Lots={position_size_lots:.2f}, Risk Capital=${risk_capital:.2f}")
                     current_position_info = mt5_interface.get_position_info(symbol)
                     current_position_lots = current_position_info['volume'] if current_position_info else 0.0
                     current_position_type = current_position_info['type'] if current_position_info else None
                     try:
                         action_type = 'BUY' if trade_signal['action'] == 1 else 'SELL'; tp_pct = trade_signal['take_profit_pct']
                         if action_type == 'BUY':
                             sl_price = current_price * (1 - sl_pct); tp_price = current_price * (1 + tp_pct)
                             if current_position_type == 1:
                                 logger.info(f"[{symbol}] Closing Short before opening Long."); close_result = mt5_interface.close_position(symbol, position_info=current_position_info)
                                 if close_result and close_result['retcode'] == mt5.TRADE_RETCODE_DONE: time.sleep(1); mt5_interface.open_position(symbol, 'BUY', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                 else: logger.error(f"[{symbol}] Failed to close short.")
                             elif current_position_lots == 0: logger.info(f"[{symbol}] Opening Long ({position_size_lots:.2f} lots)."); mt5_interface.open_position(symbol, 'BUY', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                             else: logger.info(f"[{symbol}] Buy signal, but already Long.")
                         elif action_type == 'SELL':
                             sl_price = current_price * (1 + sl_pct); tp_price = current_price * (1 - tp_pct)
                             if current_position_type == 0:
                                 logger.info(f"[{symbol}] Closing Long before opening Short."); close_result = mt5_interface.close_position(symbol, position_info=current_position_info)
                                 if close_result and close_result['retcode'] == mt5.TRADE_RETCODE_DONE: time.sleep(1); mt5_interface.open_position(symbol, 'SELL', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                                 else: logger.error(f"[{symbol}] Failed to close long.")
                             elif current_position_lots == 0: logger.info(f"[{symbol}] Opening Short ({position_size_lots:.2f} lots)."); mt5_interface.open_position(symbol, 'SELL', position_size_lots, stop_loss=sl_price, take_profit=tp_price)
                             else: logger.info(f"[{symbol}] Sell signal, but already Short.")
                     except Exception as e: logger.error(f"Error during trade execution for {symbol}: {e}", exc_info=True); error_details = f"Error during trade execution for {symbol}:\n{traceback.format_exc()}"; send_email_notification(f"GenovoTraderV2 Trade Error ({symbol})", error_details, config)


            # Wait for the next loop iteration
            elapsed_time = time.time() - start_time
            sleep_time = max(0, loop_interval_sec - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt: logger.info("Live trading loop interrupted by user (Ctrl+C)."); send_email_notification("GenovoTraderV2 Stopped", "Live trading loop stopped by user.", config)
    except Exception as e: logger.error(f"Fatal error in live trading loop: {e}", exc_info=True); error_details = f"Fatal error in live trading loop:\n{traceback.format_exc()}"; send_email_notification("GenovoTraderV2 FATAL ERROR", error_details, config)
    finally: logger.info("Shutting down live trading."); send_email_notification("GenovoTraderV2 Stopped", "Live trading process has shut down.", config);
    if mt5.terminal_info(): mt5.shutdown(); logger.info("MetaTrader 5 connection shut down.")


def main():
    parser = argparse.ArgumentParser(description="Genovo Trader v2 - Multi Symbol")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, choices=['simulation', 'live', 'train'], help='Override the mode from the config file.')
    args = parser.parse_args()
    config, logger = None, None
    try:
        config = load_config(args.config)
        logger = setup_logger(config.get('logging_config', {}))
        logger.info(f"--- Starting Genovo Trader v2 (Multi-Symbol) ---")
        mode = args.mode if args.mode else config.get('mode', 'simulation')
        logger.info(f"Mode selected: {mode}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'; logger.info(f"Using device: {device}")
        send_email_notification("GenovoTraderV2 Started", f"Process started in '{mode}' mode.", config)
        if mode == 'simulation': run_simulation_or_training(config, logger)
        elif mode == 'train': logger.info("Running Training-Only mode."); run_simulation_or_training(config, logger)
        elif mode == 'live': run_live(config, logger)
        else: logger.error(f"Invalid mode '{mode}'."); send_email_notification("GenovoTraderV2 Error", f"Startup failed: Invalid mode '{mode}'.", config)
    except FileNotFoundError as e: print(f"ERROR: Configuration file not found. {e}")
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}"); print(traceback.format_exc())
        if logger: logger.critical(f"FATAL ERROR during startup: {e}", exc_info=True)
        if config: error_details = f"Fatal error during startup:\n{traceback.format_exc()}"; send_email_notification("GenovoTraderV2 FATAL STARTUP ERROR", error_details, config)
    finally:
        if logger: logger.info("--- Genovo Trader v2 finished ---")
        else: print("--- Genovo Trader v2 finished ---")
        if mt5.terminal_info(): print("Final MT5 shutdown check."); mt5.shutdown()

if __name__ == "__main__":
    main()

