# utils/logger.py

import logging
import sys
import os
import atexit # To register shutdown function

# --- Global list to keep track of handlers ---
# This helps ensure we can close them properly on exit
_log_handlers = []

def setup_logger(config=None, default_level=logging.INFO):
    """
    Sets up the application logger with UTF-8 encoding and attempts
    graceful handler shutdown.

    Args:
        config (dict, optional): Logging configuration dictionary.
                                 Expected keys: 'log_level', 'log_file'.
        default_level (int, optional): Default logging level if not specified.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _log_handlers
    config = config or {}
    log_level_str = config.get('log_level', 'INFO').upper()
    log_file = config.get('log_file', 'genovo_traderv2.log') # Default log file name

    # Ensure results directory exists if log file path includes it
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        except OSError as e:
            print(f"Error creating log directory {log_dir}: {e}")
            log_file = os.path.basename(log_file) # Fallback to current dir
            print(f"Logging to fallback file: {log_file}")


    log_level = getattr(logging, log_level_str, default_level)

    # Get the specific logger instance
    logger = logging.getLogger("genovo_traderv2")
    logger.propagate = False # Prevent duplicate messages if root logger is configured elsewhere
    logger.setLevel(log_level)

    # --- IMPORTANT: Clear existing handlers before adding new ones ---
    # This prevents adding handlers multiple times if setup_logger is called again
    if logger.hasHandlers():
        # Close and remove existing handlers managed by this setup
        for handler in list(logger.handlers): # Iterate over a copy
            if handler in _log_handlers:
                 try:
                      handler.close()
                 except Exception:
                      pass # Ignore errors during close
                 logger.removeHandler(handler)
                 _log_handlers.remove(handler)
            # Optionally remove handlers not managed by this setup if needed,
            # but be careful not to interfere with other logging setups.
            # else:
            #     logger.removeHandler(handler)


    # Define log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')

    # --- Console Handler ---
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        _log_handlers.append(console_handler) # Track handler
    except Exception as e:
         print(f"Error setting up console handler: {e}")


    # --- File Handler ---
    try:
        # --- Explicitly use UTF-8 encoding ---
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        _log_handlers.append(file_handler) # Track handler
        logger.info(f"Logging initialized. Level: {log_level_str}. File: {log_file}")
    except Exception as e:
        # Use print here as logger might not be fully functional
        print(f"CRITICAL ERROR: Failed to set up file handler for {log_file}: {e}")
        logger.error(f"Failed to set up file handler for {log_file}: {e}", exc_info=True)


    return logger

def close_log_handlers():
    """Closes all tracked log handlers."""
    global _log_handlers
    # print("Attempting to close log handlers...") # Debug print
    for handler in _log_handlers:
        try:
            # Flush buffer before closing
            handler.flush()
            handler.close()
            # print(f"Closed handler: {handler}") # Debug print
        except Exception as e:
            print(f"Error closing log handler {handler}: {e}") # Use print as logger might be closed
    _log_handlers = [] # Clear the list

# --- Register the shutdown function ---
# This attempts to close handlers when the Python interpreter exits normally
# Note: May not always run if the process is killed abruptly (e.g., Task Manager kill)
atexit.register(close_log_handlers)

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Example config dictionary
    example_config = {
        'logging_config': {
            'log_level': 'DEBUG',
            'log_file': 'results/test_log.log'
        }
    }
    logger = setup_logger(example_config.get('logging_config'))
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    print("Log messages sent. Check results/test_log.log")
    # Handlers should be closed automatically on exit via atexit

