# utils/logger.py

import logging
import sys
import os

def setup_logger(config=None, default_level=logging.INFO):
    """
    Sets up the application logger.

    Args:
        config (dict, optional): Logging configuration dictionary.
                                 Expected keys: 'log_level', 'log_file'.
        default_level (int, optional): Default logging level if not specified.

    Returns:
        logging.Logger: Configured logger instance.
    """
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

    # Get the root logger
    logger = logging.getLogger("genovo_traderv2")
    logger.propagate = False # Prevent duplicate messages if root logger is configured elsewhere
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # --- File Handler ---
    try:
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized. Level: {log_level_str}. File: {log_file}")
    except Exception as e:
        logger.error(f"Failed to set up file handler for {log_file}: {e}", exc_info=True)


    return logger

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
