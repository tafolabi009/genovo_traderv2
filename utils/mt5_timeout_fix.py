# MT5 timeout fix
import os
import signal
import threading
import traceback
import time

def extend_ipc_timeout():
    """
    Extend the IPC timeout for MetaTrader 5 connections
    Add to broker/metatrader_interface.py to use
    """
    try:
        # Set multiple environment variables to extend timeout
        os.environ['MT5IPC_TIMEOUT'] = '300000'  # 5 minutes
        os.environ['MT5_TIMEOUT'] = '300000'  # 5 minutes
        os.environ['MT5_CONNECT_TIMEOUT'] = '300000'  # 5 minutes
        
        # Add watchdog to prevent hanging
        def watchdog_timer():
            print("MT5 watchdog timer started")
            while True:
                time.sleep(60)  # Check every minute
                try:
                    import MetaTrader5 as mt5
                    if mt5.terminal_info() is None:
                        print("MT5 connection lost, attempting to reconnect...")
                        mt5.shutdown()
                        time.sleep(5)
                except:
                    pass
        
        # Start watchdog in separate thread
        watchdog = threading.Thread(target=watchdog_timer)
        watchdog.daemon = True
        watchdog.start()
        
        # Ignore SIGPIPE errors which can cause issues with MT5 on some systems
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        except:
            pass
            
        return True
    except Exception as e:
        print(f"Error setting up MT5 timeout extension: {e}")
        traceback.print_exc()
        return False
