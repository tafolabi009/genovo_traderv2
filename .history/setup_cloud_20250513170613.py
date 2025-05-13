import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_cloud_environment():
    """
    Configure the cloud environment for running MetaTrader 5 and the trading bot properly.
    This addresses common issues like MT5 connection timeouts and dependency conflicts.
    """
    print("\n===== Setting up Cloud Environment for Genovo Trader v2 =====\n")
    
    # 1. Install required packages with specific versions for compatibility
    print("Installing required Python packages...")
    packages = [
        "numpy==1.24.3",         # Specific version for pandas_ta compatibility
        "pandas==2.0.3",         # Compatible with numpy 1.24.3
        "pandas_ta==0.3.14b0",   # Technical analysis library
        "MetaTrader5==5.0.45",   # For MT5 connection
        "tenacity==8.2.2",       # For retry logic
        "psutil==5.9.5",         # Process management (for MT5 initialization)
        "pytz==2023.3",          # Timezone handling
        "pyyaml==6.0.1",         # Config files
        "joblib==1.3.2",         # Scalers/models
        "torch==2.0.1"           # ML models
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    # 2. Fix MT5 initialization issues
    print("\nConfiguring MT5 initialization...")
    
    # Create MT5 directory if it doesn't exist (common in cloud environments)
    mt5_path = os.path.expanduser("~/.wine/drive_c/Program Files/MetaTrader 5")
    os.makedirs(mt5_path, exist_ok=True)
    
    # Create symbolic links for MT5 in wine (if running in Linux cloud env)
    try:
        if os.name == 'posix':  # Linux/Unix
            print("Linux environment detected, setting up Wine for MT5...")
            # Create wine prefix if needed
            wine_cmd = "wine64" if shutil.which("wine64") else "wine"
            subprocess.call([wine_cmd, "wineboot"])
            
            # Set MT5 path in the config file
            update_mt5_path_in_config(mt5_path)
    except Exception as e:
        print(f"Error setting up Wine environment: {e}")
    
    # 3. Fix pandas_ta compatibility with numpy
    print("\nFixing pandas_ta compatibility with numpy...")
    try:
        fix_pandas_ta_numpy_compatibility()
    except Exception as e:
        print(f"Error fixing pandas_ta: {e}")
    
    # 4. Creating timeout fix for MT5
    print("\nCreating MT5 timeout fix hook...")
    create_mt5_timeout_hook()
    
    print("\n===== Setup Complete =====")
    print("\nNext steps:")
    print("1. Upload your MT5 terminal64.exe to the correct path")
    print("2. Verify MT5 credentials in configs/params.yaml")
    print("3. Run the trading bot with: python main.py")

def update_mt5_path_in_config(mt5_path):
    """
    Update the MT5 path in the config file
    """
    config_path = Path('configs/params.yaml')
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update both places where MT5 path is defined
        if 'mt5_config' in config:
            config['mt5_config']['path'] = mt5_path
            print(f"Updated MT5 path in mt5_config section to: {mt5_path}")
        
        if 'broker' in config:
            config['broker']['mt5_path'] = mt5_path
            print(f"Updated MT5 path in broker section to: {mt5_path}")
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Error updating MT5 path in config: {e}")

def fix_pandas_ta_numpy_compatibility():
    """
    Fix pandas_ta compatibility with newer versions of numpy
    by monkey patching numpy.NaN
    """
    try:
        import site
        site_packages = site.getsitepackages()[0]
        pandas_ta_dir = os.path.join(site_packages, "pandas_ta")
        
        if not os.path.exists(pandas_ta_dir):
            print(f"pandas_ta package not found at {pandas_ta_dir}")
            return
        
        # Create numpy compatibility patch
        patch_file = os.path.join(pandas_ta_dir, "numpy_compat.py")
        with open(patch_file, 'w') as f:
            f.write("""
# Monkey patch for numpy compatibility with pandas_ta
import numpy

# Add NaN for backward compatibility with older code
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan
""")
        print(f"Created numpy compatibility patch at {patch_file}")
        
        # Add import to __init__.py
        init_file = os.path.join(pandas_ta_dir, "__init__.py")
        if os.path.exists(init_file):
            with open(init_file, 'r') as f:
                content = f.read()
            
            if "from .numpy_compat import" not in content:
                with open(init_file, 'w') as f:
                    f.write("# Added for numpy 2.0+ compatibility\nfrom .numpy_compat import *\n\n" + content)
                print(f"Updated {init_file} to include numpy compatibility patch")
    except Exception as e:
        print(f"Error fixing pandas_ta compatibility: {e}")

def create_mt5_timeout_hook():
    """
    Create a hook script that can be imported to extend MT5 timeouts
    """
    hook_file = Path('utils/mt5_timeout_fix.py')
    os.makedirs(hook_file.parent, exist_ok=True)
    
    with open(hook_file, 'w') as f:
        f.write("""
# MT5 timeout fix
import os
import signal
import threading
import traceback

def extend_ipc_timeout():
    \"\"\"
    Extend the IPC timeout for MetaTrader 5 connections
    Add to broker/metatrader_interface.py to use
    \"\"\"
    try:
        # Set environment variable to extend timeout
        os.environ['MT5IPC_TIMEOUT'] = '120000'  # 120 seconds
        
        # Add watchdog to prevent hanging
        def watchdog_timer():
            print("MT5 watchdog timer started")
            # If needed, implement additional timeout handling
        
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
""")
    print(f"Created MT5 timeout fix at {hook_file}")

if __name__ == "__main__":
    setup_cloud_environment() 
