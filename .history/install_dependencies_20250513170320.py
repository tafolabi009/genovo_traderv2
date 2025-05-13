import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies for genovo_traderv2"""
    print("Installing dependencies for Genovo Trader v2...")
    
    # List of required packages with specific versions
    dependencies = [
        "numpy==1.24.3",         # Specific version for pandas_ta compatibility
        "pandas==2.0.3",         # Compatible with numpy 1.24.3
        "pandas_ta==0.3.14b0",   # Technical analysis library
        "matplotlib==3.7.2",     # For plotting
        "scikit-learn==1.3.0",   # For scaling and data processing
        "torch==2.0.1",          # For neural networks
        "MetaTrader5==5.0.45",   # For MetaTrader 5 connection
        "tenacity==8.2.2",       # For retry logic
        "pytz==2023.3",          # For timezone handling
        "pyyaml==6.0.1",         # For loading config files
        "joblib==1.3.2",         # For saving/loading models
        "psutil==5.9.5",         # For process management
    ]
    
    # Check if pip is available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        print("Error: pip is not installed or not working correctly")
        return False
    
    # Install each dependency
    successful_installs = 0
    for package in dependencies:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--force-reinstall"])
            successful_installs += 1
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    # Report results
    if successful_installs == len(dependencies):
        print("All dependencies installed successfully!")
        return True
    else:
        print(f"Installed {successful_installs}/{len(dependencies)} dependencies")
        print("Some packages could not be installed. Please check the errors above.")
        return False

if __name__ == "__main__":
    print("Starting dependency installation for Genovo Trader v2")
    if install_dependencies():
        print("Setup complete! You can now run the trading system.")
    else:
        print("Setup encountered some errors. Please resolve them before running the system.") 
