# Cloud Setup Guide for Genovo Trader v2

This guide provides instructions for setting up and running Genovo Trader v2 on a cloud server.

## Prerequisites

- A cloud server running Windows or Linux (Windows recommended for MT5 compatibility)
- Python 3.10+ installed
- Git installed to clone/update the repository
- MetaTrader 5 terminal installed on the server (or access to the terminal64.exe)

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/genovo_traderv2.git
cd genovo_traderv2
```

### 2. Run the Cloud Setup Script

This script will install dependencies and configure the environment:

```bash
python setup_cloud.py
```

### 3. Configure MetaTrader 5

1. Ensure MT5 is installed in the path specified in your config file (`configs/params.yaml`)
2. For Linux servers, use Wine to run MT5
3. Make sure the terminal64.exe file is accessible

### 4. Update Configuration

Check your `configs/params.yaml` file and verify:

1. MT5 credentials (login, password, server)
2. MT5 path is correct for your environment
3. Symbols you want to trade are properly configured

## Common Issues and Solutions

### MetaTrader 5 IPC Timeout

If you encounter `MT5 initialize() failed, error code = (-10005, 'IPC timeout')`:

1. Make sure MT5 is actually running before starting the bot
2. The MT5 timeout fix is included in this codebase (utils/mt5_timeout_fix.py)
3. Consider increasing the timeout value in the MT5 initialization code

```python
# In main.py, find the initialize_mt5_connection function
if not mt5.initialize(path=path, login=login, password=password, server=server, timeout=120000):
    # Increase timeout from 60000 to 120000 or higher if needed
```

### pandas_ta Compatibility Issues

If you encounter issues with pandas_ta and numpy versions:

1. The cloud setup script adds compatibility fixes
2. Alternatively, manually install compatible versions:

```bash
pip install numpy==1.24.3 pandas==2.0.3 pandas_ta==0.3.14b0
```

### Running on Linux with Wine

If running on a Linux server:

1. Install Wine:
```bash
sudo apt-get update
sudo apt-get install wine-stable winetricks
```

2. Set up the wine environment:
```bash
export WINEPREFIX=~/.wine
export WINEARCH=win64
wine wineboot
```

3. Install MT5 in Wine:
```bash
wine path/to/mt5setup.exe
```

4. Update the path in configs/params.yaml to the Wine path:
```yaml
mt5_config:
  path: "/home/username/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
```

## Running the Bot

### Start in Training Mode

```bash
python main.py --mode train
```

### Start in Live Trading Mode

```bash
python main.py --mode live
```

### Running in the Background

Using nohup (Linux):

```bash
nohup python main.py --mode live > trading.log 2>&1 &
```

Using screen (Linux):

```bash
screen -S trading
python main.py --mode live
# Press Ctrl+A, then D to detach
```

## Monitoring and Troubleshooting

- Check logs in the `results/` directory
- Monitor the MT5 terminal for connection status
- Use `top` or `htop` to ensure the process is running

## Updating the Bot

```bash
git pull
# Run setup again if dependencies changed
python setup_cloud.py
``` 
