# Genovo Trader V2 - Professional Forex Trading Bot

## 🚀 Overview

Genovo Trader V2 is a state-of-the-art high-frequency forex trading system that leverages advanced machine learning, sophisticated risk management, and professional trading strategies to deliver institutional-grade performance.

## ✨ Key Features

### 1. **Advanced ML Model**
- Ultra High-Frequency Model with transformer architecture
- Multi-scale attention for different time horizons
- Adaptive Fourier features for cyclical pattern detection
- Quantile regression for uncertainty-aware predictions
- Ensemble predictions for robustness

### 2. **Multi-Broker Support (Plug & Play)**
- Abstract broker interface for easy integration
- Currently supports MetaTrader 5 (MT5)
- Easy to add: Interactive Brokers, Alpaca, OANDA, etc.
- Standardized order types and position management

### 3. **Professional Web UI**
- Real-time performance monitoring
- Live position tracking
- Risk metrics dashboard
- System logs viewer
- Configuration management

### 4. **Sophisticated Risk Management**
- Dynamic position sizing with Kelly Criterion
- Portfolio-level risk constraints
- VaR and CVaR calculations
- Drawdown protection
- Correlation-based position limits
- Market regime detection

### 5. **High-Frequency Trading Capabilities**
- Market microstructure modeling
- Adaptive slippage prediction
- Smart order routing
- Liquidity-aware execution
- Sub-second decision making

## 📋 Requirements

- Python 3.8+
- Windows (for MetaTrader 5) or Linux (for other brokers)
- CUDA-capable GPU (optional, for faster training)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/tafolabi009/genovo_traderv2.git
cd genovo_traderv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your broker settings in `configs/params.yaml`

## 🎯 Quick Start

### Training Mode
Train the model on historical data:
```bash
python main.py --mode simulation
```

### Live Trading
Start live trading:
```bash
python main.py --mode live
```

### Web UI
Access the dashboard:
```bash
python ui/app.py
```
Then open http://localhost:5000 in your browser

## 🏗️ Architecture

```
genovo_traderv2/
├── broker/              # Broker interfaces
│   ├── base_broker.py   # Abstract base class
│   ├── metatrader_interface.py  # MT5 implementation
│   └── mock.py          # Mock broker for testing
├── core/                # Core trading logic
│   ├── model.py         # Main ML model
│   ├── model_enhancements.py  # Advanced model components
│   ├── strategy.py      # Trading strategy
│   ├── features.py      # Feature engineering
│   ├── simulator.py     # Backtesting simulator
│   └── portfolio_compound.py  # Portfolio management
├── data/                # Data processing
│   ├── preprocessing.py # Data cleaning
│   └── macro.py         # Macro indicators
├── ui/                  # Web interface
│   ├── app.py          # Flask application
│   └── templates/      # HTML templates
├── utils/              # Utilities
│   ├── logger.py       # Logging
│   └── notifier.py     # Email notifications
├── configs/            # Configuration
│   └── params.yaml     # Main config file
└── main.py            # Entry point
```

## ⚙️ Configuration

Edit `configs/params.yaml` to customize:

### Broker Settings
```yaml
broker:
  account_id: YOUR_ACCOUNT_ID
  password: YOUR_PASSWORD
  server: YOUR_SERVER
  mt5_path: PATH_TO_MT5
```

### Trading Symbols
```yaml
symbols:
  - 'EURUSDm'
  - 'GBPUSDm'
  - 'USDJPYm'
```

### Risk Management
```yaml
portfolio_capital_config:
  initial_capital: 5000.0
  max_total_risk_pct: 0.05  # 5% max risk
  max_allocation_per_trade_pct: 0.02  # 2% per trade
```

### Model Settings
```yaml
model_config:
  hidden_size: 768
  num_layers: 8
  num_heads: 32
  learning_rate: 0.0001
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total P&L, daily/weekly/monthly returns
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk**: VaR, CVaR, maximum drawdown
- **Trading**: Win rate, profit factor, expectancy
- **Execution**: Slippage, market impact, fill rate

## 🔐 Security Best Practices

1. **Never commit credentials** - Use environment variables
2. **Use secure connections** - Enable SSL/TLS for broker APIs
3. **Implement 2FA** - Where supported by broker
4. **Regular backups** - Save models and configuration
5. **Monitor logs** - Watch for suspicious activity

## 🎓 Advanced Features

### Adding a New Broker

1. Create a new broker class inheriting from `BaseBroker`:
```python
from broker.base_broker import BaseBroker

class MyBroker(BaseBroker):
    def connect(self):
        # Implementation
        pass
    
    def place_order(self, symbol, order_type, volume, **kwargs):
        # Implementation
        pass
    
    # Implement other abstract methods
```

2. Update configuration to use the new broker
3. Test thoroughly in simulation mode first

### Custom Features

Add custom technical indicators in `core/features.py`:
```python
def my_custom_indicator(df):
    # Your implementation
    return indicator_values
```

### Model Customization

Modify the model architecture in `core/model.py` or add new components in `core/model_enhancements.py`.

## 🐛 Troubleshooting

### Common Issues

**Issue**: Cannot connect to MT5
- **Solution**: Check MT5 is running and credentials are correct

**Issue**: Model training is slow
- **Solution**: Enable CUDA or reduce batch size

**Issue**: High slippage
- **Solution**: Adjust `min_trade_interval` and use limit orders

## 📈 Backtesting

Run comprehensive backtests:
```bash
python main.py --mode simulation --symbols EURUSD,GBPUSD --start-date 2023-01-01
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

**Trading forex involves substantial risk of loss. This software is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly in simulation mode before live trading. The authors are not responsible for any financial losses.**

## 📞 Support

For questions and support:
- Email: tafolabi009@gmail.com
- GitHub Issues: https://github.com/tafolabi009/genovo_traderv2/issues

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- MetaTrader for the trading platform
- The open-source trading community

---

**Built with ❤️ for professional traders**
