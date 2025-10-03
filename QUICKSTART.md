# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Broker

Edit `configs/params.yaml`:

```yaml
broker:
  account_id: YOUR_ACCOUNT_ID
  password: "YOUR_PASSWORD"
  server: "YOUR_SERVER"
  mt5_path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

**⚠️ IMPORTANT**: Never commit credentials to git! Use environment variables in production.

### 3. Choose Your Mode

#### A. Backtesting (Simulation Mode)

Test strategies on historical data:

```bash
python launcher.py --mode simulation
```

This will:
- Load historical data
- Train the ML model
- Run backtests
- Generate performance reports

#### B. Live Trading

**⚠️ WARNING**: Only use with accounts you can afford to lose!

```bash
python launcher.py --mode live
```

This will:
- Connect to your broker
- Load trained models
- Execute trades in real-time
- Monitor performance

### 4. Access the Dashboard

The web UI starts automatically at: **http://localhost:5000**

Or start just the UI:

```bash
python launcher.py --ui-only
```

## Common Commands

### Validate Configuration

```bash
python launcher.py --validate-only
```

### Trade Specific Symbols

```bash
python launcher.py --mode live --symbols EURUSDm GBPUSDm
```

### Debug Mode

```bash
python launcher.py --mode simulation --debug
```

### Custom Configuration

```bash
python launcher.py --config my_custom_config.yaml --mode live
```

## Dashboard Features

Access at http://localhost:5000:

- **Real-time Performance**: See P&L, win rate, Sharpe ratio
- **Open Positions**: Monitor active trades
- **Risk Metrics**: Track drawdown, VaR, exposure
- **System Logs**: View real-time logs
- **Controls**: Start/stop/pause trading

## Safety Checklist

Before live trading:

- [ ] Test thoroughly in simulation mode
- [ ] Start with a small account
- [ ] Set appropriate risk limits in `portfolio_capital_config`
- [ ] Monitor for at least a week
- [ ] Have a stop-loss strategy
- [ ] Keep MT5/broker terminal open
- [ ] Check internet connection stability

## Performance Optimization

### For Faster Training

1. **Use GPU**: Install CUDA-enabled PyTorch
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Reduce Data**: Adjust `num_bars` in config
3. **Fewer Features**: Reduce `num_features` in model_config
4. **Smaller Model**: Reduce `hidden_size` and `num_layers`

### For Better Results

1. **More Data**: Increase `num_bars` to 100000+
2. **More Features**: Increase feature extraction windows
3. **Longer Training**: Increase `num_epochs`
4. **Ensemble Models**: Train multiple models

## Troubleshooting

### Issue: Cannot connect to MT5

**Solutions**:
- Ensure MT5 is installed and running
- Check credentials in config
- Verify server name is correct
- Try restarting MT5

### Issue: Model training is slow

**Solutions**:
- Enable GPU support
- Reduce batch size
- Use fewer features
- Train on less data initially

### Issue: Poor trading performance

**Solutions**:
- Increase training data
- Adjust risk parameters
- Test different timeframes
- Check for overfitting (compare train vs validation)

### Issue: Web UI not loading

**Solutions**:
- Check if port 5000 is available
- Try different port: `--ui-port 8080`
- Check firewall settings
- Verify Flask is installed

## Directory Structure

```
genovo_traderv2/
├── configs/
│   └── params.yaml          # Main configuration
├── results/
│   ├── *.pth               # Trained models
│   ├── *.joblib            # Scalers
│   └── *.log               # Logs
├── ui/
│   └── templates/          # Web UI
├── main.py                 # Core trading logic
└── launcher.py             # Entry point
```

## Next Steps

1. Read the full [README.md](README.md)
2. Review risk management settings
3. Customize the model architecture
4. Add your own features
5. Backtest extensively
6. Start with paper trading

## Support

- **Issues**: https://github.com/tafolabi009/genovo_traderv2/issues
- **Email**: tafolabi009@gmail.com

## License

MIT License - See LICENSE file

---

**Happy Trading! 🚀**

Remember: Past performance does not guarantee future results. Trade responsibly.
