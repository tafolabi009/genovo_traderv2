# Deployment Guide

## Production Deployment Checklist

### Pre-Deployment

- [ ] **Test thoroughly in simulation mode** (at least 3 months of backtest data)
- [ ] **Validate configuration** (`python launcher.py --validate-only`)
- [ ] **Check broker connectivity** (verify MT5/broker is accessible)
- [ ] **Review risk parameters** (ensure appropriate limits)
- [ ] **Set up monitoring** (email notifications, logging)
- [ ] **Backup strategy** (models, configs, code)
- [ ] **Emergency stop plan** (know how to stop bot quickly)

### Security Hardening

#### 1. Credentials Management

**Never commit credentials!** Use environment variables:

```bash
# Linux/Mac
export BROKER_PASSWORD="your_password"
export BROKER_ACCOUNT="your_account"

# Windows
set BROKER_PASSWORD=your_password
set BROKER_ACCOUNT=your_account
```

Update config to use environment variables:

```python
import os

config['broker']['password'] = os.getenv('BROKER_PASSWORD')
config['broker']['account_id'] = os.getenv('BROKER_ACCOUNT')
```

#### 2. Firewall Configuration

```bash
# Allow only necessary ports
ufw allow 5000/tcp  # Web UI (optional, can be localhost only)
ufw enable
```

#### 3. SSL/TLS for Web UI

Use nginx as reverse proxy:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Windows Service Deployment

#### 1. Install as Windows Service

```bash
# Install service
python genovo_trader_service.py install

# Start service
python genovo_trader_service.py start

# Check status
python genovo_trader_service.py status

# Stop service
python genovo_trader_service.py stop
```

#### 2. Configure Auto-Recovery

In `genovo_trader_service.py`, the service is configured to auto-restart on failure.

### Linux Systemd Service

Create `/etc/systemd/system/genovo-trader.service`:

```ini
[Unit]
Description=Genovo Trader V2 Trading Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/genovo_traderv2
Environment="PATH=/home/trader/genovo_traderv2/venv/bin"
ExecStart=/home/trader/genovo_traderv2/venv/bin/python launcher.py --mode live
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable genovo-trader
sudo systemctl start genovo-trader
sudo systemctl status genovo-trader
```

View logs:

```bash
sudo journalctl -u genovo-trader -f
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose UI port
EXPOSE 5000

# Run
CMD ["python", "launcher.py", "--mode", "live"]
```

Build and run:

```bash
# Build image
docker build -t genovo-trader:latest .

# Run container
docker run -d \
  --name genovo-trader \
  --restart unless-stopped \
  -p 5000:5000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/results:/app/results \
  -e BROKER_PASSWORD=${BROKER_PASSWORD} \
  genovo-trader:latest
```

### Cloud Deployment

#### AWS EC2

1. **Launch Instance**: t3.medium or larger (GPU recommended)
2. **Security Group**: Allow SSH (22), HTTPS (443)
3. **Install dependencies**:

```bash
sudo apt update
sudo apt install python3-pip python3-venv
```

4. **Setup application**:

```bash
git clone https://github.com/tafolabi009/genovo_traderv2.git
cd genovo_traderv2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Configure systemd** (see above)

#### Google Cloud Platform

Similar to AWS, use Compute Engine instance.

#### Azure

Use Azure VM with similar setup.

### Monitoring

#### 1. Email Alerts

Configure in `configs/params.yaml`:

```yaml
notifications:
  email_enabled: true
  email_address: "your_email@gmail.com"
  email_password: "app_password"
  email_recipient: "alerts@yourdomain.com"
```

#### 2. Log Monitoring

Use log aggregation:

```bash
# Install filebeat
sudo apt install filebeat

# Configure to send logs to ELK stack or similar
```

#### 3. Performance Metrics

The web UI provides real-time metrics at http://your-server:5000

#### 4. Health Checks

Create a health check endpoint:

```python
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })
```

Monitor with:

```bash
# Simple check
curl http://localhost:5000/health

# With monitoring service (e.g., UptimeRobot, Pingdom)
```

### Backup Strategy

#### 1. Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/genovo_trader"

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" results/*.pth results/*.joblib

# Backup config
cp configs/params.yaml "$BACKUP_DIR/config_$DATE.yaml"

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" results/*.log

# Keep only last 30 days
find "$BACKUP_DIR" -mtime +30 -delete
```

Add to crontab:

```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

#### 2. Cloud Backup

```bash
# AWS S3
aws s3 sync results/ s3://your-bucket/genovo-backups/

# Google Cloud Storage
gsutil rsync -r results/ gs://your-bucket/genovo-backups/
```

### Performance Tuning

#### 1. System Optimization

```bash
# Increase file descriptors
ulimit -n 65536

# Tune network
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=2048
```

#### 2. Database for Trade History

For production, consider using PostgreSQL:

```python
# utils/db.py
import psycopg2

def store_trade(trade_data):
    conn = psycopg2.connect(
        dbname="genovo",
        user="trader",
        password=os.getenv('DB_PASSWORD'),
        host="localhost"
    )
    # Store trade
    conn.close()
```

#### 3. Redis for Caching

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379)

# Cache market data
redis_client.setex('EURUSD:price', 60, price_data)
```

### Troubleshooting

#### Bot Stops Trading

1. Check logs: `tail -f results/genovo_traderv2.log`
2. Verify broker connection
3. Check account balance/margin
4. Review error emails

#### Poor Performance

1. Check slippage settings
2. Verify internet connection latency
3. Review risk parameters
4. Analyze trade history

#### High CPU/Memory Usage

1. Reduce model size
2. Decrease feature count
3. Lower update frequency
4. Use GPU for inference

### Scaling

#### Horizontal Scaling

Run multiple instances for different symbols:

```bash
# Instance 1: EUR pairs
python launcher.py --symbols EURUSDm EURGBPm --ui-port 5001

# Instance 2: GBP pairs
python launcher.py --symbols GBPUSDm GBPJPYm --ui-port 5002
```

#### Vertical Scaling

- Upgrade to GPU instance
- Increase RAM for larger models
- Use SSD for faster I/O

### Regulatory Compliance

**Important**: Ensure compliance with local regulations:

- Register as required in your jurisdiction
- Maintain trade records (typically 5-7 years)
- Report profits for tax purposes
- Follow broker terms of service
- Consider consulting a financial lawyer

### Support Contacts

- **Technical Issues**: tafolabi009@gmail.com
- **GitHub Issues**: https://github.com/tafolabi009/genovo_traderv2/issues
- **Documentation**: See README.md and QUICKSTART.md

### Maintenance Schedule

- **Daily**: Check logs, verify trades
- **Weekly**: Review performance metrics
- **Monthly**: Retrain models, update config
- **Quarterly**: Security audit, dependency updates
- **Annually**: Strategy review, regulatory compliance check

---

## Emergency Procedures

### Stop Trading Immediately

```bash
# Kill process
pkill -f launcher.py

# Or stop service
sudo systemctl stop genovo-trader

# Windows service
python genovo_trader_service.py stop
```

### Close All Positions

1. Access web UI: http://localhost:5000
2. Click "Stop" button
3. Manually close positions in MT5/broker terminal

### Rollback

```bash
# Restore from backup
cd genovo_traderv2
git checkout <previous-commit>
cp /backup/config_backup.yaml configs/params.yaml
```

---

**Remember**: Always test in simulation mode before going live!
