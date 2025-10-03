# demo_ui.py
"""
Demo script to show the UI (without actual trading)
Run this to see the dashboard without connecting to a broker
"""

from ui.app import app, update_trading_state
import random
from datetime import datetime
import threading
import time

def simulate_data():
    """Simulate trading data for demo purposes"""
    initial_balance = 10000
    balance = initial_balance
    
    positions = []
    performance = {
        'total_pnl': 0,
        'today_pnl': 0,
        'win_rate': 0,
        'sharpe': 0,
        'max_drawdown': 0,
        'current_drawdown': 0,
        'var': 0
    }
    
    while True:
        # Simulate some price movement
        pnl_change = random.uniform(-50, 100)
        balance += pnl_change
        performance['total_pnl'] = balance - initial_balance
        performance['today_pnl'] = random.uniform(-100, 200)
        performance['win_rate'] = random.uniform(55, 75)
        performance['sharpe'] = random.uniform(1.5, 3.0)
        performance['max_drawdown'] = random.uniform(-5, -1)
        performance['current_drawdown'] = random.uniform(-3, 0)
        performance['var'] = random.uniform(50, 150)
        
        # Simulate positions
        if random.random() > 0.7 and len(positions) < 3:
            positions.append({
                'symbol': random.choice(['EURUSDm', 'GBPUSDm', 'USDJPYm']),
                'type': random.choice(['BUY', 'SELL']),
                'volume': round(random.uniform(0.01, 0.1), 2),
                'open_price': random.uniform(1.0, 150.0),
                'current_price': random.uniform(1.0, 150.0),
                'pnl': random.uniform(-50, 100)
            })
        elif positions and random.random() > 0.8:
            positions.pop()
        
        # Update state
        update_trading_state({
            'status': 'running',
            'positions': positions,
            'performance': performance,
            'account_info': {
                'balance': balance,
                'equity': balance + sum(p['pnl'] for p in positions),
                'margin': random.uniform(100, 500),
                'free_margin': balance - random.uniform(100, 500)
            },
            'recent_trades': []
        })
        
        time.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    print("=" * 60)
    print("  GENOVO TRADER V2 - DEMO MODE")
    print("=" * 60)
    print()
    print("Starting demo with simulated data...")
    print()
    print("Access the dashboard at: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Start data simulation in background
    sim_thread = threading.Thread(target=simulate_data, daemon=True)
    sim_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
