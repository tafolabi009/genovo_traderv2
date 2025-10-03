# ui/app.py

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import yaml
import os
import json
from datetime import datetime
import threading
import logging

app = Flask(__name__)
CORS(app)

logger = logging.getLogger("genovo_traderv2.ui")

# Global state
trading_state = {
    'status': 'stopped',
    'positions': [],
    'performance': {},
    'last_update': None,
    'account_info': {},
    'recent_trades': []
}


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current trading status"""
    return jsonify(trading_state)


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    config_path = 'configs/params.yaml'
    
    if request.method == 'GET':
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return jsonify({'success': True, 'config': config_data})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            with open(config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            return jsonify({'success': True, 'message': 'Configuration updated'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})


@app.route('/api/positions')
def get_positions():
    """Get current open positions"""
    return jsonify(trading_state.get('positions', []))


@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    return jsonify(trading_state.get('performance', {}))


@app.route('/api/trades/recent')
def get_recent_trades():
    """Get recent trade history"""
    return jsonify(trading_state.get('recent_trades', []))


@app.route('/api/account')
def get_account():
    """Get account information"""
    return jsonify(trading_state.get('account_info', {}))


@app.route('/api/control/<action>', methods=['POST'])
def control_trading(action):
    """Control trading operations (start/stop/pause)"""
    if action in ['start', 'stop', 'pause', 'resume']:
        trading_state['status'] = action
        return jsonify({'success': True, 'action': action})
    return jsonify({'success': False, 'error': 'Invalid action'})


@app.route('/api/logs/latest')
def get_latest_logs():
    """Get latest log entries"""
    log_file = 'results/genovo_traderv2.log'
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Return last 100 lines
                return jsonify({'success': True, 'logs': lines[-100:]})
        return jsonify({'success': True, 'logs': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def update_trading_state(new_state):
    """Update global trading state (called from main trading loop)"""
    global trading_state
    trading_state.update(new_state)
    trading_state['last_update'] = datetime.now().isoformat()


def run_ui_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask UI server"""
    logger.info(f"Starting UI server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def start_ui_thread(host='0.0.0.0', port=5000):
    """Start UI server in a separate thread"""
    ui_thread = threading.Thread(
        target=run_ui_server,
        args=(host, port, False),
        daemon=True
    )
    ui_thread.start()
    logger.info(f"UI server thread started. Access dashboard at http://{host}:{port}")
    return ui_thread


if __name__ == '__main__':
    # Run standalone for testing
    logging.basicConfig(level=logging.INFO)
    run_ui_server(debug=True)
