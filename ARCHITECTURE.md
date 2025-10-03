# ARCHITECTURE.md

# Genovo Trader V2 - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Web UI     │    │   CLI Tool   │    │  Dashboard   │    │
│  │ (Flask/HTML) │    │  (launcher)  │    │  (Realtime)  │    │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    │
│         │                   │                    │             │
└─────────┼───────────────────┼────────────────────┼─────────────┘
          │                   │                    │
          └───────────────────┼────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                    CORE TRADING SYSTEM                         │
├─────────────────────────────┼─────────────────────────────────┤
│                             ▼                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              MAIN CONTROLLER (main.py)               │    │
│  │  • Mode selection (simulation/live)                  │    │
│  │  • Component orchestration                           │    │
│  │  • Lifecycle management                              │    │
│  └────┬─────────────────────────┬───────────────────────┘    │
│       │                         │                             │
│       ▼                         ▼                             │
│  ┌─────────┐              ┌──────────┐                       │
│  │ Training│              │   Live   │                       │
│  │  Loop   │              │ Trading  │                       │
│  └────┬────┘              └────┬─────┘                       │
└───────┼──────────────────────────┼──────────────────────────┘
        │                          │
        ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML/AI LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         UltraHighFrequencyModel (PyTorch)               │  │
│  │  ┌────────────────┐  ┌────────────────┐                │  │
│  │  │  Transformer   │  │  TCN Blocks    │                │  │
│  │  │  Encoder       │  │  (Multi-scale) │                │  │
│  │  └────────┬───────┘  └────────┬───────┘                │  │
│  │           └──────────┬─────────┘                        │  │
│  │                      ▼                                  │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │     Model Enhancements (NEW)                    │   │  │
│  │  │  • Adaptive Fourier Features                    │   │  │
│  │  │  • Multi-Scale Attention                        │   │  │
│  │  │  • Adaptive Risk Module                         │   │  │
│  │  │  • Market Microstructure                        │   │  │
│  │  │  • Quantile Heads                               │   │  │
│  │  │  • Ensemble Predictions                         │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                      ▼                                  │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │         PPO Agent (Strategy)                    │   │  │
│  │  │  • Action selection                             │   │  │
│  │  │  • Reward shaping                               │   │  │
│  │  │  • Policy optimization                          │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │ Technical   │  │  Wavelets    │  │  Kalman Filter │        │
│  │ Indicators  │  │  Transform   │  │                │        │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘        │
│         └─────────────────┼──────────────────┘                 │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │          Feature Pipeline                           │       │
│  │  • Normalization (Robust Quantile)                 │       │
│  │  • Feature selection (Mutual Info)                 │       │
│  │  • 512-dimensional feature space                   │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RISK MANAGEMENT LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Portfolio Capital Manager                        │  │
│  │  • Position sizing (Kelly, confidence-weighted)         │  │
│  │  • Portfolio risk limits (5% max total risk)            │  │
│  │  • Per-trade allocation (2% max)                        │  │
│  │  • Dynamic capital allocation                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Risk Metrics & Monitoring                        │  │
│  │  • VaR (Value at Risk)                                   │  │
│  │  • CVaR (Conditional VaR)                               │  │
│  │  • Drawdown tracking                                     │  │
│  │  • Correlation monitoring                               │  │
│  │  • Regime detection                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BROKER ABSTRACTION LAYER (NEW)               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Broker Factory                              │  │
│  │  • Plug & Play broker switching                          │  │
│  │  • Standardized interface                                │  │
│  │  • Order type abstraction                                │  │
│  └─────────┬────────────────────────────────────────────────┘  │
│            │                                                    │
│            ├───────────┬─────────────┬─────────────┐           │
│            ▼           ▼             ▼             ▼           │
│  ┌──────────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐      │
│  │ MetaTrader 5 │ │ Alpaca  │ │  IBKR    │ │  Custom  │      │
│  │  (Active)    │ │(Template│ │ (Future) │ │ (Future) │      │
│  └──────────────┘ └─────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │ Market Data │  │  Historical  │  │  Macro Data    │        │
│  │  (Live)     │  │  Data Cache  │  │  (FRED API)    │        │
│  └─────────────┘  └──────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interface Layer

#### Web UI (NEW)
- **Technology**: Flask + HTML/CSS/JavaScript
- **Features**:
  - Real-time performance dashboard
  - Position monitoring
  - Risk metrics visualization
  - System logs
  - Trading controls (start/stop/pause)
- **Port**: 5000 (configurable)

#### CLI Launcher (NEW)
- **File**: `launcher.py`
- **Features**:
  - Mode selection
  - Config validation
  - Easy startup
  - Debug mode
  - Symbol override

### 2. Core Trading System

#### Main Controller
- **File**: `main.py`
- **Responsibilities**:
  - Initialize all components
  - Manage training/live mode
  - Handle scheduling (weekly retraining)
  - Error recovery
  - MT5 connection management

#### Training Loop
- Data loading and cleaning
- Feature engineering
- Model training (PPO)
- Model evaluation
- Checkpoint saving

#### Live Trading Loop
- Real-time data fetching
- Feature calculation
- Model inference
- Signal generation
- Order execution
- Position management

### 3. ML/AI Layer

#### UltraHighFrequencyModel
- **Architecture**: Transformer + TCN
- **Size**: 768 hidden, 8 layers, 32 heads
- **Enhancements** (NEW):
  - Adaptive Fourier Features
  - Multi-Scale Attention
  - Adaptive Risk Module
  - Market Microstructure Module
  - Quantile Regression Heads
  - Ensemble Predictions

#### PPO Agent
- **Algorithm**: Proximal Policy Optimization
- **Actions**: Hold, Buy, Sell
- **Reward**: Sophisticated multi-factor reward function
- **Updates**: Every 256 steps (configurable)

### 4. Feature Engineering

#### Technical Indicators
- 500+ indicators from pandas-ta
- Multi-timeframe analysis
- Volume profile
- Order flow

#### Advanced Features
- Wavelet decomposition
- Kalman filtering
- Regime indicators
- Market microstructure

#### Feature Pipeline
- Robust quantile normalization
- Mutual information selection
- Top 512 features
- Online scaling

### 5. Risk Management

#### Portfolio Manager
- **Max Total Risk**: 5%
- **Max Per Trade**: 2%
- **Methods**: Kelly Criterion, Confidence-weighted
- **Min Position**: $100

#### Risk Metrics
- VaR (95%)
- CVaR
- Maximum Drawdown
- Current Drawdown
- Position correlation

### 6. Broker Abstraction (NEW)

#### Base Broker Interface
- Abstract methods for all operations
- Standardized order types
- Position management
- Historical data fetching

#### Broker Factory
- Plug-and-play switching
- Easy addition of new brokers
- Configuration-driven selection

#### Supported Brokers
- **MetaTrader 5**: Full support
- **Alpaca**: Template provided
- **IBKR**: Planned
- **Custom**: Easy to add

### 7. Data Layer

#### Market Data
- Real-time from broker
- Historical data cache
- Data preprocessing
- Quality checks

#### External Data
- FRED API (macro indicators)
- News API (sentiment)
- Economic calendar

## Data Flow

### Training Mode
```
1. Load Historical Data → 2. Feature Engineering → 
3. Model Training → 4. Validation → 5. Save Model
```

### Live Trading Mode
```
1. Fetch Live Data → 2. Calculate Features → 
3. Model Inference → 4. Risk Check → 
5. Generate Signal → 6. Execute Order → 
7. Update Portfolio → 8. Monitor Position
```

## Performance Characteristics

- **Latency**: Sub-second inference
- **Throughput**: 100+ decisions/second
- **Memory**: ~2GB RAM (CPU mode)
- **Storage**: ~1GB (models + data)
- **GPU**: Optional, 10x faster training

## Security

- Environment variable credentials
- No hardcoded secrets
- SSL/TLS support
- Input validation
- Error sanitization

## Scalability

- Horizontal: Multiple instances per symbol
- Vertical: GPU acceleration
- Distributed: Redis/PostgreSQL support ready

## Monitoring

- Real-time web dashboard
- Email notifications
- Log aggregation ready
- Health check endpoint
- Performance metrics

---

**Legend**:
- ✨ NEW = Components added in this update
- → = Data flow direction
- ▼ = Hierarchical dependency
