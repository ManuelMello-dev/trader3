# 🤖 Adaptive Crypto Trading Agent

A biologically-inspired, self-learning trading agent that connects to Coinbase, selects its own coins, performs technical market structure analysis, and executes live trades 24/7 — learning and adapting from every outcome.

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN LOOP (every 60s)                │
├─────────────────────────────────────────────────────────┤
│  CoinSelector  ──►  Scans 20+ markets, scores & picks   │
│  MarketStructure ──► Swings, BOS/ChoCH, S/R zones, OBs │
│  NeuralAgent   ──►  DQN + Hebbian RL decision maker     │
│  RiskManager   ──►  ATR stops, trailing, drawdown guard │
│  CoinbaseClient ──► JWT auth, live orders               │
└─────────────────────────────────────────────────────────┘
```

### Neural Network (Biologically Inspired)
- **Deep Q-Network (DQN)**: Learns optimal buy/sell/hold decisions through reinforcement learning
- **Hebbian Synaptic Strengthening**: Frequently rewarded neural pathways self-reinforce over time (mimics biological long-term potentiation)
- **Eligibility Traces**: Spike-timing inspired reward attribution across time steps
- **Online Continuous Learning**: Updates after every trade — never stops improving
- **Dual Network**: Policy net (active) + Target net (stable) for training stability

### Market Structure Analysis
- Swing high/low detection (configurable lookback)
- Break of Structure (BOS) detection
- Change of Character (ChoCH) — early reversal signal
- Dynamic support/resistance zone clustering
- Order block identification
- ATR, RSI, EMA indicators
- Volume analysis

### Coin Selection
The agent autonomously selects the best coin to trade every 30 minutes based on:
- Liquidity (24h volume)
- Volatility sweet spot (ATR %)
- Trend clarity and structure quality
- RSI positioning
- Momentum

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Coinbase API Keys
1. Go to [Coinbase Advanced Trade](https://advanced.coinbase.com)
2. Click your profile → **API Keys**
3. Create a new key with permissions:
   - ✅ View
   - ✅ Trade
   - ❌ Transfer (not needed, don't grant this)
4. Copy your **API Key Name** and **Private Key (PEM)**

### 3. Configure the agent
Edit `config.py`:
```python
API_KEY    = "organizations/YOUR_ORG/apiKeys/YOUR_KEY_ID"
API_SECRET = """-----BEGIN EC PRIVATE KEY-----
YOUR PRIVATE KEY HERE
-----END EC PRIVATE KEY-----"""
```

### 4. Tune risk settings (important!)
```python
MAX_TRADE_PCT   = 0.10   # 10% of portfolio per trade — start lower (0.05)
STOP_LOSS_PCT   = 0.025  # 2.5% stop loss
MAX_OPEN_TRADES = 3
MIN_USDC_RESERVE = 50.0  # Always keep $50 USDC untouched
```

### 5. Run
```bash
python run.py
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `config.py` | All settings — API keys, risk params, NN hyperparams |
| `run.py` | Entry point |
| `main.py` | Main loop orchestrator |
| `coinbase_client.py` | Coinbase Advanced Trade API wrapper (JWT auth) |
| `market_structure.py` | TA engine: swings, BOS, ChoCH, S/R, indicators |
| `neural_agent.py` | DQN + Hebbian neural network agent |
| `coin_selector.py` | Autonomous market scanner and coin scorer |
| `risk_manager.py` | Position sizing, stops, drawdown guard |
| `agent.log` | Live log output |
| `trades.json` | Full trade history |
| `model_weights.npz` | Saved neural network (auto-updates) |

---

## 🔄 How a Tick Works (every 60 seconds)

```
1. Check if time to rescan for a better coin (every 30 min)
2. Fetch 200 candles for current coin
3. Run market structure analysis → 24 features extracted
4. If in position:
   a. Update trailing stop
   b. Check exit conditions (SL, TP, trail, structure rejection, resistance zones)
   c. If exiting → market sell → compute PnL → teach NN
   d. If holding → give NN small reward for unrealized gain
5. If no position:
   a. NN decides (explore or exploit Q-values) + rule signal blended
   b. If BUY: size = Kelly-inspired ATR-adjusted USDC amount
   c. Execute market buy → open position with ATR-based SL/TP
6. NN trains on batch of 32 past experiences (experience replay)
7. Every 50 steps: sync target network, save model weights
```

---

## ⚠️ Important Notes

- **This trades real money.** Start with a small account.
- **No strategy wins every trade.** The agent is designed to be net positive over time, not perfect.
- **Monitor it.** Check `agent.log` regularly, especially in the first few hours.
- **Coinbase fees**: ~0.6% per trade (taker). The agent accounts for this in reward shaping.
- **The NN starts with epsilon=1.0** (fully random) and explores before exploiting. Initial trades may seem random — this is intentional exploration.
- After ~100-200 trades, the model has learned enough to be more decisive. Model is saved to `model_weights.npz` and resumed on restart.

---

## 🛡️ Risk Controls Summary

| Control | Default | Description |
|---------|---------|-------------|
| Max trade size | 10% portfolio | Per-trade cap |
| Stop loss | 2.5% or 1.5×ATR | Whichever is larger |
| Take profit | 5% or 3×ATR | ~2:1 R/R ratio |
| Trailing stop | Activated after entry | Locks in profits |
| Max open trades | 3 | Portfolio diversification |
| USDC reserve | $50 | Never fully depleted |
| Max drawdown | 5% from peak | Emergency stop |

---

## 🔧 Advanced Tuning

**To make it more aggressive (higher returns, higher risk):**
```python
MAX_TRADE_PCT   = 0.20   # 20% per trade
STOP_LOSS_PCT   = 0.015  # Tighter stops
EPSILON_DECAY   = 0.999  # Slower exploration decay
```

**To make it more conservative:**
```python
MAX_TRADE_PCT   = 0.05
MIN_USDC_RESERVE = 200.0
MAX_OPEN_TRADES  = 1
```

**To change timeframe:**
```python
CANDLE_GRANULARITY = "FIFTEEN_MINUTE"  # More trades, more noise
CANDLE_GRANULARITY = "ONE_DAY"         # Fewer trades, cleaner structure
```
