"""
Configuration for the Adaptive Trading Agent
Set your API keys and parameters here.
"""

# ─── Coinbase Advanced Trade API ───────────────────────────────────────────────
API_KEY    = "your_api_key_here"          # e.g. "organizations/xxx/apiKeys/yyy"
API_SECRET = "your_api_secret_here"       # PEM private key string

# ─── Risk Management ───────────────────────────────────────────────────────────
MAX_TRADE_PCT      = 0.10    # Max % of portfolio per trade (10%)
STOP_LOSS_PCT      = 0.025   # Stop loss below entry (2.5%)
TAKE_PROFIT_PCT    = 0.05    # Initial TP above entry (5%)
MAX_OPEN_TRADES    = 3       # Max simultaneous positions
MIN_USDC_RESERVE   = 50.0    # Always keep this much USDC untouched

# ─── Coin Selection ────────────────────────────────────────────────────────────
SCAN_QUOTE_CURRENCY = "USDC"             # Quote currency to trade against
TOP_N_BY_VOLUME     = 20                 # How many coins to evaluate
MIN_24H_VOLUME_USD  = 1_000_000          # Min liquidity filter
EXCLUDED_COINS      = ["USDC", "USDT", "DAI", "BUSD"]  # Stable coins to skip

# ─── Technical Analysis ────────────────────────────────────────────────────────
CANDLE_GRANULARITY  = "ONE_HOUR"         # ONE_MINUTE, FIVE_MINUTE, ONE_HOUR, ONE_DAY
LOOKBACK_CANDLES    = 200                # Candles to fetch per analysis
SWING_LOOKBACK      = 5                  # Bars each side to confirm swing high/low
ATR_PERIOD          = 14
RSI_PERIOD          = 14
EMA_FAST            = 12
EMA_SLOW            = 26
VOLUME_MA_PERIOD    = 20

# ─── Neural Agent ──────────────────────────────────────────────────────────────
STATE_SIZE          = 24     # Feature vector size fed into the network
ACTION_SIZE         = 3      # 0=HOLD, 1=BUY, 2=SELL
HIDDEN_SIZE         = 64
LEARNING_RATE       = 0.001
GAMMA               = 0.95   # Discount factor for RL
EPSILON_START       = 1.0    # Exploration rate (decays over time)
EPSILON_MIN         = 0.05
EPSILON_DECAY       = 0.995
REPLAY_BUFFER_SIZE  = 2000
BATCH_SIZE          = 32
TARGET_UPDATE_FREQ  = 50     # Steps between target network syncs

# ─── Loop Timing ───────────────────────────────────────────────────────────────
MAIN_LOOP_SECONDS   = 60     # How often the agent ticks (seconds)
COIN_RESCAN_MINUTES = 30     # How often to re-evaluate which coin to trade

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE            = "agent.log"
TRADE_LOG_FILE      = "trades.json"
MODEL_SAVE_PATH     = "model_weights.npz"
