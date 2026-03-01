"""
Main Trading Orchestrator
─────────────────────────────────────────────────────────────────────────────
Coordinates: CoinSelector → MarketStructure → NeuralAgent → RiskManager → Execution
Runs 24/7 in a main loop, learns from every trade outcome.
"""

import sys, os
# Ensure the directory containing this file is always on the Python path
# so sibling modules (market_structure, neural_agent, etc.) are importable
# regardless of how or where the container launches main.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time, json, logging, sys
from datetime import datetime
import config
from coinbase_client   import CoinbaseClient
from market_structure  import MarketStructureAnalyzer
from neural_agent      import TradingAgent
from coin_selector     import CoinSelector
from risk_manager      import RiskManager, Position
from price_fmt         import fmt, fmt_pct, fmt_size

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("main")


class TradingOrchestrator:
    def __init__(self):
        log.info("="*60)
        log.info("  🤖 ADAPTIVE TRADING AGENT STARTING")
        log.info("="*60)

        self.client    = CoinbaseClient()
        self.analyzer  = MarketStructureAnalyzer()
        self.agent     = TradingAgent()
        self.selector  = CoinSelector(self.client)
        self.risk      = RiskManager()
        self.trade_log = []

        self.current_coin   = None
        self.last_state     = None
        self.last_action    = "HOLD"
        self.last_scan_time = 0
        self.step           = 0

        # Pre-select coin
        self._rescan_coin()

    # ── Main Loop ─────────────────────────────────────────────────────────────
    def run(self):
        log.info("🔄 Entering main loop...")
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("⛔ Interrupted by user. Shutting down...")
                self._save_trade_log()
                break
            except Exception as e:
                log.error(f"Tick error: {e}", exc_info=True)
                time.sleep(10)

            time.sleep(config.MAIN_LOOP_SECONDS)

    def _tick(self):
        self.step += 1
        now = time.time()

        # ── Rescan coin every N minutes ───────────────────────────────────────
        if now - self.last_scan_time > config.COIN_RESCAN_MINUTES * 60:
            # Only rescan if we have no open position on current coin
            if self.current_coin not in self.risk.positions:
                self._rescan_coin()
            self.last_scan_time = now

        if not self.current_coin:
            log.warning("No coin selected. Retrying scan...")
            self._rescan_coin()
            return

        pid = self.current_coin
        log.info(f"── Tick {self.step} | {pid} | {datetime.now().strftime('%H:%M:%S')} ──")

        # ── Fetch market data ─────────────────────────────────────────────────
        candles = self.client.get_candles(pid, config.CANDLE_GRANULARITY,
                                          config.LOOKBACK_CANDLES)
        if len(candles) < 20:
            log.warning(f"Insufficient candles for {pid} (got {len(candles)})")
            return

        bid, ask = self.client.get_best_bid_ask(pid)
        current_price = (bid + ask) / 2 if (bid and ask and bid > 0 and ask > 0) else candles[-1]["close"]
        
        # Stricter guard: if price is still zero or invalid, skip tick
        if current_price <= 0:
            log.warning(f"Skipping tick for {pid}: current_price is {current_price}")
            return

        # ── Market analysis ───────────────────────────────────────────────────
        analysis = self.analyzer.analyze(candles)
        if analysis.reason == "Zero price data from API" or analysis.nearest_support <= 0:
            log.warning(f"Skipping tick for {pid}: {analysis.reason or 'Invalid levels'}")
            return
            
        state    = analysis.features  # 24-element feature vector

        log.info(f"  Price={fmt(current_price)} | Trend={analysis.trend} | "
                 f"Structure={analysis.structure} | RSI={analysis.rsi:.1f} | "
                 f"Support={fmt(analysis.nearest_support)} | Resist={fmt(analysis.nearest_resist)} | "
                 f"Swings H={analysis.swing_highs_count} L={analysis.swing_lows_count} | "
                 f"Rule={analysis.signal}({analysis.confidence:.2f}) | Reason: {analysis.reason}")

        # ── Portfolio update ──────────────────────────────────────────────────
        usdc_balance   = self.client.get_balance("USDC")
        portfolio      = self.client.get_portfolio()
        portfolio_value= self._estimate_portfolio_value(portfolio, current_price, pid)

        log.info(f"  💰 USDC={usdc_balance:.2f} | Portfolio≈${portfolio_value:.2f} | "
                 f"Positions={list(self.risk.positions.keys())}")

        # ── Check drawdown guard ──────────────────────────────────────────────
        if self.risk.check_max_drawdown(portfolio_value):
            log.warning("🛑 Drawdown guard: closing all positions, pausing 30 min")
            self._close_all_positions(current_price, analysis, "DRAWDOWN_GUARD")
            time.sleep(1800)
            return

        # ── Manage open position ──────────────────────────────────────────────
        if pid in self.risk.positions:
            self.risk.update_trailing_stop(pid, current_price)
            should_exit, exit_reason = self.risk.should_exit(pid, current_price, analysis)
            if should_exit:
                log.info(f"  🚪 Exit signal: {exit_reason}")
                self._execute_sell(pid, current_price, state, exit_reason)
                return

            # Structure holds — log and hold
            log.info(f"  ⏳ Holding position. Trail stop={self.risk.positions[pid].trail_stop:.6f}")
            # Give NN a reward signal for holding through profit
            pos = self.risk.positions[pid]
            unrealized_pnl = (current_price - pos.entry_price) / pos.entry_price
            hold_reward = unrealized_pnl * 0.1  # small reward per tick of unrealized gain
            if self.last_state is not None:
                self.agent.record_experience(self.last_state, "HOLD", hold_reward, state)

        # ── Look for new entry ────────────────────────────────────────────────
        elif self.risk.can_open_trade(usdc_balance, pid):
            # Neural agent decision
            nn_action = self.agent.decide(state, analysis.signal, analysis.confidence)
            log.info(f"  🧠 NN Decision: {nn_action} (ε={self.agent.epsilon:.3f}) | "
                     f"Stats: {self.agent.stats()}")

            # Combine NN + rule signal
            final_action = self._combine_signals(nn_action, analysis.signal, analysis.confidence)
            log.info(f"  📊 Final Action: {final_action}")

            if final_action == "BUY":
                size_usdc = self.risk.position_size_usdc(usdc_balance, analysis.atr_pct)
                if size_usdc >= 10.0:  # minimum $10 trade
                    self._execute_buy(pid, current_price, size_usdc, analysis, state)
                else:
                    log.info(f"  Position size too small (${size_usdc:.2f}), skipping")

        self.last_state  = state
        self.last_action = "HOLD"

    # ── Execution ─────────────────────────────────────────────────────────────
    def _execute_buy(self, pid: str, price: float, size_usdc: float,
                     analysis, state: list):
        log.info(f"  🟢 BUY {pid} | ${size_usdc:.2f} USDC | Price≈{fmt(price)}")
        result = self.client.market_buy(pid, size_usdc)
        if result and result.get("success"):
            # Estimate base size received
            base_size = size_usdc / price * 0.998  # account for ~0.2% fee
            self.risk.open_position(
                pid, price, base_size, size_usdc,
                analysis.atr, analysis.reason
            )
            self.last_action = "BUY"
            self._log_trade("BUY", pid, price, size_usdc, base_size, analysis.reason)
        else:
            log.error(f"  ❌ BUY order failed: {result}")
            # Negative reward for failed action
            if self.last_state:
                self.agent.record_experience(self.last_state, "BUY", -0.01, state)

    def _execute_sell(self, pid: str, price: float, state: list, reason: str):
        pos = self.risk.positions.get(pid)
        if not pos:
            return
        log.info(f"  🔴 SELL {pid} | {fmt_size(pos.base_size)} | Price≈{fmt(price)}")
        result = self.client.market_sell(pid, pos.base_size)
        if result and result.get("success"):
            pnl = self.risk.close_position(pid, price)
            reward = pnl if pnl is not None else 0.0
            log.info(f"  💸 PnL: {fmt_pct(reward)} | reward={reward:.6f}")

            # Teach the agent
            if self.last_state:
                self.agent.record_experience(self.last_state, "SELL", reward, state, done=True)
            self.last_action = "SELL"
            self._log_trade("SELL", pid, price, 0, pos.base_size, reason, pnl=reward)
        else:
            log.error(f"  ❌ SELL order failed: {result}")

    def _close_all_positions(self, price: float, analysis, reason: str):
        for pid in list(self.risk.positions.keys()):
            self._execute_sell(pid, price, [], reason)

    # ── Signal Combination ────────────────────────────────────────────────────
    def _combine_signals(self, nn: str, rule: str, rule_conf: float) -> str:
        """
        Both signals agree → strong action
        Only rule agrees with high confidence → follow rule
        Only NN agrees → follow NN if not HOLD
        Both say HOLD → HOLD
        """
        if nn == rule and nn != "HOLD":
            return nn
        if rule_conf > 0.7 and rule != "HOLD":
            return rule
        if nn != "HOLD":
            return nn
        return "HOLD"

    # ── Coin Rescanning ───────────────────────────────────────────────────────
    def _rescan_coin(self):
        try:
            self.current_coin  = self.selector.select_best_coin()
            self.last_scan_time = time.time()
        except Exception as e:
            log.error(f"Coin scan failed: {e}. Keeping {self.current_coin}")

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _estimate_portfolio_value(self, portfolio: dict, current_price: float, pid: str) -> float:
        base_currency = pid.split("-")[0] if pid else "BTC"
        total = portfolio.get("USDC", 0.0)
        for curr, amt in portfolio.items():
            if curr == "USDC":
                continue
            if curr == base_currency:
                total += amt * current_price
            # Rough estimate for other held coins — use last known price
            # (production: would fetch each price)
        return total

    def _log_trade(self, side: str, pid: str, price: float, usdc: float,
                   base: float, reason: str, pnl: float = None):
        entry = {
            "time":   datetime.utcnow().isoformat(),
            "side":   side,
            "coin":   pid,
            "price":  price,
            "usdc":   usdc,
            "base":   base,
            "reason": reason,
            "pnl":    pnl,
        }
        self.trade_log.append(entry)
        self._save_trade_log()
        log.info(f"  📋 Trade logged: {entry}")

    def _save_trade_log(self):
        with open(config.TRADE_LOG_FILE, "w") as f:
            json.dump(self.trade_log, f, indent=2)
