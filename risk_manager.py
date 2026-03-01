"""
Risk Manager
─────────────────────────────────────────────────────────────────────────────
• Kelly-inspired position sizing
• Dynamic stop loss based on ATR
• Trailing stop logic
• Max drawdown guard
• Portfolio exposure limits
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import config

log = logging.getLogger("risk_manager")

@dataclass
class Position:
    product_id:    str
    entry_price:   float
    base_size:     float     # amount of base currency held
    quote_spent:   float     # USD/USDC spent
    stop_loss:     float
    take_profit:   float
    trail_stop:    float     # trailing stop price (updates with price)
    highest_price: float     # for trailing stop tracking
    entry_time:    float     # unix timestamp
    reason:        str = ""
    order_id:      str = ""

@dataclass
class RiskManager:
    positions: dict = field(default_factory=dict)  # product_id → Position
    peak_portfolio: float = 0.0

    def can_open_trade(self, usdc_balance: float, product_id: str) -> bool:
        if len(self.positions) >= config.MAX_OPEN_TRADES:
            log.info(f"Max open trades reached ({config.MAX_OPEN_TRADES})")
            return False
        if product_id in self.positions:
            log.info(f"Already in position for {product_id}")
            return False
        if usdc_balance <= config.MIN_USDC_RESERVE:
            log.info("Insufficient USDC (reserve limit)")
            return False
        return True

    def position_size_usdc(self, usdc_balance: float, atr_pct: float) -> float:
        """
        Kelly-inspired sizing: risk a fraction, reduce size for high volatility.
        Never more than MAX_TRADE_PCT of portfolio.
        """
        base_pct = config.MAX_TRADE_PCT
        # Reduce size if ATR is high (more volatile = smaller position)
        vol_adj  = max(0.3, 1.0 - (atr_pct / 0.05))
        size_pct = base_pct * vol_adj
        size_usdc = (usdc_balance - config.MIN_USDC_RESERVE) * size_pct
        return max(0.0, round(size_usdc, 2))

    def compute_stops(self, entry: float, atr: float, direction: str = "long"):
        """ATR-based stop loss and take profit."""
        sl_dist = max(atr * 1.5, entry * config.STOP_LOSS_PCT)
        tp_dist = max(atr * 3.0, entry * config.TAKE_PROFIT_PCT)
        if direction == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return round(sl, 8), round(tp, 8)

    def open_position(self, product_id: str, entry: float, base_size: float,
                      quote_spent: float, atr: float, reason: str = ""):
        sl, tp = self.compute_stops(entry, atr)
        pos = Position(
            product_id=product_id,
            entry_price=entry,
            base_size=base_size,
            quote_spent=quote_spent,
            stop_loss=sl,
            take_profit=tp,
            trail_stop=sl,
            highest_price=entry,
            entry_time=__import__("time").time(),
            reason=reason,
        )
        self.positions[product_id] = pos
        log.info(f"📥 Position opened: {product_id} @ {entry:.6f} "
                 f"SL={sl:.6f} TP={tp:.6f} qty={base_size:.8f}")
        return pos

    def update_trailing_stop(self, product_id: str, current_price: float):
        """Update trailing stop as price moves in our favor."""
        if product_id not in self.positions:
            return
        pos = self.positions[product_id]
        if current_price > pos.highest_price:
            pos.highest_price = current_price
            # Trail at 1.5x ATR below the highest price
            new_trail = current_price * (1 - config.STOP_LOSS_PCT * 0.8)
            if new_trail > pos.trail_stop:
                pos.trail_stop = new_trail
                log.debug(f"Trailing stop updated: {product_id} trail={new_trail:.6f}")

    def should_exit(self, product_id: str, current_price: float,
                    analysis) -> tuple:
        """
        Returns (should_exit: bool, reason: str)
        Checks: stop loss, take profit, trailing stop, structure rejection.
        """
        if product_id not in self.positions:
            return False, ""
        pos = self.positions[product_id]

        # Hard stop loss
        if current_price <= pos.stop_loss:
            return True, f"STOP_LOSS hit @ {current_price:.6f}"

        # Trailing stop
        if current_price <= pos.trail_stop and current_price < pos.entry_price:
            return True, f"TRAIL_STOP hit @ {current_price:.6f}"

        # Take profit
        if current_price >= pos.take_profit:
            return True, f"TAKE_PROFIT hit @ {current_price:.6f}"

        # Structure rejection: bearish ChoCH or BOS while we're long
        if analysis.last_choch == "bearish_choch" and analysis.confidence > 0.6:
            return True, "Bearish ChoCH — structure rejection"

        if analysis.last_bos == "bearish_bos" and analysis.trend == "bearish":
            return True, "Bearish BOS — trend reversed"

        # At strong resistance zone
        if analysis.resist_zones:
            for zone in analysis.resist_zones[:2]:
                if zone.contains(current_price) and zone.strength >= 2:
                    return True, f"At resistance zone {zone.price:.6f} (strength={zone.strength})"

        return False, ""

    def close_position(self, product_id: str, exit_price: float) -> Optional[float]:
        """Removes position and returns PnL."""
        if product_id not in self.positions:
            return None
        pos = self.positions.pop(product_id)
        pnl = (exit_price - pos.entry_price) / pos.entry_price  # % return
        pnl_usdc = pos.base_size * (exit_price - pos.entry_price)
        log.info(f"📤 Position closed: {product_id} @ {exit_price:.6f} | "
                 f"Entry={pos.entry_price:.6f} | PnL={pnl*100:.2f}% (${pnl_usdc:.2f})")
        return pnl

    def check_max_drawdown(self, current_portfolio_value: float) -> bool:
        """Return True if max drawdown exceeded (5% from peak)."""
        if current_portfolio_value > self.peak_portfolio:
            self.peak_portfolio = current_portfolio_value
        drawdown = (self.peak_portfolio - current_portfolio_value) / (self.peak_portfolio + 1e-9)
        if drawdown > 0.05:
            log.warning(f"⚠️  Max drawdown reached: {drawdown*100:.1f}%")
            return True
        return False
