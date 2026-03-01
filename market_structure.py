"""
Market Structure Analysis
- Swing high / low detection (adaptive lookback for micro-price coins)
- Break of Structure (BOS) & Change of Character (ChoCH)
- Dynamic support / resistance zones
- Order blocks
- ATR, RSI, EMA, Volume indicators
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import config

# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Zone:
    price: float
    strength: int   # hit count
    zone_type: str  # "support" | "resistance"
    upper: float = 0.0
    lower: float = 0.0

    def __post_init__(self):
        if not self.upper:
            self.upper = self.price * 1.003
            self.lower = self.price * 0.997

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

@dataclass
class StructurePoint:
    idx:   int
    price: float
    kind:  str   # "HH","LH","LL","HL","swing_high","swing_low"

@dataclass
class MarketAnalysis:
    # ── Trend ─────────────────────────────────────────────
    trend:          str   = "neutral"  # "bullish","bearish","neutral"
    structure:      str   = "intact"   # "intact","broken","changing"
    last_bos:       str   = "none"     # "bullish_bos","bearish_bos","none"
    last_choch:     str   = "none"

    # ── Levels ────────────────────────────────────────────
    support_zones:  List[Zone] = field(default_factory=list)
    resist_zones:   List[Zone] = field(default_factory=list)
    nearest_support: float = 0.0
    nearest_resist:  float = 0.0
    order_block_bull: Optional[float] = None
    order_block_bear: Optional[float] = None

    # ── Indicators ────────────────────────────────────────
    rsi:        float = 50.0
    atr:        float = 0.0
    atr_pct:    float = 0.0
    ema_fast:   float = 0.0
    ema_slow:   float = 0.0
    ema_trend:  str   = "neutral"
    vol_ratio:  float = 1.0

    # ── Signal ────────────────────────────────────────────
    signal:     str   = "HOLD"
    confidence: float = 0.0
    reason:     str   = ""

    # ── Raw features for NN ───────────────────────────────
    features:   List[float] = field(default_factory=list)

    # ── Debug ─────────────────────────────────────────────
    swing_highs_count: int = 0
    swing_lows_count:  int = 0


# ─────────────────────────────────────────────────────────────────────────────
class MarketStructureAnalyzer:

    def analyze(self, candles: list) -> MarketAnalysis:
        if len(candles) < 20:
            return MarketAnalysis()

        c = np.array([x["close"]  for x in candles], dtype=np.float64)
        h = np.array([x["high"]   for x in candles], dtype=np.float64)
        l = np.array([x["low"]    for x in candles], dtype=np.float64)
        v = np.array([x["volume"] for x in candles], dtype=np.float64)

        # Guard: all-zero price data means API returned bad data
        if c[-1] == 0 or np.max(c) == 0:
            return MarketAnalysis(reason="Zero price data from API")

        result = MarketAnalysis()

        # Indicators
        result.rsi      = self._rsi(c, config.RSI_PERIOD)
        result.atr      = self._atr(h, l, c, config.ATR_PERIOD)
        result.atr_pct  = result.atr / c[-1] if c[-1] > 0 else 0
        ema_f           = self._ema(c, config.EMA_FAST)
        ema_s           = self._ema(c, config.EMA_SLOW)
        result.ema_fast = ema_f[-1]
        result.ema_slow = ema_s[-1]
        result.ema_trend= "bullish" if result.ema_fast > result.ema_slow else "bearish"
        vol_ma          = np.mean(v[-config.VOLUME_MA_PERIOD:]) if len(v) >= config.VOLUME_MA_PERIOD else np.mean(v)
        result.vol_ratio= v[-1] / (vol_ma + 1e-30)  # 1e-30 avoids div/0 at any scale

        # Swing points (adaptive — handles PEPE/SHIB style micro prices)
        swings = self._detect_swings(h, l, config.SWING_LOOKBACK)
        highs_sw = [s for s in swings if s[2] == "high"]
        lows_sw  = [s for s in swings if s[2] == "low"]
        result.swing_highs_count = len(highs_sw)
        result.swing_lows_count  = len(lows_sw)

        # Market structure
        trend, structure, last_bos, last_choch = self._classify_structure(swings, c)
        result.trend      = trend
        result.structure  = structure
        result.last_bos   = last_bos
        result.last_choch = last_choch

        # Support / Resistance zones
        sup_zones, res_zones = self._build_zones(swings, c[-1])
        result.support_zones  = sup_zones
        result.resist_zones   = res_zones

        # Fallback: if no zones detected, use ATR-derived levels
        if sup_zones:
            result.nearest_support = sup_zones[0].price
        else:
            result.nearest_support = c[-1] - result.atr * 2.0  # 2 ATRs below

        if res_zones:
            result.nearest_resist = res_zones[0].price
        else:
            result.nearest_resist = c[-1] + result.atr * 2.0   # 2 ATRs above

        # Order blocks
        result.order_block_bull, result.order_block_bear = self._find_order_blocks(candles)

        # Signal generation
        result.signal, result.confidence, result.reason = \
            self._generate_signal(result, c[-1])

        # Feature vector for neural net (normalized)
        result.features = self._build_features(result, c, h, l, v)

        return result

    # ── Swing Detection ───────────────────────────────────────────────────────
    def _detect_swings(self, h: np.ndarray, l: np.ndarray, n: int) -> list:
        """
        Adaptive swing detection — tries decreasing lookback values until
        enough swings are found. Works at any price magnitude (BTC or PEPE).
        """
        best_swings = []
        for lookback in [n, max(n - 2, 2), 2, 1]:
            swings = []
            for i in range(lookback, len(h) - lookback):
                is_sh = all(h[i] >= h[i - j] for j in range(1, lookback + 1)) and \
                        all(h[i] >= h[i + j] for j in range(1, lookback + 1))
                is_sl = all(l[i] <= l[i - j] for j in range(1, lookback + 1)) and \
                        all(l[i] <= l[i + j] for j in range(1, lookback + 1))
                if is_sh:
                    swings.append((i, float(h[i]), "high"))
                if is_sl:
                    swings.append((i, float(l[i]), "low"))
            swings.sort(key=lambda x: x[0])
            highs = [s for s in swings if s[2] == "high"]
            lows  = [s for s in swings if s[2] == "low"]
            if len(highs) >= 3 and len(lows) >= 3:
                return swings
            if len(swings) > len(best_swings):
                best_swings = swings
        return best_swings

    # ── Structure Classification ───────────────────────────────────────────────
    def _classify_structure(self, swings: list, c: np.ndarray):
        highs = [(i, p) for i, p, k in swings if k == "high"]
        lows  = [(i, p) for i, p, k in swings if k == "low"]

        trend = "neutral"; structure = "intact"
        last_bos = "none"; last_choch = "none"

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1][1] > highs[-2][1]
            lh = highs[-1][1] < highs[-2][1]
            hl = lows[-1][1]  > lows[-2][1]
            ll = lows[-1][1]  < lows[-2][1]

            if hh and hl:
                trend = "bullish"
            elif lh and ll:
                trend = "bearish"
            else:
                trend = "neutral"

            if len(highs) >= 1 and c[-1] > highs[-1][1] * 1.001:
                last_bos = "bullish_bos"
            elif len(lows) >= 1 and c[-1] < lows[-1][1] * 0.999:
                last_bos = "bearish_bos"

            if trend == "bullish" and ll:
                last_choch = "bearish_choch"
                structure  = "changing"
            elif trend == "bearish" and hh:
                last_choch = "bullish_choch"
                structure  = "changing"

        return trend, structure, last_bos, last_choch

    # ── Support / Resistance Zones ────────────────────────────────────────────
    def _build_zones(self, swings: list, current_price: float):
        # Use relative clustering (0.5% of price) — works at any magnitude
        cluster_pct = 0.005
        sup_prices  = sorted([p for _, p, k in swings if k == "low"  and p < current_price], reverse=True)
        res_prices  = sorted([p for _, p, k in swings if k == "high" and p > current_price])

        def cluster(prices, zone_type):
            zones = []
            for p in prices:
                if zones and abs(p - zones[-1].price) / (zones[-1].price + 1e-30) < cluster_pct:
                    zones[-1].strength += 1
                    zones[-1].price = (zones[-1].price + p) / 2
                    zones[-1].upper = zones[-1].price * 1.003
                    zones[-1].lower = zones[-1].price * 0.997
                else:
                    zones.append(Zone(price=p, strength=1, zone_type=zone_type))
            return zones

        sup = cluster(sup_prices, "support")
        res = cluster(res_prices, "resistance")
        sup = sorted(sup, key=lambda z: (-z.strength, abs(z.price - current_price)))[:5]
        res = sorted(res, key=lambda z: (-z.strength, abs(z.price - current_price)))[:5]
        return sup, res

    # ── Order Blocks ──────────────────────────────────────────────────────────
    def _find_order_blocks(self, candles: list):
        bull_ob = bear_ob = None
        for i in range(max(0, len(candles) - 10), len(candles) - 1):
            c = candles[i]
            n = candles[i + 1]
            body_c = abs(c["close"] - c["open"])
            body_n = abs(n["close"] - n["open"])
            if body_c == 0 or body_n == 0:
                continue
            if c["close"] < c["open"] and n["close"] > n["open"] and body_n > 1.5 * body_c:
                bull_ob = (c["low"] + c["high"]) / 2
            if c["close"] > c["open"] and n["close"] < n["open"] and body_n > 1.5 * body_c:
                bear_ob = (c["low"] + c["high"]) / 2
        return bull_ob, bear_ob

    # ── Signal Generation ─────────────────────────────────────────────────────
    def _generate_signal(self, m: "MarketAnalysis", price: float):
        score = 0.0
        sell_score = 0.0
        reasons = []

        # Guard against bad price
        if price <= 0 or m.nearest_support <= 0 or m.nearest_resist <= 0:
            return "HOLD", 0.0, "Invalid price data"

        dist_sup = abs(price - m.nearest_support) / price
        dist_res = abs(price - m.nearest_resist)  / price
        near_sup = dist_sup < 0.015
        near_res = dist_res < 0.015
        at_bull_ob = m.order_block_bull and abs(price - m.order_block_bull) / price < 0.01

        # BUY
        if m.trend == "bullish" and m.structure == "intact":
            score += 0.3; reasons.append("bullish structure")
        if near_sup:
            score += 0.25; reasons.append(f"at support ({dist_sup*100:.2f}% away)")
        if at_bull_ob:
            score += 0.2;  reasons.append("bullish OB")
        if m.last_bos == "bullish_bos":
            score += 0.15; reasons.append("BOS bullish")
        if m.rsi < 40:
            score += 0.1;  reasons.append(f"RSI oversold {m.rsi:.1f}")
        if m.ema_trend == "bullish":
            score += 0.1;  reasons.append("EMA bullish")
        if m.vol_ratio > 1.5:
            score += 0.1;  reasons.append(f"vol spike {m.vol_ratio:.1f}x")

        # SELL
        if m.trend == "bearish" and m.structure == "intact":
            sell_score += 0.3
        if near_res:
            sell_score += 0.25; reasons.append(f"at resistance ({dist_res*100:.2f}% away)")
        if m.last_choch == "bearish_choch":
            sell_score += 0.25; reasons.append("bearish ChoCH")
        if m.last_bos == "bearish_bos":
            sell_score += 0.15; reasons.append("BOS bearish")
        if m.rsi > 70:
            sell_score += 0.1;  reasons.append(f"RSI overbought {m.rsi:.1f}")
        if m.ema_trend == "bearish":
            sell_score += 0.1

        if score > sell_score and score > 0.4:
            return "BUY", min(score, 1.0), " | ".join(reasons)
        elif sell_score > score and sell_score > 0.4:
            return "SELL", min(sell_score, 1.0), " | ".join(reasons)
        return "HOLD", 0.5, f"No clear signal (buy={score:.2f} sell={sell_score:.2f})"

    # ── Feature Vector ────────────────────────────────────────────────────────
    def _build_features(self, m: "MarketAnalysis", c, h, l, v) -> list:
        price = c[-1]
        if price <= 0:
            return [0.0] * config.STATE_SIZE
        sup = m.nearest_support if m.nearest_support > 0 else price * 0.97
        res = m.nearest_resist  if m.nearest_resist  > 0 else price * 1.03

        def ret(n):
            return (c[-1] - c[-n]) / (c[-n] + 1e-30) if len(c) >= n else 0.0

        features = [
            ret(2), ret(5), ret(10), ret(20),
            m.rsi / 100.0,
            min(m.atr_pct, 0.05) / 0.05,
            (m.ema_fast - m.ema_slow) / (price + 1e-30),
            (price - sup) / (price + 1e-30),
            (res - price) / (price + 1e-30),
            min(m.vol_ratio, 3.0) / 3.0,
            1.0 if m.trend == "bullish"  else (-1.0 if m.trend == "bearish"  else 0.0),
            1.0 if m.structure == "intact" else (0.0 if m.structure == "changing" else -1.0),
            1.0 if m.last_bos    == "bullish_bos"   else (-1.0 if m.last_bos == "bearish_bos" else 0.0),
            1.0 if m.last_choch  == "bullish_choch" else (-1.0 if m.last_choch == "bearish_choch" else 0.0),
            1.0 if m.ema_trend == "bullish" else -1.0,
            (h[-1] - l[-1]) / (price + 1e-30),
            (max(h[-5:]) - min(l[-5:])) / (price + 1e-30),
            (m.support_zones[0].strength / 10.0) if m.support_zones else 0.0,
            (m.resist_zones[0].strength  / 10.0) if m.resist_zones  else 0.0,
            1.0 if m.order_block_bull else 0.0,
            1.0 if m.order_block_bear else 0.0,
            abs(c[-1] - c[-2]) / ((h[-1] - l[-1]) + 1e-30),
            1.0 if c[-1] > c[-2] else -1.0,
            1.0 if c[-2] > c[-3] else -1.0,
        ]
        return [float(f) for f in features]

    # ── Classic Indicators ────────────────────────────────────────────────────
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        k = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _rsi(self, close: np.ndarray, period: int) -> float:
        window = close[-(period * 2):]
        deltas = np.diff(window)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])  + 1e-30
        avg_loss = np.mean(losses[-period:]) + 1e-30
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _atr(self, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> float:
        tr = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))
