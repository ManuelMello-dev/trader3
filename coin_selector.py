"""
Autonomous Coin Selector
─────────────────────────────────────────────────────────────────────────────
Scans available markets and scores each coin on:
  • Liquidity (24h volume)
  • Volatility (ATR%) — more volatile = more opportunity
  • Trend strength (EMA alignment)
  • Market structure quality (clear swings)
  • Recent momentum
Selects the highest-scoring coin to trade.
"""

import logging
import numpy as np
from market_structure import MarketStructureAnalyzer
import config

log = logging.getLogger("coin_selector")

class CoinSelector:
    def __init__(self, client):
        self.client   = client
        self.analyzer = MarketStructureAnalyzer()

    def select_best_coin(self) -> str:
        """Returns product_id of the best coin to trade right now."""
        log.info("🔍 Scanning coins for best opportunity...")

        products = self.client.list_products(config.SCAN_QUOTE_CURRENCY)
        if not products:
            log.warning("No products returned. Defaulting to BTC-USDC")
            return "BTC-USDC"

        # Filter products
        candidates = []
        for p in products:
            pid = p.get("product_id", "")
            base = p.get("base_currency_id", "")
            if base in config.EXCLUDED_COINS:
                continue
            if p.get("status", "") != "online":
                continue
            try:
                vol = float(p.get("volume_24h", 0))
            except:
                continue
            if vol < config.MIN_24H_VOLUME_USD:
                continue
            candidates.append((pid, vol))

        if not candidates:
            log.warning("No candidates after filter. Defaulting to BTC-USDC")
            return "BTC-USDC"

        # Sort by volume and take top N
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:config.TOP_N_BY_VOLUME]
        log.info(f"Evaluating {len(candidates)} coins...")

        scores = []
        for pid, vol in candidates:
            score = self._score_coin(pid, vol)
            if score is not None:
                scores.append((pid, score))
                log.info(f"  {pid:20s}  score={score:.3f}")

        if not scores:
            return "BTC-USDC"

        best = max(scores, key=lambda x: x[1])
        log.info(f"✅ Best coin selected: {best[0]} (score={best[1]:.3f})")
        return best[0]

    def _score_coin(self, product_id: str, volume_24h: float) -> float:
        try:
            candles = self.client.get_candles(
                product_id, config.CANDLE_GRANULARITY, limit=100
            )
            if len(candles) < 50:
                return None

            analysis = self.analyzer.analyze(candles)

            # ── Scoring components ────────────────────────────────────────────
            # 1. Volatility score: sweet spot 0.5-3% ATR — too low = boring, too high = risky
            atr_pct = analysis.atr_pct * 100
            vol_score = self._bell_curve(atr_pct, mu=1.5, sigma=1.0)

            # 2. Volume score (log normalized, 7 = ~$10M volume)
            import math
            vol_log_score = min(math.log10(max(volume_24h, 1)) / 7.0, 1.0)

            # 3. Trend clarity: clear bullish or bearish > neutral
            trend_score = 0.8 if analysis.trend in ("bullish", "bearish") else 0.3

            # 4. Structure quality: intact > changing
            struct_score = 1.0 if analysis.structure == "intact" else 0.5

            # 5. RSI not in extreme (tradeable range 35-65)
            rsi = analysis.rsi
            rsi_score = 1.0 if 35 < rsi < 65 else (0.6 if 25 < rsi < 75 else 0.2)

            # 6. EMA alignment bonus
            ema_score = 0.8 if analysis.ema_trend in ("bullish", "bearish") else 0.4

            # 7. Momentum: recent candles moving
            closes = [c["close"] for c in candles[-10:]]
            mom = abs(closes[-1] - closes[0]) / (closes[0] + 1e-9)
            mom_score = min(mom / 0.03, 1.0)

            # Weighted composite
            score = (
                vol_score    * 0.25 +
                vol_log_score* 0.20 +
                trend_score  * 0.15 +
                struct_score * 0.15 +
                rsi_score    * 0.10 +
                ema_score    * 0.10 +
                mom_score    * 0.05
            )
            return score

        except Exception as e:
            log.debug(f"Scoring failed for {product_id}: {e}")
            return None

    @staticmethod
    def _bell_curve(x: float, mu: float, sigma: float) -> float:
        """Gaussian scoring — peak at mu, falls off with sigma width."""
        return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
