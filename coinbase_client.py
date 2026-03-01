"""
Coinbase Advanced Trade API client.
Uses JWT authentication as required by the newer API.
"""

import time, json, hashlib, hmac, uuid, logging
from typing import Optional
import requests
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import config

log = logging.getLogger("coinbase_client")

BASE_URL = "https://api.coinbase.com"

class CoinbaseClient:
    def __init__(self):
        self.api_key    = config.API_KEY
        self.api_secret = config.API_SECRET
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Authentication ────────────────────────────────────────────────────────
    def _build_jwt(self, method: str, path: str) -> str:
        """Build a short-lived JWT for Coinbase Advanced Trade API."""
        uri = f"{method} api.coinbase.com{path}"
        private_key = serialization.load_pem_private_key(
            self.api_secret.encode("utf-8"),
            password=None,
            backend=default_backend()
        )
        payload = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 60,
            "uri": uri,
        }
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers={"kid": self.api_key, "nonce": uuid.uuid4().hex},
        )
        return token

    def _request(self, method: str, path: str, params=None, body=None):
        token = self._build_jwt(method.upper(), path)
        headers = {"Authorization": f"Bearer {token}"}
        url = BASE_URL + path
        try:
            resp = self.session.request(
                method, url, headers=headers,
                params=params,
                json=body,
                timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP error {resp.status_code}: {resp.text}")
            return None
        except Exception as e:
            log.error(f"Request failed: {e}")
            return None

    # ── Market Data ───────────────────────────────────────────────────────────
    def get_candles(self, product_id: str, granularity: str, limit: int = 200):
        """Fetch OHLCV candles. Returns list of dicts sorted oldest→newest."""
        end   = int(time.time())
        gran_seconds = {
            "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600, "SIX_HOUR": 21600, "ONE_DAY": 86400
        }.get(granularity, 3600)
        start = end - gran_seconds * limit
        data = self._request("GET", f"/api/v3/brokerage/products/{product_id}/candles",
                             params={"start": start, "end": end, "granularity": granularity})
        if not data or "candles" not in data:
            return []
        candles = sorted(data["candles"], key=lambda x: int(x["start"]))
        return [
            {
                "time":   int(c["start"]),
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "close":  float(c["close"]),
                "volume": float(c["volume"]),
            }
            for c in candles
        ]

    def get_best_bid_ask(self, product_id: str):
        data = self._request("GET", f"/api/v3/brokerage/best_bid_ask",
                             params={"product_ids": product_id})
        if data and "pricebooks" in data and data["pricebooks"]:
            pb = data["pricebooks"][0]
            try:
                bid = float(pb["bids"][0]["price"]) if pb.get("bids") else None
                ask = float(pb["asks"][0]["price"]) if pb.get("asks") else None
                if bid is not None and ask is not None:
                    return bid, ask
            except (IndexError, ValueError, KeyError):
                pass
        return None, None

    def get_product(self, product_id: str):
        return self._request("GET", f"/api/v3/brokerage/products/{product_id}")

    def list_products(self, quote_currency: str = "USDC"):
        data = self._request("GET", "/api/v3/brokerage/products",
                             params={"quote_currency_id": quote_currency})
        if data and "products" in data:
            return data["products"]
        return []

    def get_24h_stats(self, product_id: str):
        """Returns approximate 24h stats from product info."""
        prod = self.get_product(product_id)
        if prod:
            return {
                "volume": float(prod.get("volume_24h", 0)),
                "price_pct_change": float(prod.get("price_percentage_change_24h", 0)),
            }
        return None

    # ── Account / Portfolio ───────────────────────────────────────────────────
    def get_accounts(self):
        data = self._request("GET", "/api/v3/brokerage/accounts")
        if data and "accounts" in data:
            return data["accounts"]
        return []

    def get_balance(self, currency: str) -> float:
        accounts = self.get_accounts()
        for acc in accounts:
            if acc["currency"] == currency:
                return float(acc["available_balance"]["value"])
        return 0.0

    def get_portfolio(self) -> dict:
        """Returns {currency: available_balance} for all non-zero accounts."""
        accounts = self.get_accounts()
        portfolio = {}
        for acc in accounts:
            val = float(acc["available_balance"]["value"])
            if val > 0:
                portfolio[acc["currency"]] = val
        return portfolio

    # ── Orders ────────────────────────────────────────────────────────────────
    def market_buy(self, product_id: str, quote_size: float) -> Optional[dict]:
        """Buy using quote currency (e.g. spend $100 USDC of BTC)."""
        body = {
            "client_order_id": uuid.uuid4().hex,
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": f"{quote_size:.6f}"
                }
            }
        }
        log.info(f"MARKET BUY {product_id} quote_size={quote_size:.2f}")
        return self._request("POST", "/api/v3/brokerage/orders", body=body)

    def market_sell(self, product_id: str, base_size: float) -> Optional[dict]:
        """Sell base currency amount (e.g. sell 0.001 BTC)."""
        body = {
            "client_order_id": uuid.uuid4().hex,
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{base_size:.8f}"
                }
            }
        }
        log.info(f"MARKET SELL {product_id} base_size={base_size:.8f}")
        return self._request("POST", "/api/v3/brokerage/orders", body=body)

    def limit_buy(self, product_id: str, base_size: float, limit_price: float) -> Optional[dict]:
        body = {
            "client_order_id": uuid.uuid4().hex,
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size":  f"{base_size:.8f}",
                    "limit_price": f"{limit_price:.6f}",
                    "post_only": False,
                }
            }
        }
        log.info(f"LIMIT BUY {product_id} size={base_size:.8f} @ {limit_price:.4f}")
        return self._request("POST", "/api/v3/brokerage/orders", body=body)

    def limit_sell(self, product_id: str, base_size: float, limit_price: float) -> Optional[dict]:
        body = {
            "client_order_id": uuid.uuid4().hex,
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size":  f"{base_size:.8f}",
                    "limit_price": f"{limit_price:.6f}",
                    "post_only": False,
                }
            }
        }
        log.info(f"LIMIT SELL {product_id} size={base_size:.8f} @ {limit_price:.4f}")
        return self._request("POST", "/api/v3/brokerage/orders", body=body)

    def cancel_orders(self, order_ids: list) -> Optional[dict]:
        return self._request("POST", "/api/v3/brokerage/orders/batch_cancel",
                            body={"order_ids": order_ids})

    def list_open_orders(self, product_id: str = None) -> list:
        params = {"order_status": "OPEN"}
        if product_id:
            params["product_id"] = product_id
        data = self._request("GET", "/api/v3/brokerage/orders/historical/batch", params=params)
        if data and "orders" in data:
            return data["orders"]
        return []
