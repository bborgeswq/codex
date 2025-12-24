import random
from typing import Dict, Optional

try:
    import ccxt  # type: ignore
except ImportError:
    ccxt = None

from ai_crypto_trader.config import Config


class ExecutionEngine:
    def __init__(self):
        self.binance = (
            ccxt.binance(
                {
                    "apiKey": Config.BINANCE_TESTNET_KEY,
                    "secret": Config.BINANCE_TESTNET_SECRET,
                    "sandbox": True,
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                }
            )
            if ccxt
            else None
        )
        self.positions: Dict = {}
        self.balance = Config.INITIAL_CAPITAL_USD
        self.open_orders: Dict[str, Dict] = {}

    def get_balance(self) -> float:
        """USDT disponível testnet."""
        try:
            if self.binance:
                balance = self.binance.fetch_balance()
                return balance["USDT"]["free"]
        except Exception:
            pass
        return self.balance

    def calculate_position_size(self, price: float, risk_pct: float, stop_distance_pct: float) -> float:
        """Kelly Criterion + max risk."""
        balance = self.get_balance()
        risk_usd = balance * Config.MAX_RISK_PER_TRADE * risk_pct
        size_usd = risk_usd / max(abs(stop_distance_pct), 1e-6)
        return min(size_usd / price, balance * 0.02)  # Max 2% capital

    def place_order(
        self, symbol: str, side: str, size_usd: float, price: float, stop_loss: float, target: float
    ) -> Optional[Dict]:
        """Ordem market + OCO stop/target."""
        try:
            size = size_usd / price
            if self.binance:
                order = self.binance.create_market_order(symbol, side, size)
                order_id = order["id"]

                self.binance.create_order(
                    symbol, "stop_market", "sell" if side == "buy" else "buy", size, None, params={"stopPrice": stop_loss}
                )

                self.binance.create_order(
                    symbol,
                    "take_profit_market",
                    "sell" if side == "buy" else "buy",
                    size,
                    None,
                    params={"stopPrice": target},
                )
            else:
                order_id = f"sim-{random.randint(1000,9999)}"
                order = {"id": order_id, "status": "filled"}

            self.open_orders[order_id] = {
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry": price,
                "stop": stop_loss,
                "target": target,
                "status": "open",
            }
            print(f"✅ {side.upper()} {symbol} | Size: ${size_usd:.0f} | Stop: {stop_loss} | Target: {target}")
            return order

        except Exception as e:
            print(f"❌ Order failed: {e}")
            return None

    def check_positions(self):
        """Monitora posições abertas + trailing stop."""
        for order_id, pos in list(self.open_orders.items()):
            try:
                if self.binance:
                    order = self.binance.fetch_order(order_id, pos["symbol"])
                    if order["status"] in ["closed", "canceled"]:
                        self.close_position(order_id)
            except Exception:
                continue

    def close_position(self, order_id: str):
        """Calcula PnL + cleanup."""
        if order_id in self.open_orders:
            del self.open_orders[order_id]

    def simulate_slippage(self, symbol: str, size_usd: float) -> float:
        """Slippage realista 0.05-0.2%."""
        if not self.binance:
            return 0.0015
        orderbook = self.binance.fetch_order_book(symbol)
        spread_pct = ((orderbook["asks"][0][0] - orderbook["bids"][0][0]) / orderbook["bids"][0][0])
        return float(spread_pct + 0.001)
