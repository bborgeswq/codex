import signal
import threading
import time
from datetime import datetime

from ai_crypto_trader.config import Config
from ai_crypto_trader.core.ensemble import TradingEnsemble
from ai_crypto_trader.core.database import TraderDatabase
from ai_crypto_trader.core.monitor import TradingMonitor


class AITrader:
    def __init__(self):
        self.ensemble = TradingEnsemble()
        self.db = TraderDatabase(Config.DB_PATH)
        self.monitor = TradingMonitor(self.ensemble)
        self.trading_active = False
        self.last_cycle = {}
        # attach ensemble reference for risk manager
        self.ensemble.risk_manager.attach_ensemble(self.ensemble)

    def trading_loop(self):
        """Loop principal 24/7 - 1.6 checks/asset por minuto."""
        print("🚀 AI Crypto Trader v5.0 ENTERPRISE iniciado")
        print("📈 Assets: BTCUSDT, ETHUSDT, XRPUSDT | Binance Testnet")

        cycle = 0
        while self.trading_active:
            cycle_start = time.time()

            for symbol in Config.ASSETS:
                try:
                    features = self.ensemble.data_engine.get_features(symbol)
                    current_price = features.get("close", 0)

                    decision = self.ensemble.predict(symbol)

                    self.monitor.show_thinking(symbol, features, decision)

                    if decision["signal"] != "HOLD" and decision["confidence"] >= Config.CONFIDENCE_THRESHOLD:
                        risk_ok = self.ensemble.risk_manager.check_risk_limits(symbol, decision["size_pct"], decision)

                        if risk_ok.get("all_clear", False):
                            success = self.ensemble.execute_trade(symbol, decision)
                            if success:
                                print(f"✅ TRADE EXECUTADO: {symbol} {decision['signal']}")

                    self.last_cycle[symbol] = {
                        "decision": decision["signal"],
                        "confidence": decision["confidence"],
                        "price": current_price,
                    }

                except Exception as e:
                    print(f"❌ Error {symbol}: {e}")

            if cycle % 10 == 0:
                self.monitor.update_metrics()

            cycle_time = time.time() - cycle_start
            sleep_time = max(45 - cycle_time, 5)
            time.sleep(sleep_time)
            cycle += 1

    def start(self):
        """Inicia trader com signal handlers."""
        self.trading_active = True

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        trader_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trader_thread.start()

        monitor_thread = threading.Thread(target=self.monitor.start, daemon=True)
        monitor_thread.start()

        print("⏳ Press Ctrl+C para parar graciosamente")
        trader_thread.join()

    def stop(self, signum=None, frame=None):
        print("\n🛑 Parando AI Trader...")
        self.trading_active = False
        self.db.update_metrics(datetime.now().date().isoformat(), "1D", self.monitor.get_final_metrics())
        print("💾 Dados salvos no banco. Até logo!")


if __name__ == "__main__":
    trader = AITrader()
    trader.start()
