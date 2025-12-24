from datetime import datetime

import numpy as np


class TradingMonitor:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.metrics = {
            "total_trades": 0,
            "wins": 0,
            "pnl_total": 0,
            "sharpe_24h": 0,
            "max_dd": 0,
            "winrate": 0,
        }
        self.last_thinking = {}

    def show_thinking(self, symbol, features, decision):
        """TELA DA IA - raciocínio em tempo real."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        regime = self.classify_regime(features)
        rsi = features.get("rsi_14", 50)
        confidence = decision.get("confidence", 0)
        signal = decision.get("signal", "HOLD")

        memory_data = self.ensemble.memory.query_trade_memory(symbol, features)

        thinking = f"""

🧠 TJR THINKING [{symbol}] {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGIME: {regime:^10} | RSI: {rsi:>5.1f} | HV20: {features.get('hv_20',0)*100:>5.1f}%
ORDERBOOK: imb {features.get('imbalance',0)*100:>5.1f}% | spread {features.get('spread_pct',0):>4.2f}%
NEWS: {features.get('sentiment',0):>5.2f} "{features.get('latest_news','')[:60]}..."
MEMÓRIA: {memory_data.get('pattern_winrate',0)*100:>4.0f}% winrate
🎯 ENSEMBLE VOTE:
Claude: {decision.get('claude_confidence',0)*100:>4.0f}%
Chronos: {decision.get('chronos_pred',0)*100:>4.0f}%
PPO: {decision.get('ppo_action','HOLD')}
ANOMALY: {decision.get('anomaly_risk',1.0):>4.2f}
➤ {signal:^7} {confidence*100:>5.1f}% | Size: {decision.get('size_pct','0')*100:>4.1f}%
""".strip()

        print(thinking)
        self.last_thinking[symbol] = thinking

    def classify_regime(self, features):
        adx = features.get("adx_14", 20)
        if adx > 25:
            return "TRENDING"
        elif features.get("hv_20", 0) > 0.03:
            return "CHOPPY"
        else:
            return "RANGE"

    def update_metrics(self):
        """Métricas reais 24h."""
        recent_trades = self.ensemble.db.get_recent_trades(24 * 60)  # 24h
        if recent_trades:
            pnls = [t["pnl_pct"] for t in recent_trades if t.get("pnl_pct") is not None]
            if pnls:
                self.metrics["sharpe_24h"] = np.mean(pnls) / np.std(pnls) * np.sqrt(1440) if np.std(pnls) else 0
                self.metrics["winrate"] = sum(1 for p in pnls if p > 0) / len(pnls)

    def get_final_metrics(self):
        return self.metrics

    def start(self):
        """Streamlit UI opcional em thread separada."""
        try:
            import streamlit as st
        except Exception:
            return
        # Minimal placeholder UI
        with st.empty():
            st.write("Monitoring thread active.")
