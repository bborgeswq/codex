import numpy as np


class RiskManager:
    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = -0.02  # -2%
        self.correlation_exposure = {}
        self.circuit_breakers = []
        self.ensemble = None

    def attach_ensemble(self, ensemble):
        """Link back to ensemble for regime updates."""
        self.ensemble = ensemble

    def check_risk_limits(self, symbol: str, proposed_size_pct: float, decision):
        """5 camadas de proteção."""
        checks = {}

        if self.daily_pnl < self.max_daily_loss:
            checks["daily_loss"] = False
            return checks

        if self.daily_trades >= 20:
            checks["trade_limit"] = False
            return checks

        var_risk = decision.get("var_risk", 0.02)
        max_size = min(proposed_size_pct, 0.005 / max(var_risk, 1e-6))  # 0.5% / VaR

        corr_exposure = self.correlation_exposure.get(symbol, 0)
        if corr_exposure > 0.3:
            max_size *= 0.5

        regime = decision.get("regime", "range")
        regime_multipliers = {"trending": 1.5, "range": 0.7, "choppy": 0.3}
        max_size *= regime_multipliers.get(regime, 1.0)

        anomaly_risk = decision.get("anomaly_risk", 1.0)
        max_size *= anomaly_risk

        checks["approved_size"] = max(max_size, 0.001)  # Min 0.1%
        checks["all_clear"] = True

        return checks

    def update_exposure(self, symbol: str, side: str, size_pct: float, pnl_pct: float):
        """Track correlation + daily metrics."""
        self.daily_pnl += pnl_pct * size_pct
        self.daily_trades += 1

        direction = 1 if side.lower() == "buy" else -1
        self.correlation_exposure[symbol] = self.correlation_exposure.get(symbol, 0) + direction * size_pct

        if self.daily_pnl < -0.015:
            self.circuit_breakers.append("daily_loss_1.5")
            print("🚨 CIRCUIT BREAKER: Daily loss -1.5%")

        return self.update_market_regime(symbol, pnl_pct)

    def update_market_regime(self, symbol: str, pnl_pct: float):
        """HMM stub for regime detection."""
        if not self.ensemble or not getattr(self.ensemble, "self_learning", None):
            recent_pnls = [pnl_pct]
        else:
            recent_pnls = [t["pnl_pct"] for t in self.ensemble.self_learning.trade_history[-20:]] or [pnl_pct]

        mean_ret = np.mean(recent_pnls)
        std_ret = np.std(recent_pnls)

        if mean_ret > 0.002:
            return "trending"
        elif std_ret > 0.01:
            return "choppy"
        else:
            return "range"
