import hashlib
import random
from datetime import datetime
from typing import Dict, List

import numpy as np

try:
    import optuna  # type: ignore
except ImportError:
    optuna = None


class SelfLearning:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.trade_history: List[Dict] = []
        self.setup_patterns: Dict[str, Dict] = {}
        self.study = optuna.create_study(direction="maximize") if optuna else None

    def log_trade(self, symbol: str, decision: Dict, pnl_pct: float, duration_min: int, features: Dict):
        """Log completo para análise posterior."""
        trade = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "signal": decision.get("signal"),
            "confidence": decision.get("confidence"),
            "size_pct": decision.get("size_pct"),
            "pnl_pct": pnl_pct,
            "duration_min": duration_min,
            "features_hash": self.hash_features(features),
            "features": features,
            "model_votes": decision,
        }
        self.trade_history.append(trade)

        pattern_id = f"{symbol}_{self.classify_setup(features)}"
        if pattern_id not in self.setup_patterns:
            self.setup_patterns[pattern_id] = {"wins": 0, "losses": 0, "total_pnl": 0}

        if pnl_pct > 0:
            self.setup_patterns[pattern_id]["wins"] += 1
        else:
            self.setup_patterns[pattern_id]["losses"] += 1
        self.setup_patterns[pattern_id]["total_pnl"] += pnl_pct

    def classify_setup(self, features: Dict) -> str:
        """Classifica setup: breakout/pullback/divergence/etc."""
        rsi = features.get("rsi_14", 50)
        macd = features.get("macd", 0)
        bb_pos = features.get("bb_position", 0.5)

        if rsi > 70 and macd > 0:
            return "rsi_overbought"
        elif rsi < 30 and macd < 0:
            return "rsi_oversold"
        elif bb_pos > 0.95:
            return "bb_breakout_upper"
        elif bb_pos < 0.05:
            return "bb_breakout_lower"
        else:
            return "range"

    def hash_features(self, features: Dict) -> str:
        """MD5 hash para similaridade."""
        feature_vec = np.array(list(features.values())[:72])
        return hashlib.md5(feature_vec.tobytes()).hexdigest()

    def daily_retrain(self):
        """Pipeline noturno completo (15min)."""
        print("🔄 Starting daily retrain...")
        recent_trades = self.trade_history[-1000:]
        self.ensemble.ppo.daily_retrain()
        sharpe = self.calculate_sharpe(recent_trades)
        print(f"📊 90d Sharpe: {sharpe:.2f}")
        self.optimize_hyperparams()
        self.curriculum_update()
        print("✅ Retrain complete")

    def calculate_sharpe(self, trades: List[Dict]) -> float:
        """Sharpe ratio real (annualizado)."""
        returns = [t["pnl_pct"] for t in trades if t.get("pnl_pct") not in (None, 0)]
        if not returns:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        return mean_ret / std_ret * np.sqrt(252 * 24 * 60) if std_ret > 0 else 0.0

    def optimize_hyperparams(self):
        """Optuna para RSI period, BB length, confidence threshold."""
        if not self.study:
            return

        def objective(trial):
            params = {
                "rsi_period": trial.suggest_int("rsi_period", 10, 21),
                "confidence_threshold": trial.suggest_float("confidence_threshold", 0.6, 0.8),
            }
            return self.backtest_params(params)

        self.study.optimize(objective, n_trials=5)
        print(f"🎯 Best params: {self.study.best_params}")

    def curriculum_update(self):
        """Treina regimes simples primeiro → complexos."""
        regimes = self.analyze_regimes()
        simple_regimes = ["range", "pullback"]
        complex_regimes = ["breakout", "trend"]

        for regime in simple_regimes + complex_regimes:
            if regime in regimes and regimes[regime]["sharpe"] > 1.5:
                self.focus_retrain(regime)

    # --- Helper stubs -------------------------------------------------------
    def analyze_regimes(self):
        """Placeholder regime analytics."""
        return {"range": {"sharpe": 1.6}, "breakout": {"sharpe": 1.2}}

    def focus_retrain(self, regime: str):
        """Placeholder for targeted retraining."""
        print(f"🎛️ Focusing retrain on regime: {regime}")

    def backtest_params(self, params: Dict) -> float:
        """Simple heuristic backtest stub for hyperparameter tuning."""
        base = 1.0 if params.get("confidence_threshold", 0.65) >= 0.6 else 0.8
        noise = np.random.random() * 0.1 if hasattr(np, "random") else random.random() * 0.1
        return base + noise
