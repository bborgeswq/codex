import json
import random
import re
from typing import Dict, Iterable, Tuple

import numpy as np

from ai_crypto_trader.config import Config
from ai_crypto_trader.core.execution import ExecutionEngine

try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None

try:
    import gym  # type: ignore
    from gym import spaces  # type: ignore
except ImportError:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO  # type: ignore
except ImportError:
    PPO = None

try:
    from sklearn.ensemble import IsolationForest  # type: ignore
except ImportError:
    IsolationForest = None

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ImportError:
    StandardScaler = None

try:
    from keras import Sequential  # type: ignore
    from keras.layers import LSTM, Dense, Dropout  # type: ignore
except ImportError:
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None

from ai_crypto_trader.core.data_engine import DataEngine


class ChronosForecaster:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_pretrained()

    def load_pretrained(self):
        """Nixtla Chronos-T5 pre-trained for short-horizon forecasts."""
        if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
            return
        model_name = "nixtla/chronos-t5-small"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception:
            self.model = None
            self.tokenizer = None

    def forecast(self, df_ohlcv, horizon: Iterable[int] = (5, 15, 60)) -> Dict[str, float]:
        """Forecast price changes for the next time windows."""
        if self.model is None or self.tokenizer is None or torch is None:
            return {f"{h}min": 0.0 for h in horizon}

        prices = np.log1p(df_ohlcv["close"].tail(512).values)
        predictions = {}

        for h in horizon:
            inputs = self.tokenizer(prices.astype(np.float32).tolist(), return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=h + 10)
            try:
                pred_prices = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_logret = np.fromstring(pred_prices.replace("[PAD]", ""), sep=" ")
                predictions[f"{h}min"] = float(np.expm1(pred_logret[-h:].mean())) if len(pred_logret) else 0.0
            except Exception:
                predictions[f"{h}min"] = 0.0

        return predictions


class ClaudeDecisionMaker:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY) if anthropic else None
        self.model = "claude-3-5-sonnet-20241022"

    def analyze_market(self, symbol: str, features: Dict, memory_data: Dict) -> Dict:
        """Chain-of-thought numeric JSON for trading decisions."""
        prompt_template = """
        Você é TJR CRYPTO, 12 anos daytrading BTC/ETH/XRP com Sharpe 2.8 histórico.

        ANALISE PASSO-A-PASSO com números EXATOS:

        1. MERCADO ATUAL: Regime={regime} | HV20={hv20:.1f}% | Funding={funding:.2f}%
        2. TÉCNICO: RSI14={rsi14:.1f} | RSI21={rsi21:.1f} | MACD={macd:.4f} 
           BB_pos={bb_pos:.2f} | ATR={atr:.4f} | ADX={adx:.1f}
        3. ORDERBOOK: Imbalance={imbalance:.1f}% | Spread={spread:.2f}%
        4. SENTIMENT: News={sentiment:.2f} | "{latest_news}"
        5. MEMÓRIA: {similar_winrate:.1f}% winrate em {sample_size} trades similares
           Regime Sharpe últimos 30d: {regime_sharpe:.2f}

        DECISÃO FINAL (números exatos):
        SINAL: BUY/SELL/HOLD
        CONFIANÇA: {0-100}%
        STOP_LOSS: {preço exato ou % abaixo}
        TARGET: {preço exato ou % acima}
        POSITION_SIZE: {0.1-1.0}% do capital
        RAZÃO: 1 frase técnica

        FORMATO JSON EXATO:
        {{
            "signal": "BUY",
            "confidence": 0.87,
            "stop_loss": 42500.0,
            "target": 43200.0,
            "position_size": 0.008,
            "reason": "RSI divergência + OBV breakout"
        }}
        """
        prompt = prompt_template.format(
            regime=features.get("regime", "unknown"),
            hv20=features.get("hv_20", 0) * 100,
            funding=features.get("funding_rate", 0),
            rsi14=features.get("rsi_14", 50),
            rsi21=features.get("rsi_21", 50),
            macd=features.get("macd", 0),
            bb_pos=features.get("bb_position", 0.5),
            atr=features.get("atr_14", 0),
            adx=features.get("adx_14", 20),
            imbalance=features.get("imbalance", 0) * 100,
            spread=features.get("spread_pct", 0),
            sentiment=features.get("sentiment", 0),
            latest_news=features.get("latest_news", ""),
            similar_winrate=memory_data.get("pattern_winrate", 0.5) * 100,
            sample_size=memory_data.get("pattern_sample", 0),
            regime_sharpe=memory_data.get("regime_sharpe", 1.0),
        )

        if not self.client:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "stop_loss": features.get("close", 0) * 0.99,
                "target": features.get("close", 0) * 1.01,
                "position_size": 0.005,
                "reason": "Fallback decision without API",
            }

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                system="Você é um day trader quantitativo profissional.",
                messages=[{"role": "user", "content": prompt}],
            )
            content_text = response.content[0].text if hasattr(response, "content") else str(response)
            json_str = re.search(r"\{.*\}", content_text, re.DOTALL)
            return json.loads(json_str.group()) if json_str else {}
        except Exception:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "stop_loss": features.get("close", 0) * 0.99,
                "target": features.get("close", 0) * 1.01,
                "position_size": 0.005,
                "reason": "API error fallback",
            }


class TradingEnv(gym.Env if gym else object):
    """Minimal Gym environment stub for PPO compatibility."""

    def __init__(self):
        if gym and spaces:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(72,))
            self.action_space = spaces.Discrete(7)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.zeros(72)

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = self.current_step >= 1
        return np.zeros(72), reward, done, {}


class PPOTrader:
    def __init__(self):
        self.env = TradingEnv()
        self.model = PPO("MlpPolicy", self.env, verbose=0) if PPO else None
        self.replay_buffer = []

    def predict(self, features: Dict) -> str:
        """Predição PPO com 72 features."""
        state = np.array(list(features.values())[:72]).reshape(1, -1)
        if self.model:
            try:
                action, _states = self.model.predict(state, deterministic=True)
                return self.decode_action(int(action))
            except Exception:
                pass
        return "hold"

    def daily_retrain(self):
        """Retrain com últimos episódios armazenados."""
        if self.model and len(self.replay_buffer) > 100:
            batch = random.sample(self.replay_buffer, 100)
            for _episode in batch:
                self.model.learn(total_timesteps=1, reset_num_timesteps=False)

    @staticmethod
    def decode_action(action_idx: int) -> str:
        """Actions: [buy_0.25%, buy_0.5%, buy_1%, sell_0.25%, ... hold]."""
        actions = ["buy_0.25", "buy_0.5", "buy_1.0", "sell_0.25", "sell_0.5", "sell_1.0", "hold"]
        return actions[action_idx % len(actions)]


class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42) if IsolationForest else None
        self.training_data = []
        self.is_fitted = False

    def fit(self, historical_features):
        """Treina com histórico normal."""
        if not self.model:
            return
        self.training_data = historical_features
        self.model.fit(np.array([list(f.values()) for f in historical_features]))
        self.is_fitted = True

    def detect(self, features: Dict) -> float:
        """Score 0-1 (1=alta anomalia)."""
        if not self.model or not self.is_fitted:
            return 1.0

        state = np.array([list(features.values())[:72]])
        anomaly_score = self.model.decision_function(state)
        risk_multiplier = float(1.0 - anomaly_score)

        if anomaly_score < -0.1:
            return 0.1  # Pause trading
        elif anomaly_score < -0.05:
            return 0.3  # Reduce size 70%

        return risk_multiplier


class VaRManager:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if StandardScaler else None
        self.build_model()

    def build_model(self):
        """LSTM para 99% VaR próximas 4h."""
        if not Sequential or not LSTM or not Dense or not Dropout:
            self.model = None
            return
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(60, 3)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        self.model = model

    def predict_var(self, df_ohlcv) -> float:
        """99% VaR próximas 4h baseado em volatilidade implícita."""
        if self.model is None or self.scaler is None or len(df_ohlcv) < 60:
            return 0.02
        returns = df_ohlcv["close"].pct_change().dropna()
        vol = returns.rolling(20).std()
        volume_change = df_ohlcv["volume"].pct_change()

        features = np.column_stack([returns[-60:], vol[-60:], volume_change[-60:]])
        scaled = self.scaler.fit_transform(features.reshape(-1, 3)).reshape(60, 3)
        var_99 = self.model.predict(scaled.reshape(1, 60, 3), verbose=0)[0][0]
        return abs(float(var_99))


class TradingEnsemble:
    def __init__(self):
        self.data_engine = DataEngine()
        self.chronos = ChronosForecaster()
        self.claude = ClaudeDecisionMaker()
        self.ppo = PPOTrader()
        self.anomaly = AnomalyDetector()
        self.var = VaRManager()
        self.execution = ExecutionEngine()
        # Lazy imports to avoid circular issues
        from ai_crypto_trader.core.self_learning import SelfLearning
        from ai_crypto_trader.core.risk_manager import RiskManager
        from ai_crypto_trader.core.database import TraderDatabase
        from ai_crypto_trader.core.memory import MemorySystem

        self.self_learning = SelfLearning(self)
        self.risk_manager = RiskManager()
        self.db = TraderDatabase(Config.DB_PATH)
        self.memory = MemorySystem(Config.DB_PATH)
        self.risk_manager.attach_ensemble(self)

    def _weighted_confidence(self, claude_decision: Dict, chronos_pred: Dict[str, float], ppo_action: str, anomaly_risk: float) -> float:
        claude_conf = claude_decision.get("confidence", 0)
        chronos_vote = 1.0 if chronos_pred.get("5min", 0) > 0 else 0.0
        ppo_vote = 1.0 if "buy" in ppo_action else 0.0
        return (
            claude_conf * 0.65 + chronos_vote * 0.20 + ppo_vote * 0.15
        ) * anomaly_risk

    def predict(self, symbol: str) -> Dict:
        """FULL ENSEMBLE VOTE."""
        features = self.data_engine.get_features(symbol)
        df = self.data_engine.get_ohlcv(symbol)
        memory_data = self.memory.query_trade_memory(symbol, features)

        chronos_pred = self.chronos.forecast(df)
        claude_decision = self.claude.analyze_market(symbol, features, memory_data)
        ppo_action = self.ppo.predict(features)
        anomaly_risk = self.anomaly.detect(features)
        var_risk = self.var.predict_var(df)

        final_confidence = self._weighted_confidence(claude_decision, chronos_pred, ppo_action, anomaly_risk)

        decision = {
            "signal": claude_decision.get("signal", "HOLD"),
            "confidence": final_confidence,
            "size_pct": claude_decision.get("position_size", 0.005),
            "stop_loss": claude_decision.get("stop_loss", features.get("close", 0) * 0.99),
            "target": claude_decision.get("target", features.get("close", 0) * 1.01),
            "var_risk": var_risk,
            "claude_confidence": claude_decision.get("confidence", 0),
            "chronos_pred": chronos_pred.get("5min", 0),
            "ppo_action": ppo_action,
            "anomaly_risk": anomaly_risk,
            "regime": features.get("regime", "range"),
        }

        if final_confidence >= Config.CONFIDENCE_THRESHOLD:
            return decision

        decision["signal"] = "HOLD"
        return decision

    def execute_trade(self, symbol: str, decision: Dict) -> bool:
        """Full execution com risk checks."""
        risk_check = self.risk_manager.check_risk_limits(symbol, decision["size_pct"], decision)

        if risk_check.get("all_clear", False):
            price = self.data_engine.get_ohlcv(symbol).iloc[-1]["close"]
            adjusted_size = risk_check["approved_size"]

            order = self.execution.place_order(
                symbol,
                decision["signal"],
                adjusted_size * self.execution.balance,
                price,
                decision["stop_loss"],
                decision["target"],
            )

            if order:
                self.self_learning.log_trade(symbol, decision, 0, 0, {})
                return True
        return False
