import hashlib
import json
import sqlite3
from typing import Dict

import numpy as np

from ai_crypto_trader.core.database import TraderDatabase

try:
    import chromadb  # type: ignore
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None


class MemorySystem:
    def __init__(self, db_path: str = "trader.db"):
        self.db = TraderDatabase(db_path)
        self.cache: Dict = {}
        self.vector_store = self.init_vector_store()

    def init_vector_store(self):
        """ChromaDB local para similaridade semântica."""
        if not chromadb:
            return None
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            collection = client.get_collection("trade_states")
        except Exception:
            collection = client.create_collection("trade_states")
        return collection

    def query_trade_memory(self, symbol: str, features: Dict) -> Dict:
        """Busca memória completa: vector + patterns + regime stats."""
        results = {}

        feature_vec = np.array(list(features.values())[:72])
        embedding = self.create_embedding(feature_vec)

        if self.vector_store:
            similar_trades = self.vector_store.query(
                query_embeddings=[embedding], n_results=50, where={"symbol": symbol}
            )
            results["similar_count"] = len(similar_trades.get("ids", []))
        else:
            results["similar_count"] = 0

        features_hash = hashlib.md5(feature_vec.tobytes()).hexdigest()
        winrate, sample_size = self.db.get_pattern_winrate(symbol, features_hash)
        results["pattern_winrate"] = winrate
        results["pattern_sample"] = sample_size

        regime = self.classify_regime(features)
        regime_stats = self.get_regime_stats(symbol, regime)
        results["regime_sharpe"] = regime_stats.get("sharpe", 1.0)

        recent_winrate = self.get_recent_winrate(symbol, 20)
        results["recent_winrate"] = recent_winrate

        return results

    def create_embedding(self, features):
        """Embedding 384d para similaridade."""
        if SentenceTransformer:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode([str(features)]).flatten()
        # Fallback: normalized numeric vector
        arr = np.array(features, dtype=float)
        norm = np.linalg.norm(arr) + 1e-9
        return (arr / norm).tolist()

    def classify_regime(self, features: Dict) -> str:
        """HMM stub: trend/range/choppy."""
        rsi = features.get("rsi_14", 50)
        adx = features.get("adx_14", 20)
        hv = features.get("hv_20", 0)

        if adx > 25 and abs(rsi - 50) > 20:
            return "trending"
        elif hv > 0.03:
            return "choppy"
        else:
            return "range"

    def get_regime_stats(self, symbol: str, regime: str) -> Dict:
        """Stats regime últimos 30d (stub)."""
        return {"sharpe": 1.8, "winrate": 0.65}

    def get_recent_winrate(self, symbol: str, n_trades: int = 20) -> float:
        """Winrate últimos N trades."""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) as total, SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins
            FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?
        """,
            (symbol, n_trades),
        )
        result = cursor.fetchone()
        conn.close()
        if result and result[0] > 0:
            return result[1] / result[0]
        return 0.5

    def log_episode(self, state, action, reward, next_state):
        """PPO replay buffer."""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rl_episodes (state_features, action, reward, next_state, done)
            VALUES (?, ?, ?, ?, ?)
        """,
            (json.dumps(state), action, reward, json.dumps(next_state), False),
        )
        conn.commit()
        conn.close()
