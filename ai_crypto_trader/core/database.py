import json
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd


class TraderDatabase:
    def __init__(self, db_path: str = "trader.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Cria 5 tabelas production + índices críticos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                duration_min INTEGER,
                reason TEXT,
                model_votes TEXT,
                regime TEXT,
                confidence REAL,
                slippage_pct REAL,
                features_hash TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                date DATE,
                period TEXT,
                sharpe REAL,
                sortino REAL,
                calmar REAL,
                profit_factor REAL,
                winrate_pct REAL,
                max_dd_pct REAL,
                trades_count INTEGER,
                avg_hold_min REAL,
                PRIMARY KEY (date, period)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                state_features TEXT,
                action TEXT,
                reward REAL,
                next_state TEXT,
                done BOOLEAN
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS news_impact (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                title TEXT,
                sentiment REAL,
                price_reaction_pct REAL,
                relevance_score REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                symbol TEXT,
                features_hash TEXT,
                winrate_historic REAL,
                sample_size INTEGER,
                avg_pnl_pct REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_hash ON trades(features_hash);
            CREATE INDEX IF NOT EXISTS idx_rl_state ON rl_episodes(state_features);
            CREATE INDEX IF NOT EXISTS idx_patterns_hash ON patterns(features_hash);
            CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_impact(symbol, timestamp);
        """
        )

        conn.commit()
        conn.close()

    def log_trade(self, trade_data: Dict):
        """Log trade completo."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (symbol, side, entry_price, exit_price, size_usd, 
                              pnl_usd, pnl_pct, duration_min, reason, model_votes, 
                              regime, confidence, slippage_pct, features_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade_data.get("symbol"),
                trade_data.get("side"),
                trade_data.get("entry_price"),
                trade_data.get("exit_price"),
                trade_data.get("size_usd"),
                trade_data.get("pnl_usd"),
                trade_data.get("pnl_pct"),
                trade_data.get("duration_min"),
                trade_data.get("reason"),
                json.dumps(trade_data.get("model_votes", {})),
                trade_data.get("regime"),
                trade_data.get("confidence"),
                trade_data.get("slippage_pct"),
                trade_data.get("features_hash"),
            ),
        )
        conn.commit()
        conn.close()

    def update_metrics(self, date, period, metrics: Dict):
        """Atualiza métricas diárias."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO metrics (date, period, sharpe, sortino, calmar, 
                                           profit_factor, winrate_pct, max_dd_pct, 
                                           trades_count, avg_hold_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                date,
                period,
                metrics.get("sharpe"),
                metrics.get("sortino"),
                metrics.get("calmar"),
                metrics.get("profit_factor"),
                metrics.get("winrate_pct"),
                metrics.get("max_dd_pct"),
                metrics.get("trades_count"),
                metrics.get("avg_hold_min"),
            ),
        )
        conn.commit()
        conn.close()

    def get_pattern_winrate(self, symbol: str, features_hash: str) -> Tuple[float, int]:
        """Winrate histórico setup similar."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT winrate_historic, sample_size FROM patterns 
            WHERE symbol = ? AND features_hash = ?
        """,
            (symbol, features_hash),
        )
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0], result[1]
        return 0.5, 0

    def get_recent_trades(self, minutes: int = 1440) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"""
            SELECT * FROM trades
            WHERE timestamp > datetime('now', '-{int(minutes)} minutes')
            ORDER BY timestamp DESC
        """,
            conn,
        )
        conn.close()
        return df.to_dict("records")
