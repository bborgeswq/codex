import os
from dotenv import load_dotenv

# Load environment variables as early as possible to ensure all downstream
# modules see consistent values.
load_dotenv()


class Config:
    """Centralized configuration driven by environment variables."""

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    BINANCE_TESTNET_KEY = os.getenv("BINANCE_TESTNET_KEY")
    BINANCE_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
    INITIAL_CAPITAL_USD = float(os.getenv("INITIAL_CAPITAL_USD", 10000))
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.01))  # 1%
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))
    ASSETS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    DB_PATH = os.getenv("DB_PATH", "./trader.db")


__all__ = ["Config"]
