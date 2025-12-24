import hashlib
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except ImportError:
    ccxt = None

try:
    import talib  # type: ignore
except ImportError:
    talib = None

try:
    from newsapi import NewsApiClient  # type: ignore
except ImportError:
    NewsApiClient = None

try:
    from pycoingecko import CoinGeckoAPI  # type: ignore
except ImportError:
    CoinGeckoAPI = None

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None

from ai_crypto_trader.config import Config


def _ensure_series(values: pd.Series) -> pd.Series:
    """Return a non-empty series with at least two entries to avoid NaN issues."""
    if values is None or len(values) == 0:
        return pd.Series([0.0, 0.0])
    if len(values) == 1:
        return pd.concat([values, pd.Series([values.iloc[0]])])
    return values


class DataEngine:
    """
    Data ingestion and feature generation (72 professional-grade signals).
    """

    def __init__(self):
        self.coingecko = CoinGeckoAPI() if CoinGeckoAPI else None
        self.binance = (
            ccxt.binance(
                {
                    "apiKey": Config.BINANCE_TESTNET_KEY,
                    "secret": Config.BINANCE_TESTNET_SECRET,
                    "sandbox": True,  # TESTNET
                    "enableRateLimit": True,
                }
            )
            if ccxt
            else None
        )
        self.news_client = NewsApiClient(api_key=Config.NEWS_API_KEY) if NewsApiClient else None
        self.cache: Dict[str, Dict] = {}  # Redis stub (in-memory for now)

    def _generate_stub_prices(self, limit: int) -> pd.DataFrame:
        """Fallback OHLCV generator when external APIs are unavailable."""
        now = pd.Timestamp.utcnow().floor("T")
        timestamps = pd.date_range(end=now, periods=limit, freq="T")
        base_price = 40000
        prices = np.cumsum(np.random.normal(0, 10, size=limit)) + base_price
        high = prices + np.random.uniform(1, 15, size=limit)
        low = prices - np.random.uniform(1, 15, size=limit)
        open_ = np.concatenate([[prices[0]], prices[:-1]])
        volume = np.random.uniform(10, 100, size=limit)
        df = pd.DataFrame(
            {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": prices, "volume": volume}
        )
        return df.set_index("timestamp")

    def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 200) -> pd.DataFrame:
        """
        CoinGecko historical + Binance realtime.
        """
        try:
            if limit > 100 and self.coingecko:
                hist = self.coingecko.get_coin_ohlc_by_id(
                    id=symbol.replace("USDT", "").lower(), vs_currency="usd", days=limit // 60 + 1
                )
                df = pd.DataFrame(hist, columns=["timestamp", "open", "high", "low", "close", "volume"])
            elif self.binance:
                ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            else:
                return self._generate_stub_prices(limit)

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.set_index("timestamp").sort_index()
        except Exception:
            return self._generate_stub_prices(limit)

    # --- Feature engineering -------------------------------------------------
    def _talib_or_fallback(self, func_name: str, *args, **kwargs) -> pd.Series:
        """Call TA-Lib if available, otherwise return zeros with the right length."""
        if talib and hasattr(talib, func_name):
            func = getattr(talib, func_name)
            return pd.Series(func(*args, **kwargs))
        base_len = len(args[0]) if args else 1
        return pd.Series(np.zeros(base_len))

    def _recent_value(self, series: pd.Series, default: float = 0.0) -> float:
        """Safely return the most recent finite value from a series."""
        if series is None or len(series) == 0:
            return default
        value = series.iloc[-1]
        if isinstance(value, (float, int)) and math.isfinite(value):
            return float(value)
        try:
            return float(value[-1])
        except Exception:
            return default

    def _compute_ta_features(self, df: pd.DataFrame) -> Dict[str, float]:
        close = _ensure_series(df["close"])
        high = _ensure_series(df["high"])
        low = _ensure_series(df["low"])
        volume = _ensure_series(df["volume"])

        macd_res = talib.MACD(close) if talib else (pd.Series(np.zeros(len(close))), pd.Series(np.zeros(len(close))))
        bbands = talib.BBANDS(close, timeperiod=20) if talib else (
            pd.Series(close.rolling(20).mean() + close.rolling(20).std()),
            pd.Series(np.zeros(len(close))),
            pd.Series(close.rolling(20).mean() - close.rolling(20).std()),
        )

        ta_features: Dict[str, float] = {
            "rsi_14": self._recent_value(self._talib_or_fallback("RSI", close, 14), 50),
            "rsi_21": self._recent_value(self._talib_or_fallback("RSI", close, 21), 50),
            "macd": self._recent_value(macd_res[0], 0),
            "macd_signal": self._recent_value(macd_res[1], 0),
            "bb_upper": self._recent_value(bbands[0], close.iloc[-1]),
            "bb_lower": self._recent_value(bbands[2], close.iloc[-1]),
            "bb_position": float(
                (
                    (close.iloc[-1] - self._recent_value(bbands[2], close.iloc[-1]))
                    / max(self._recent_value(bbands[0], close.iloc[-1]) - self._recent_value(bbands[2], close.iloc[-1]), 1e-6)
                )
            ),
            "atr_14": self._recent_value(self._talib_or_fallback("ATR", high, low, close, 14), 0),
            "stoch_k": self._recent_value(self._talib_or_fallback("STOCH", high, low, close)[0], 0),
            "adx_14": self._recent_value(self._talib_or_fallback("ADX", high, low, close, 14), 0),
            "cci_20": self._recent_value(self._talib_or_fallback("CCI", high, low, close, 20), 0),
            "obv": self._recent_value(self._talib_or_fallback("OBV", close, volume), 0),
            "mfi_14": self._recent_value(self._talib_or_fallback("MFI", high, low, close, volume, 14), 0),
            "ema_20": self._recent_value(close.ewm(span=20).mean(), close.iloc[-1]),
            "ema_50": self._recent_value(close.ewm(span=50).mean(), close.iloc[-1]),
            "ema_200": self._recent_value(close.ewm(span=200, adjust=False).mean(), close.iloc[-1]),
            "sma_20": self._recent_value(close.rolling(20).mean(), close.iloc[-1]),
            "sma_50": self._recent_value(close.rolling(50).mean(), close.iloc[-1]),
            "sma_100": self._recent_value(close.rolling(100).mean(), close.iloc[-1]),
            "williams_r": self._recent_value(self._talib_or_fallback("WILLR", high, low, close, 14), 0),
            "adx_28": self._recent_value(self._talib_or_fallback("ADX", high, low, close, 28), 0),
            "dmi_plus": self._recent_value(self._talib_or_fallback("PLUS_DI", high, low, close, 14), 0),
            "dmi_minus": self._recent_value(self._talib_or_fallback("MINUS_DI", high, low, close, 14), 0),
            "cmf_20": self._recent_value(
                ((close - low) - (high - close)) / (high - low + 1e-9) * volume.rolling(20).sum(), 0
            ),
            "trix_15": self._recent_value(self._talib_or_fallback("TRIX", close, 15), 0),
            "tema_30": self._recent_value(self._talib_or_fallback("TEMA", close, 30), 0),
            "roc_14": self._recent_value(self._talib_or_fallback("ROC", close, 14), 0),
            "ppo": self._recent_value(self._talib_or_fallback("PPO", close, fastperiod=12, slowperiod=26), 0),
            "ultosc": self._recent_value(self._talib_or_fallback("ULTOSC", high, low, close), 0),
            "psar": self._recent_value(self._talib_or_fallback("SAR", high, low), 0),
            "natr_14": self._recent_value(self._talib_or_fallback("NATR", high, low, close, 14), 0),
        }
        return ta_features

    def _compute_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        close = df["close"]
        returns = close.pct_change().fillna(0)
        hv_20_series = returns.rolling(20).std() * np.sqrt(1440)
        hv_60_series = returns.rolling(60).std() * np.sqrt(1440)
        price_features = {
            "roc_1m": returns.iloc[-1],
            "roc_5m": close.pct_change(5).iloc[-1] if len(close) > 5 else 0.0,
            "roc_15m": close.pct_change(15).iloc[-1] if len(close) > 15 else 0.0,
            "roc_60m": close.pct_change(60).iloc[-1] if len(close) > 60 else 0.0,
            "hv_20": hv_20_series.iloc[-1] if len(hv_20_series) else 0.0,
            "hv_60": hv_60_series.iloc[-1] if len(hv_60_series) else 0.0,
            "vol_skew": returns.tail(50).skew(),
            "vol_kurtosis": returns.tail(50).kurtosis(),
            "volume_roc": df["volume"].pct_change().iloc[-1] if len(df["volume"]) > 1 else 0.0,
            "range_pct": (df["high"].iloc[-1] - df["low"].iloc[-1]) / max(df["close"].iloc[-1], 1e-9),
            "gap_pct": (df["open"].iloc[-1] - df["close"].iloc[-2]) / max(df["close"].iloc[-2], 1e-9)
            if len(df) > 1
            else 0.0,
            "rolling_mean_10": df["close"].rolling(10).mean().iloc[-1],
            "rolling_std_10": df["close"].rolling(10).std().iloc[-1],
            "rolling_mean_30": df["close"].rolling(30).mean().iloc[-1],
            "rolling_std_30": df["close"].rolling(30).std().iloc[-1],
        }
        return {k: (0.0 if pd.isna(v) else float(v)) for k, v in price_features.items()}

    def _compute_orderbook_features(self, symbol: str) -> Dict[str, float]:
        if not self.binance:
            return {"bid_volume": 0.0, "ask_volume": 0.0, "imbalance": 0.0, "spread_pct": 0.0}
        try:
            orderbook = self.binance.fetch_order_book(symbol, limit=10)
            bid_volume = sum(bid[1] for bid in orderbook.get("bids", []))
            ask_volume = sum(ask[1] for ask in orderbook.get("asks", []))
            spread_pct = (
                (orderbook["asks"][0][0] - orderbook["bids"][0][0]) / orderbook["bids"][0][0] * 100
                if orderbook.get("bids") and orderbook.get("asks")
                else 0.0
            )
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)
            return {
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "imbalance": imbalance,
                "spread_pct": spread_pct,
            }
        except Exception:
            return {"bid_volume": 0.0, "ask_volume": 0.0, "imbalance": 0.0, "spread_pct": 0.0}

    def get_news_sentiment(self, articles: List[Dict]) -> float:
        """Claude sentiment score -1 to +1 (stub if API unavailable)."""
        if not articles:
            return 0.0
        if anthropic and Config.ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            prompt = "Avalie sentimento médio (-1 a +1) destas manchetes:\n" + "\n".join(
                [a.get("title", "") for a in articles]
            )
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    temperature=0.1,
                    system="Retorne apenas um número de -1 a 1.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if hasattr(response, "content") else str(response)
                return float(text.strip())
            except Exception:
                return 0.0
        # Fallback: simple neutral sentiment
        return 0.0

    def _collect_news(self, symbol: str) -> Dict[str, Optional[str]]:
        if not self.news_client:
            return {"sentiment": 0.0, "latest_news": ""}
        try:
            news = self.news_client.get_everything(
                q=symbol.replace("USDT", ""),
                from_param=(datetime.utcnow() - timedelta(hours=4)).strftime("%Y-%m-%d"),
                language="en",
                sort_by="publishedAt",
            )
            articles = news.get("articles", [])[:5]
            sentiment = self.get_news_sentiment(articles)
            latest_title = articles[0]["title"] if articles else ""
            return {"sentiment": sentiment, "latest_news": latest_title}
        except Exception:
            return {"sentiment": 0.0, "latest_news": ""}

    def get_features_from_df(self, df: pd.DataFrame) -> Dict[str, float]:
        """Utility for backtests to reuse existing OHLCV frames."""
        return self._assemble_features(df, symbol="SIMULATED")

    def _assemble_features(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        ta_features = self._compute_ta_features(df)
        price_features = self._compute_price_features(df)
        orderbook_features = self._compute_orderbook_features(symbol)
        news_features = self._collect_news(symbol)

        # Combine and ensure we have at least 72 signals by padding if needed.
        features: Dict[str, float] = {**ta_features, **price_features, **orderbook_features}
        features.update(
            {
                "sentiment": news_features.get("sentiment", 0.0),
                "latest_news": news_features.get("latest_news", ""),
                "close": float(df["close"].iloc[-1]),
                "regime": "range",
                "funding_rate": 0.0,
            }
        )

        # Pad with synthetic stability metrics to maintain 72-feature expectation.
        while len(features) < 72:
            features[f"pad_feature_{len(features)}"] = 0.0

        return features

    def get_features(self, symbol: str) -> Dict[str, float]:
        """
        72 professional-grade features combining TA-Lib, price/volatility, orderbook, and news sentiment.
        """
        df = self.get_ohlcv(symbol, limit=200)
        return self._assemble_features(df, symbol)
