# settings.py 
import os
from typing import List

# Helper function for safe float conversion with a default
def safe_float_env(key: str, default: float) -> float:
    """Safely gets an environment variable as a float, using a default on failure."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

# Helper function for safe int conversion with a default
def safe_int_env(key: str, default: int) -> int:
    """Safely gets an environment variable as an integer, using a default on failure."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

class Config:
    """
    Configuration settings for Binance data client and strategy.
    Optimized for multi-symbol 5-minute (5m) scalping/trend trading.
    """

    # ------------------- Binance API Credentials -------------------
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "False").lower() in ("true", "1", "t")

    RUN_MODE: str = os.getenv("RUN_MODE", "PRODUCTION").upper()
    
    # ------------------------ Market Data --------------------------
    
    QUOTE_ASSET: str = os.getenv("QUOTE_ASSET", "USDT")
    MAX_SYMBOLS: int = safe_int_env("MAX_SYMBOLS", 30) 

    # High-liquidity symbols (approx. 20+ pairs)
    SYMBOLS: List[str] = os.getenv(
        "SYMBOLS", 
        (
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,"
            "DOTUSDT,TRXUSDT,BCHUSDT,LTCUSDT,UNIUSDT,NEARUSDT,"
            "ETCUSDT,XLMUSDT,APTUSDT,SUIUSDT,IMXUSDT,"
            "FILUSDT,ATOMUSDT,VETUSDT"
        )
    ).upper().split(',') 
    
    # CRITICAL: Set the timeframe to 5m for day trading focus
    TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")

    # ----------------------- Polling & API Control ----------------------
    
    # CRITICAL: Reduced to 30s to accurately catch 5m candle closures
    POLLING_INTERVAL_SECONDS: int = safe_int_env("POLLING_INTERVAL_SECONDS", 5)
    API_TIMEOUT_SECONDS: int = safe_int_env("API_TIMEOUT_SECONDS", 5)
    
    # --------------------- Strategy Parameters (Strict 5m Scalping) ---------------------
    
    # Traders Dynamic Index (TDI) Parameters
    TDI_RSI_PERIOD: int = 13
    TDI_BB_LENGTH: int = 34 # Slower BB on RSI for major trend filtering
    TDI_FAST_MA_PERIOD: int = 1 # Green Line 
    TDI_SLOW_MA_PERIOD: int = 7 # Red Line (Slower trend confirmation)
    
    # Super Bollinger Band Parameters (Price action)
    BB_PERIOD: int = 20 # Standard period for short-term volatility and rejection mapping.
    BB_DEV: float = 2.0 # Standard 2.0 deviation for identifying true price extremes.
    BB_TREND_PERIOD: int = 9 

    # TDI Trade Zones
    TDI_CENTER_LINE: float = 50.0 
    TDI_SOFT_BUY_LEVEL: float = 35.0
    TDI_HARD_BUY_LEVEL: float = 25.0
    TDI_SOFT_SELL_LEVEL: float = 65.0
    TDI_HARD_SELL_LEVEL: float = 75.0
    
    # --------------------- Risk Management -------------------------
    
    MAX_TOTAL_RISK_CAPITAL_PERCENT: float = safe_float_env("MAX_TOTAL_RISK_CAPITAL_PERCENT", 10.0) 
    RISK_PER_TRADE_PERCENT: float = safe_float_env("RISK_PER_TRADE_PERCENT", 0.5)
    
    # ------------------------ Telegram Bot -------------------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
