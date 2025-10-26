# settings.py
import os
import logging
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
    Optimized for multi-symbol trading and strict 15-minute strategy.
    """

    # ------------------- Logging & Core Settings -------------------
    # FIX: Added LOG_LEVEL for main.py logging setup
    LOG_LEVEL = os.getenv('LOG_LEVEL', logging.INFO)

    # ------------------- Binance API Credentials -------------------
    # Note: main.py/data_fetcher.py use these full names.
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "False").lower() in ("true", "1", "t")

    RUN_MODE: str = os.getenv("RUN_MODE", "PRODUCTION").upper()
    
    # ------------------------ Market Data --------------------------
    
    QUOTE_ASSET: str = os.getenv("QUOTE_ASSET", "USDT")
    MAX_SYMBOLS: int = safe_int_env("MAX_SYMBOLS", 30)

    SYMBOLS: List[str] = os.getenv(
        "SYMBOLS", 
        (
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,"
            "DOTUSDT,TRXUSDT,LINKUSDT,MATICUSDT,BCHUSDT,LTCUSDT,UNIUSDT,NEARUSDT,"
            "ETCUSDT,XLMUSDT,APTUSDT,OPUSDT,ARBIBUSDT,SUIUSDT,INJUSDT,IMXUSDT,"
            "FILUSDT,ATOMUSDT,VETUSDT,ICPUSDT"
        )
    ).upper().split(',')
    
    # FIX: ADDED KLINES_LIMIT to resolve the latest NameError
    KLINES_LIMIT: int = safe_int_env("KLINES_LIMIT", 300) 

    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")

    # ----------------------- Polling & API Control ----------------------
    
    POLLING_INTERVAL_SECONDS: int = safe_int_env("POLLING_INTERVAL_SECONDS", 60)
    API_TIMEOUT_SECONDS: int = safe_int_env("API_TIMEOUT_SECONDS", 10)
    
    # --------------------- Strategy Parameters ---------------------
    
    # TDI Parameters
    TDI_RSI_PERIOD: int = 10 
    TDI_BB_LENGTH: int = 20 
    TDI_FAST_MA_PERIOD: int = 1 
    TDI_SLOW_MA_PERIOD: int = 5
    
    # Super Bollinger Band Parameters
    BB_PERIOD: int = 34
    BB_STD_DEV: float = 1.750
    BB_TREND_PERIOD: int = 9 

    # TDI Trade Zones
    TDI_CENTER_LINE: float = 50.0 
    TDI_SOFT_BUY_LEVEL: float = 35.0
    TDI_HARD_BUY_LEVEL: float = 25.0
    TDI_SOFT_SELL_LEVEL: float = 65.0
    TDI_HARD_SELL_LEVEL: float = 75.0
    
    # ATR Settings
    ATR_PERIOD: int = safe_int_env("ATR_PERIOD", 14)
    ATR_MULTIPLIER: float = safe_float_env("ATR_MULTIPLIER", 1.5)
    
    # --------------------- Risk Management -------------------------
    
    MAX_TOTAL_RISK_CAPITAL_PERCENT: float = safe_float_env("MAX_TOTAL_RISK_CAPITAL_PERCENT", 10.0) 
    RISK_PER_TRADE_PERCENT: float = safe_float_env("RISK_PER_TRADE_PERCENT", 0.5)
    
    # ------------------------ Telegram Bot -------------------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
