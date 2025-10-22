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
    Enhanced for multi-symbol trading, risk management, and operational robustness.
    """

    # ------------------- Binance API Credentials -------------------
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "False").lower() in ("true", "1", "t")

    # Idea 3: Run Mode for safety and context
    RUN_MODE: str = os.getenv("RUN_MODE", "PRODUCTION").upper()
    
    # ------------------------ Market Data --------------------------
    
    # Idea 1: Quote Asset and Symbol Max
    QUOTE_ASSET: str = os.getenv("QUOTE_ASSET", "USDT")  # All symbols must end with this asset
    MAX_SYMBOLS: int = safe_int_env("MAX_SYMBOLS", 15)  # Safety limit for number of pairs to monitor

    # Define a list of symbols for high-frequency checks
    # The list is defined as a comma-separated string in the environment, and splits here.
    SYMBOLS: List[str] = os.getenv(
        "SYMBOLS", 
        "ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SUIUSDT,AVAXUSDT,LTCUSDT,TRXUSDT,LINKUSDT,UNIUSDT,XLMUSDT"
    ).upper().split(',')
    
    TIMEFRAME: str = os.getenv("TIMEFRAME", "30m")  # Candlestick interval

    # ----------------------- Polling & API Control ----------------------
    
    POLLING_INTERVAL_SECONDS: int = safe_int_env("POLLING_INTERVAL_SECONDS", 60)
    
    # Idea 3: API Timeout for operational robustness
    API_TIMEOUT_SECONDS: int = safe_int_env("API_TIMEOUT_SECONDS", 10) # Timeout for individual API requests
    
    # --------------------- Strategy Parameters ---------------------
    
    # TDI Parameters (as before)
    TDI_RSI_PERIOD: int = 20
    TDI_PRICE_MA_PERIOD: int = 2
    TDI_FAST_MA_PERIOD: int = 7
    TDI_SLOW_MA_PERIOD: int = 14
    BB_PERIOD: int = 34
    BB_DEV: float = 2.0

    # TDI Trade Zones (as before)
    TDI_NO_TRADE_ZONE_START: float = 45.0
    TDI_NO_TRADE_ZONE_END: float = 55.0
    TDI_SOFT_BUY_LEVEL: float = 35.0
    TDI_HARD_BUY_LEVEL: float = 25.0
    TDI_SOFT_SELL_LEVEL: float = 65.0
    TDI_HARD_SELL_LEVEL: float = 75.0
    
    # Idea 2: Multi-Symbol Risk Management
    # These settings help control exposure across multiple simultaneous trades
    MAX_TOTAL_RISK_CAPITAL_PERCENT: float = safe_float_env("MAX_TOTAL_RISK_CAPITAL_PERCENT", 10.0) # Max 10% of portfolio open at once
    RISK_PER_TRADE_PERCENT: float = safe_float_env("RISK_PER_TRADE_PERCENT", 1.0) # Risk 1% of total capital per trade
    
    # ------------------------ Telegram Bot -------------------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
