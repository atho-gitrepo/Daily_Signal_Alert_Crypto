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
    Optimized for multi-symbol trading and the Super BB + TDI strategy.
    """

    # ------------------- Binance API Credentials -------------------
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "False").lower() in ("true", "1", "t")

    RUN_MODE: str = os.getenv("RUN_MODE", "PRODUCTION").upper()
    
    # ------------------------ Market Data --------------------------
    
    QUOTE_ASSET: str = os.getenv("QUOTE_ASSET", "USDT")
    MAX_SYMBOLS: int = safe_int_env("MAX_SYMBOLS", 30) # Increased capacity for more symbols

    # High-liquidity symbols
    SYMBOLS: List[str] = os.getenv(
        "SYMBOLS", 
        (
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,"
            "DOTUSDT,TRXUSDT,BCHUSDT,LTCUSDT,UNIUSDT,NEARUSDT,"
            "ETCUSDT,XLMUSDT,APTUSDT,SUIUSDT,IMXUSDT,"
            "FILUSDT,ATOMUSDT,VETUSDT"
        )
    ).upper().split(',')
    
    # CRITICAL: Set the timeframe (Ensure this is appropriate for your strategy)
    # The TDI/BB strategy is typically used on H4/H1, but H1 is set here.
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m") 

    # ----------------------- Polling & API Control ----------------------
    
    POLLING_INTERVAL_SECONDS: int = safe_int_env("POLLING_INTERVAL_SECONDS", 60)
    API_TIMEOUT_SECONDS: int = safe_int_env("API_TIMEOUT_SECONDS", 10)
    
    # --------------------- Strategy Parameters (Confirming Strict Rules) ---------------------
    
    # TDI Parameters (Must match the Pine Script logic)
    TDI_RSI_PERIOD: int = safe_int_env("TDI_RSI_PERIOD", 10)
    TDI_BB_LENGTH: int = safe_int_env("TDI_BB_LENGTH", 20)
    TDI_FAST_MA_PERIOD: int = safe_int_env("TDI_FAST_MA_PERIOD", 1) # Green Line (Bulls)
    TDI_SLOW_MA_PERIOD: int = safe_int_env("TDI_SLOW_MA_PERIOD", 5) # Red Line (Bears)
    
    # Super Bollinger Band Parameters
    BB_PERIOD: int = safe_int_env("BB_PERIOD", 34)
    BB_DEV: float = safe_float_env("BB_DEV", 1.750)
    BB_TREND_PERIOD: int = safe_int_env("BB_TREND_PERIOD", 9) # For SMMA approximation

    # TDI Trade Zones (Match the 25/35/50/65/75 levels)
    TDI_CENTER_LINE: float = safe_float_env("TDI_CENTER_LINE", 50.0) 
    TDI_SOFT_BUY_LEVEL: float = safe_float_env("TDI_SOFT_BUY_LEVEL", 35.0)
    TDI_HARD_BUY_LEVEL: float = safe_float_env("TDI_HARD_BUY_LEVEL", 25.0)
    TDI_SOFT_SELL_LEVEL: float = safe_float_env("TDI_SOFT_SELL_LEVEL", 65.0)
    TDI_HARD_SELL_LEVEL: float = safe_float_env("TDI_HARD_SELL_LEVEL", 75.0)
    
    # --- NEW: Additional Indicators for Strategy Confirmation ---
    
    # 200 EMA for Trend Filter
    EMA_200_PERIOD: int = safe_int_env("EMA_200_PERIOD", 200)

    # Stochastic Oscillator for Momentum/Oversold/Overbought
    STOCH_K_PERIOD: int = safe_int_env("STOCH_K_PERIOD", 14)
    STOCH_D_PERIOD: int = safe_int_env("STOCH_D_PERIOD", 3)
    STOCH_SMOOTHING: int = safe_int_env("STOCH_SMOOTHING", 3)
    
    # --------------------- Risk Management -------------------------
    
    MAX_TOTAL_RISK_CAPITAL_PERCENT: float = safe_float_env("MAX_TOTAL_RISK_CAPITAL_PERCENT", 10.0) 
    RISK_PER_TRADE_PERCENT: float = safe_float_env("RISK_PER_TRADE_PERCENT", 0.5)
    
    # ------------------------ Telegram Bot -------------------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
