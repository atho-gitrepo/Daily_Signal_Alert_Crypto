# settings.py
import os

class Config:
    """Configuration settings for Binance data client and strategy."""

    # ------------------- Binance API Credentials -------------------
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "False").lower() in ("true", "1", "t")

    # ------------------------ Market Data --------------------------
    # ❌ OLD LINE: SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")  # Trading pair
    
    # ✅ NEW LINE: Define a list of symbols for high-frequency checks
    # The list is defined as a comma-separated string in the environment,
    # and splits into a list here. Using ETH and SOL first for potentially cleaner signals.
    SYMBOLS: list[str] = os.getenv("SYMBOLS", "ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,SUIUSDT").split(',')
    
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")  # Candlestick interval

    # ----------------------- Polling Interval ----------------------
    POLLING_INTERVAL_SECONDS: int = int(os.getenv("POLLING_INTERVAL_SECONDS", "60"))

    # --------------------- Strategy Parameters ---------------------
    TDI_RSI_PERIOD: int = 20
    TDI_PRICE_MA_PERIOD: int = 2
    TDI_FAST_MA_PERIOD: int = 7
    TDI_SLOW_MA_PERIOD: int = 14
    BB_PERIOD: int = 34
    BB_DEV: float = 2.0

    TDI_NO_TRADE_ZONE_START: float = 45.0
    TDI_NO_TRADE_ZONE_END: float = 55.0
    TDI_SOFT_BUY_LEVEL: float = 35.0
    TDI_HARD_BUY_LEVEL: float = 25.0
    TDI_SOFT_SELL_LEVEL: float = 65.0
    TDI_HARD_SELL_LEVEL: float = 75.0

    # ------------------------ Telegram Bot -------------------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
