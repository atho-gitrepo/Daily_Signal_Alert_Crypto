import os

class Config:
    # ------------------- Binance API Credentials -------------------
    # These are loaded from environment variables for security.
    # Set them in your Railway project.
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

    # --------------------- Binance Environment ---------------------
    # Set to 'True' for Testnet, 'False' for Live (Production).
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "False").lower() in ("true", "1", "t")

    # The base URL for the Binance Futures API.
    # It automatically selects the Testnet or Live URL based on BINANCE_TESTNET.
    BINANCE_FUTURES_API_URL: str = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"

    # ------------------------ Market Data --------------------------
    SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT") # Trading pair
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m") # Candlestick interval (e.g., '1m', '15m', '1h')

    # ----------------------- Polling Interval ----------------------
    POLLING_INTERVAL_SECONDS: int = int(os.getenv("POLLING_INTERVAL_SECONDS", "30")) # How often to check for new data

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
