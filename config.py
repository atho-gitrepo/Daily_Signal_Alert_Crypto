import os
class Config:
    # Binance API Credentials (Optional for public data, can improve rate limits)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "") # Keep empty if not using for public data rate limits
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "") # Keep empty if not using

    # Binance Environment (True for Testnet, False for Live)
    BINANCE_TESTNET = True # Still good to specify environment for data fetching

    # Futures Specific Configuration for Data
    BINANCE_FUTURES_API_URL = "https://testnet.binancefuture.com" if BINANCE_TESTNET else "https://fapi.binance.com"

    # Market Data
    SYMBOL = "BTCUSDT"  # Trading pair for notifications
    TIMEFRAME = "15min"    # Candlestick interval

    # Polling Interval (how often the bot checks for new candles/signals)
    POLLING_INTERVAL_SECONDS = 30 # Check every 30 seconds

    # Strategy Parameters (Consolidated Trend - same as before)
    TDI_RSI_PERIOD = 20
    TDI_PRICE_MA_PERIOD = 2
    TDI_FAST_MA_PERIOD = 7
    TDI_SLOW_MA_PERIOD = 14
    BB_PERIOD = 34
    BB_DEV = 2.0

    TDI_NO_TRADE_ZONE_START = 45
    TDI_NO_TRADE_ZONE_END = 55

    TDI_SOFT_BUY_LEVEL = 35
    TDI_HARD_BUY_LEVEL = 25
    TDI_SOFT_SELL_LEVEL = 65
    TDI_HARD_SELL_LEVEL = 75

    # Telegram Bot
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") # REPLACE WITH YOUR TOKEN
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") # REPLACE WITH YOUR CHAT ID
