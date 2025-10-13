import time
import logging
from settings import Config
from utils.telegram_bot import send_telegram_message_sync as send_telegram_message
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd

# Import your strategy
from strategy.consolidated_trend import ConsolidatedTrendStrategy  # ← Add this import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class BinanceDataClient:
    """Client for fetching real-time and historical data from Binance USD-M Futures."""

    def __init__(self):
        self.symbol = Config.SYMBOL
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.is_testnet = Config.BINANCE_TESTNET

        if not self.api_key or not self.api_secret:
            logger.warning("⚠️ Binance API Key/Secret not set. Using public endpoints (rate-limited).")

        # ✅ FIXED: Use keyword arguments instead of positional arguments
        base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
        
        if self.api_key and self.api_secret:
            self.futures_client = UMFutures(
                key=self.api_key,
                secret=self.api_secret,
                base_url=base_url
            )
        else:
            # Public client without authentication
            self.futures_client = UMFutures(base_url=base_url)

        self.price_precision = 2
        self._get_symbol_precision()
        logger.info(f"✅ Binance Data Client initialized. Testnet: {self.is_testnet}")

    def _get_symbol_precision(self):
        """Fetch price precision from exchange info."""
        try:
            info = self.futures_client.exchange_info()
            symbol_info = next(
                (s for s in info['symbols'] if s['symbol'] == self.symbol),
                None
            )
            if symbol_info:
                price_filter = next(
                    (f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'),
                    None
                )
                if price_filter:
                    step_size = price_filter['tickSize']
                    self.price_precision = len(step_size.split('.')[-1].rstrip('0'))
            logger.info(f"ℹ️ Price precision for {self.symbol}: {self.price_precision}")
        except Exception as e:
            logger.error(f"❌ Could not fetch symbol precision. Using default {self.price_precision}. Error: {e}")

    def _round_price(self, price: float) -> float:
        return round(price, self.price_precision) if price else 0.0

    def get_historical_klines(self, symbol: str = None, interval: str = None, limit: int = 500) -> pd.DataFrame:
        """Fetch historical klines (OHLCV) from Binance USD-M Futures."""
        symbol = symbol or self.symbol
        interval = interval or Config.TIMEFRAME
        try:
            klines = self.futures_client.klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df.set_index('close_time', inplace=True)
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
            logger.info(f"📊 Fetched {len(df)} klines for {symbol}.")
            return df
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"❌ Binance Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching klines: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float | None:
        """Fetch current market price."""
        try:
            ticker = self.futures_client.ticker_price(symbol=self.symbol)
            return self._round_price(float(ticker['price']))
        except Exception as e:
            logger.error(f"❌ Failed to fetch current price: {e}")
            return None


def escape_markdown(text: str) -> str:
    """Escape Telegram MarkdownV2-sensitive characters."""
    # List of characters that need escaping in MarkdownV2
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text


def safe_send_telegram_message(message: str):
    """Safely send Telegram message with error handling."""
    try:
        escaped_message = escape_markdown(message)
        send_telegram_message(escaped_message)
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram message: {e}")


def format_signal_message(signal_type: str, signal_data: dict, current_price: float) -> str:
    """Format trading signal for Telegram notification."""
    if signal_type == "NO_TRADE":
        return f"⚪ No Trade Signal | Price: {current_price}"
    
    entry = signal_data.get('entry_price', current_price)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    risk_factor = signal_data.get('risk_factor', 1.0)
    tdi_value = signal_data.get('tdi_value', 0)
    
    if signal_type == "BUY":
        emoji = "🟢"
        action = "LONG"
    else:  # SELL
        emoji = "🔴" 
        action = "SHORT"
    
    return (
        f"{emoji} *{action} SIGNAL* {emoji}\n"
        f"💰 Entry: ${entry:.2f}\n"
        f"🛑 Stop Loss: ${sl:.2f}\n"
        f"🎯 Take Profit: ${tp:.2f}\n"
        f"📊 Current: ${current_price:.2f}\n"
        f"⚖️ Risk Factor: {risk_factor}x\n"
        f"📈 TDI RSI: {tdi_value:.2f}"
    )


def main():
    """Main entry point for Binance Data Client with Trading Strategy."""
    logger.info("🚀 Starting Binance Data Client with Consolidated Trend Strategy...")

    try:
        # Initialize clients
        client = BinanceDataClient()
        
        # ✅ Initialize your trading strategy
        strategy = ConsolidatedTrendStrategy()
        
        safe_send_telegram_message("✅ Binance Data Client & Consolidated Trend Strategy started successfully")

        # Track last signal to avoid spam
        last_signal = None
        signal_cooldown = 0  # Cooldown counter

        while True:
            # Get current price and historical data
            current_price = client.get_current_price()
            historical_data = client.get_historical_klines(limit=100)  # Get enough data for indicators
            
            if current_price and not historical_data.empty:
                # ✅ Analyze data with strategy indicators
                analyzed_data = strategy.analyze_data(historical_data)
                
                if not analyzed_data.empty:
                    # ✅ Generate trading signal
                    signal_type, signal_data = strategy.generate_signal(analyzed_data)
                    
                    # Only send signal if it's new and not in cooldown
                    if signal_type != "NO_TRADE" and (signal_type != last_signal or signal_cooldown <= 0):
                        message = format_signal_message(signal_type, signal_data, current_price)
                        logger.info(f"🎯 Strategy Signal: {signal_type}")
                        safe_send_telegram_message(message)
                        last_signal = signal_type
                        signal_cooldown = 5  # Set cooldown (5 iterations)
                    
                    # Decrement cooldown if active
                    if signal_cooldown > 0:
                        signal_cooldown -= 1
                        
                    # Log current status (less verbose)
                    if signal_type == "NO_TRADE":
                        logger.info(f"⚪ No trade signal | Price: {current_price}")
                else:
                    logger.warning("⚠️ Analyzed data is empty after processing")
            else:
                logger.warning("⚠️ Missing price data or historical data for analysis")

            time.sleep(Config.POLLING_INTERVAL_SECONDS)

    except Exception as e:
        raw_error = f"🔥 Critical error in main loop: {e}"
        logger.critical(raw_error)
        safe_send_telegram_message(escape_markdown(raw_error))


if __name__ == "__main__":
    main()