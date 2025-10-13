# main.py
import time
import logging
from settings import Config
from utils.telegram_bot import send_telegram_message_sync as send_telegram_message
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd

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

        # ✅ Correct UMFutures initialization
        base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
        self.futures_client = UMFutures(base_url, self.api_key, self.api_secret)

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


def main():
    """Main entry point for Binance Data Client."""
    logger.info("🚀 Starting Binance Data Client container...")

    try:
        client = BinanceDataClient()
        send_telegram_message("✅ Binance Data Client started successfully!")

        while True:
            price = client.get_current_price()
            if price:
                msg = f"💹 {client.symbol} current price: {price} USD"
                logger.info(msg)
                send_telegram_message(msg)
            else:
                logger.warning("⚠️ Failed to fetch price.")

            time.sleep(Config.POLLING_INTERVAL_SECONDS)

    except Exception as e:
        # Escape periods for Telegram Markdown
        escaped_error = str(e).replace(".", "\\.")
        error_msg = f"🔥 Critical error in main loop: {escaped_error}"
        logger.critical(error_msg)
        send_telegram_message(error_msg)


if __name__ == "__main__":
    main()
