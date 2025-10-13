import pandas as pd
import logging
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException, BinanceRequestException
from settings import Config

logger = logging.getLogger(__name__)

class BinanceDataClient:
    """
    Client for fetching REAL data (klines and price) from Binance USD-M Futures.
    Uses the dedicated UMFutures client to avoid the 'futures' attribute error.
    """
    def __init__(self):
        self.symbol = Config.SYMBOL
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.is_testnet = Config.BINANCE_TESTNET

        # ✅ FIXED: Correct positional arguments for UMFutures
        base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
        self.futures_client = UMFutures(base_url, self.api_key, self.api_secret)

        self.price_precision = 2

        logger.info(f"Binance Data Client initialized. Testnet: {self.is_testnet}")
        if not self.api_key:
            logger.warning("Binance API Key is NOT set. Using public endpoints (lower rate limit).")

        self._get_symbol_precision()

    def _get_symbol_precision(self):
        """Fetches the price precision from exchange info."""
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

        except Exception as e:
            logger.error(f"Could not fetch symbol precision. Defaulting to {self.price_precision}. Error: {e}")

    def _round_price(self, price: float) -> float:
        if price is None:
            return 0.0
        return round(price, self.price_precision)

    def get_historical_klines(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetches REAL historical klines from Binance Futures."""
        try:
            klines = self.futures_client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df.set_index('close_time', inplace=True)

            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

            logger.info(f"Successfully fetched {len(df)} klines from Binance.")
            return df

        except BinanceAPIException as e:
            logger.error(f"Binance API Error: {e.status_code} - {e.message}. Rate limit exceeded or invalid symbol/timeframe.")
            return pd.DataFrame()
        except BinanceRequestException as e:
            logger.error(f"Binance Request Error: Network or connection issue. {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred during klines fetch: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float | None:
        """Fetches the current market price."""
        try:
            ticker = self.futures_client.ticker_price(symbol=self.symbol)
            price = float(ticker['price'])
            return price
        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None
