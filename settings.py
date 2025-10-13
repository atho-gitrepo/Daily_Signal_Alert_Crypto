import pandas as pd
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BinanceDataClient:
    """Client for fetching real-time and historical data from Binance USD-M Futures."""

    def __init__(self):
        self.symbol = Config.SYMBOL
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.is_testnet = Config.BINANCE_TESTNET

        if not self.api_key or not self.api_secret:
            logger.warning("Binance API Key/Secret not set. Using public endpoints (rate-limited).")

        # Initialize official Binance Client
        self.client = Client(self.api_key, self.api_secret, testnet=self.is_testnet)

        # USDⓈ-M Futures client
        self.futures_client = self.client.futures  # For USDT-M Futures

        self.price_precision = 2
        self._get_symbol_precision()
        logger.info(f"Binance Data Client initialized. Testnet: {self.is_testnet}")

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

            logger.info(f"Price precision for {self.symbol}: {self.price_precision}")
        except Exception as e:
            logger.error(f"Could not fetch symbol precision. Defaulting to {self.price_precision}. Error: {e}")

    def _round_price(self, price: float) -> float:
        return round(price, self.price_precision) if price else 0.0

    def get_historical_klines(self, symbol: str = None, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        """Fetch historical klines (OHLCV) from Binance USD-M Futures."""
        symbol = symbol or self.symbol
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
            logger.info(f"Fetched {len(df)} klines for {symbol}.")
            return df
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Binance Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching klines: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float | None:
        """Fetch current market price."""
        try:
            ticker = self.futures_client.ticker_price(symbol=self.symbol)
            return self._round_price(float(ticker['price']))
        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None
