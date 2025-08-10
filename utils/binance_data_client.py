# utils/binance_data_client.py
from binance.um_futures import UMFutures
from binance.error import ClientError
from config import Config
import logging
import pandas as pd
import math

logger = logging.getLogger(__name__)

class BinanceDataClient:
    def __init__(self):
        # Initialize UMFutures client. API keys are optional for public endpoints,
        # but providing them can help with higher rate limits.
        self.client = UMFutures(
            key=Config.BINANCE_API_KEY if Config.BINANCE_API_KEY else None,
            secret=Config.BINANCE_API_SECRET if Config.BINANCE_API_SECRET else None,
            base_url=Config.BINANCE_FUTURES_API_URL
        )
        self.symbol = Config.SYMBOL
        self.symbol_info = self.get_exchange_info(self.symbol)
        
        if self.symbol_info:
            logger.info(f"Binance Data client initialized for {Config.SYMBOL}. Base URL: {Config.BINANCE_FUTURES_API_URL}")
        else:
            logger.error(f"Failed to get exchange info for {Config.SYMBOL}. Data client might not function correctly.")
            # Critical for data parsing, so raise an error
            raise RuntimeError(f"Could not retrieve exchange info for {Config.SYMBOL}")
        
        # Get price precision (useful for formatting prices in notifications)
        self.price_precision = self._get_precision('PRICE_FILTER', self.symbol_info)
        
        logger.info(f"Price precision for {self.symbol}: {self.price_precision}")


    def _get_precision(self, filter_type, symbol_info):
        """Helper to get precision from symbol info."""
        if not symbol_info:
            return 0 # Default if info not available

        for f in symbol_info['filters']:
            if f['filterType'] == filter_type:
                if filter_type == 'PRICE_FILTER':
                    return int(-math.log10(float(f['tickSize'])))
        return 0

    def _round_price(self, price):
        """Rounds price to the correct precision for the symbol."""
        if self.price_precision is None:
            return price
        # Using math.floor to truncate to precision
        return math.floor(price * (10**self.price_precision)) / (10**self.price_precision) * 1.0

    def get_historical_klines(self, symbol, interval, limit=500):
        """Fetches historical candlestick data for Futures."""
        try:
            klines = self.client.klines(symbol=symbol, interval=interval, limit=limit)
            data = []
            for kline in klines:
                data.append({
                    'timestamp': kline[0],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': kline[6]
                })
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ClientError as e:
            logger.error(f"Binance API error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol):
        """Fetches the current mark price for Futures."""
        try:
            ticker = self.client.mark_price(symbol=symbol) # Using mark price for futures
            return float(ticker['markPrice'])
        except ClientError as e:
            logger.error(f"Binance API error fetching current price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def get_exchange_info(self, symbol):
        """Fetches exchange information for a specific symbol."""
        try:
            info = self.client.exchange_info()
            for s_info in info['symbols']:
                if s_info['symbol'] == symbol:
                    return s_info
            return None
        except ClientError as e:
            logger.error(f"Binance API error getting exchange info for {symbol}: {e.error_message} (Code: {e.error_code})")
            return None
        except Exception as e:
            logger.error(f"Error getting exchange info for {symbol}: {e}")
            return None

