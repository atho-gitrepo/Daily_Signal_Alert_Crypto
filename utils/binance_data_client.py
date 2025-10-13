# utils/binance_data_client.py (New File)

import pandas as pd
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from config import Config

logger = logging.getLogger(__name__)

class BinanceDataClient:
    """
    Client for fetching REAL data (klines and price) from Binance Futures.
    Uses API keys if available for higher rate limits, but works in public mode otherwise.
    """
    def __init__(self):
        self.symbol = Config.SYMBOL
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.is_testnet = Config.BINANCE_TESTNET
        self.base_url = Config.BINANCE_FUTURES_API_URL
        
        self.client = Client(
            self.api_key, 
            self.api_secret, 
            base_url=self.base_url,
            tld='com'
        )
        self.futures_client = self.client.futures
        
        # Determine precision for output messages
        # Default to 2, will attempt to fetch real precision later
        self.price_precision = 2 
        
        logger.info(f"Binance Data Client initialized. Testnet: {self.is_testnet}")
        if not self.api_key:
             logger.warning("Binance API Key is NOT set. Using public endpoints (lower rate limit).")

        # Set precision
        self._get_symbol_precision()

    def _get_symbol_precision(self):
        """Fetches the price precision from exchange info."""
        try:
            info = self.futures_client.get_exchange_info()
            symbol_info = next(
                (s for s in info['symbols'] if s['symbol'] == self.symbol), 
                None
            )
            if symbol_info:
                # Price filter defines precision
                price_filter = next(
                    (f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), 
                    None
                )
                if price_filter:
                    # stepSize like '0.01' or '0.0001'
                    step_size = price_filter['tickSize']
                    self.price_precision = len(step_size.split('.')[-1].rstrip('0'))
            
        except Exception as e:
            logger.error(f"Could not fetch symbol precision. Defaulting to {self.price_precision}. Error: {e}")
    
    def _round_price(self, price: float) -> float:
        """Utility for rounding price based on fetched precision."""
        if price is None:
            return 0.0
        return round(price, self.price_precision)

    def get_historical_klines(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetches REAL historical klines from Binance Futures."""
        try:
            # interval maps directly to Binance's format (e.g., '15m')
            klines = self.futures_client.klines(
                symbol=symbol, 
                interval=timeframe, 
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Clean and format data
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Set close time as index, as it represents the completed candle
            df.set_index('close_time', inplace=True)
            
            # Convert OHLCV columns to numeric
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
            ticker = self.futures_client.get_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            return price
        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None
