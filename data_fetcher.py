# data_fetcher.py
import pandas as pd
import logging
from typing import Optional, List, Any
# 💡 REQUIRED: Import the actual Binance Client
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Local imports
from settings import Config 

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Handles connection to the exchange (Binance) and fetches/cleans market data.
    """
    
    def __init__(self):
        """Initialize the real Binance client."""
        # 💡 Connect to the actual Binance API
        self.client = BinanceClient( 
            api_key=Config.BINANCE_API_KEY, 
            api_secret=Config.BINANCE_API_SECRET,
            testnet=Config.BINANCE_TESTNET # Use testnet if configured in settings
        )
        logger.info(f"Binance Client initialized. Testnet mode: {Config.BINANCE_TESTNET}.")

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Fetches historical kline data using the real client and converts it to a clean DataFrame.
        """
        try:
            # Call the actual Binance API to get klines
            raw_klines = self.client.get_historical_klines(
                symbol=symbol, 
                interval=interval, 
                limit=limit
            )
            
            if not raw_klines:
                logger.warning(f"No klines returned for {symbol}.")
                return None

            df = pd.DataFrame(raw_klines)
            
            # Binance klines format has 12 columns
            df.columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            # Keep only the relevant OHLCV data
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()

            # Convert OHLCV columns to numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Convert timestamp to datetime and set as index
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            return df
            
        except (BinanceAPIException, BinanceRequestException) as e:
            # Handle specific Binance errors (e.g., API key issue, invalid symbol)
            logger.error(f"Binance API Error fetching klines for {symbol}: {str(e)}")
            return None
        except Exception as e:
            # This is where your Pandas compatibility issue might still surface
            logger.error(f"Error processing klines for {symbol}: {str(e)}")
            return None
