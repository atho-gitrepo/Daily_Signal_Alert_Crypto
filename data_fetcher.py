# data_fetcher.py
import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from settings import Config # Assuming Config holds API keys/secrets

logger = logging.getLogger(__name__)

# Mock Client: Replace this with your actual exchange client initialization
class MockBinanceClient:
    """
    A stand-in for a real exchange client (like python-binance's Client).
    In a real setup, this would connect to the exchange API.
    """
    def __init__(self, api_key: str, api_secret: str):
        # In a real scenario, you'd initialize the exchange client here
        self.api_key = api_key
        self.api_secret = api_secret
        logger.info("Mock Client initialized. Using dummy data for demonstration.")

    def get_historical_klines(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        """
        Mocks fetching historical klines. Replace this with actual API calls.
        The data structure mimics the standard Binance kline format.
        """
        
        # --- DUMMY DATA ---
        # NOTE: In a real bot, replace this entire function body with your
        # client.get_historical_klines(symbol, interval, limit=limit) call.
        
        # Create a sequential date range
        end_time = pd.Timestamp.now(tz='UTC').floor('1min')
        start_time = end_time - pd.Timedelta(minutes=limit)
        
        timestamps = pd.date_range(start=start_time, end=end_time, periods=limit)
        
        # Simple, non-volatile data for testing indicators (replace with real data)
        base_price = 100.0
        data = []
        for i, ts in enumerate(timestamps):
            price = base_price + (i * 0.01) + (i % 10 - 5) * 0.001
            # [open_time, open, high, low, close, volume, close_time, ...]
            data.append([
                ts.value // 10**6,  # Open Time (ms)
                price - 0.005,      # Open
                price + 0.01,       # High
                price - 0.01,       # Low
                price + 0.005,      # Close
                100.0,              # Volume
                (ts + pd.Timedelta(minutes=1)).value // 10**6 # Close Time (ms)
            ])
            
        return data


class DataFetcher:
    """
    Handles connection to the exchange and fetches/cleans market data.
    """
    
    def __init__(self):
        """Initialize the exchange client using settings from Config."""
        self.client = MockBinanceClient(
            api_key=Config.API_KEY, 
            api_secret=Config.API_SECRET
        )
        # You would replace MockBinanceClient with your actual Client here.

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Fetches historical kline data, converts it to a clean DataFrame, 
        and prepares it for indicator calculation.
        """
        try:
            # 1. Fetch data from the client (Mock or real)
            raw_klines = self.client.get_historical_klines(
                symbol=symbol, 
                interval=interval, 
                limit=limit
            )
            
            if not raw_klines:
                logger.warning(f"No klines returned for {symbol}.")
                return None

            # 2. Convert to DataFrame
            df = pd.DataFrame(raw_klines)
            
            # Standard Binance kline columns (indexes 0 to 4 are OHLCV)
            df.columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            # 3. Data Cleaning and Type Conversion
            # Keep only the essential columns for strategy
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()

            # Convert all price/volume columns to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Convert open_time to datetime index
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {str(e)}")
            return None
