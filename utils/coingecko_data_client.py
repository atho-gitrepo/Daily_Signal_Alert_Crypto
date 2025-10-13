import logging
import pandas as pd
import httpx
import time
from datetime import datetime
from httpx import HTTPStatusError # Import specific error class

logger = logging.getLogger(__name__)

COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_PARAMS = {
    'ids': 'bitcoin', 
    'vs_currencies': 'usd',
    'include_24hr_vol': 'true'
}

class CoinGeckoDataClient:
    """
    A client for fetching data from the CoinGecko API.
    Uses caching and fetches price only once per loop to prevent 429 rate limit errors.
    """
    def __init__(self, symbol="BTCUSDT", price_precision=2):
        self.symbol = symbol 
        self.price_precision = price_precision 
        self.coingecko_id = 'bitcoin'
        self._cached_current_price = None 
        self._last_price_fetch_time = 0   
        logger.info(f"CoinGecko Data Client initialized for {self.symbol}.")
        
    def _round_price(self, price: float) -> float:
        """Simple rounding utility to mimic precision."""
        if price is None:
            return 0.0
        return round(price, self.price_precision)

    def _fetch_real_current_price(self) -> float | None:
        """Internal method to fetch price from CoinGecko, update cache, with error handling."""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(COINGECKO_API_URL, params=COINGECKO_PARAMS)
                response.raise_for_status()
                
                data = response.json()
                price = data.get(self.coingecko_id, {}).get('usd')
                
                if price is not None:
                    self._cached_current_price = float(price)
                    self._last_price_fetch_time = time.time()
                    return self._cached_current_price
                
                logger.warning(f"Price data not found for {self.coingecko_id} in CoinGecko response.")
                return None
                
        except HTTPStatusError as e:
            # Catch and log 429 explicitly
            logger.error(f"CoinGecko API error {e.response.status_code} fetching price: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Network error fetching price from CoinGecko: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching price from CoinGecko: {e}")
            return None

    def get_current_price(self) -> float | None:
        """
        Returns the cached current ticker price, fetched during the historical klines call.
        """
        return self._cached_current_price

    def get_historical_klines(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetches current price to use as basis, then generates MOCK historical data.
        This is the only method that calls the CoinGecko API (via _fetch_real_current_price).
        """
        logger.warning("Using MOCK data for historical klines because CoinGecko's simple API lacks OHLCV data. Strategy indicators will be unreliable.")
        
        # 1. Update the cached price once at the start of the loop
        latest_close = self._fetch_real_current_price()
        
        if latest_close is None:
            # Fallback if API fails (will use the last known price or a default)
            latest_close = self._cached_current_price if self._cached_current_price is not None else 45000.0
            logger.warning(f"CoinGecko fetch failed. Using last known/default price: {latest_close}")

        current_time = datetime.now()
        
        # 2. Generate 'limit' mock candles based on the fetched price
        data = []
        for i in range(limit, 0, -1):
            # Create a plausible but mock open, high, low based on the latest close
            mock_close = latest_close * (1 + (i / 10000.0) * (1 - 2 * (i % 2)))
            mock_open = mock_close * (1 - 0.0001 * (i % 5))
            mock_high = max(mock_open, mock_close) * 1.0005
            mock_low = min(mock_open, mock_close) * 0.9995
            mock_volume = 10000 + (i * 100)
            
            # Simple timestamp approximation
            timestamp_ms = int((current_time - pd.to_timedelta(i, unit='m')).timestamp() * 1000)
            
            data.append([timestamp_ms, mock_open, mock_high, mock_low, mock_close, mock_volume])

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.iloc[:-1] # Exclude the very last candle which is incomplete
