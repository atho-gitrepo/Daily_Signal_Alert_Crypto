# utils/coingecko_data_client.py

import logging
import pandas as pd
import httpx
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# CoinGecko's free API is limited. We'll use the price endpoint for current price
# and provide mock data for historical klines since OHLCV is not directly available 
# on the simple price endpoint.
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"
# Parameters for BTC/USDT and ETH/USDT prices and 24h volume
COINGECKO_PARAMS = {
    'ids': 'bitcoin',  # CoinGecko only provides BTC/ETH. We'll use BTC as a stand-in.
    'vs_currencies': 'usd',
    'include_24hr_vol': 'true'
}

class CoinGeckoDataClient:
    """
    A client for fetching data from the CoinGecko API.
    It replaces the Binance client, using mock historical data where needed.
    """
    def __init__(self, symbol="BTCUSDT", price_precision=2):
        self.symbol = symbol # e.g., 'BTCUSDT'
        self.price_precision = price_precision # Used for formatting in main.py
        
        # NOTE: We assume 'bitcoin' is the proxy for your main symbol (BTCUSDT)
        # due to CoinGecko limitations.
        self.coingecko_id = 'bitcoin'
        
        logger.info(f"CoinGecko Data Client initialized for {self.symbol}.")
        
    def _round_price(self, price: float) -> float:
        """Simple rounding utility to mimic precision."""
        if price is None:
            return 0.0
        return round(price, self.price_precision)

    def get_current_price(self) -> float | None:
        """
        Fetches the current ticker price for the configured symbol synchronously.
        """
        try:
            # Synchronous request using httpx for simplicity in this utility file
            with httpx.Client(timeout=10) as client:
                response = client.get(COINGECKO_API_URL, params=COINGECKO_PARAMS)
                response.raise_for_status()
                
                data = response.json()
                price = data.get(self.coingecko_id, {}).get('usd')
                
                if price is not None:
                    return float(price)
                
                logger.warning(f"Price data not found for {self.coingecko_id} in CoinGecko response.")
                return None
                
        except httpx.RequestError as e:
            logger.error(f"Network error fetching price from CoinGecko: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching price from CoinGecko: {e}")
            return None

    def get_historical_klines(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetches historical data. WARNING: This returns MOCK/PLACEHOLDER data 
        because CoinGecko's simple API does not offer OHLCV klines. 
        Your strategy indicators will likely need real data to work correctly.
        """
        logger.warning("Using MOCK data for historical klines because CoinGecko's simple API lacks OHLCV data. Strategy indicators will be unreliable.")
        
        # Get the latest real price for a basis
        latest_close = self.get_current_price()
        if latest_close is None:
            latest_close = 4500.0 # Fallback close if API fails
            
        current_time = datetime.now()
        
        # Generate 'limit' mock candles ending now
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
