import ccxt
import logging
import pandas as pd
import math
from config import Config

# Configure logging to show timestamps and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataClient:
    """
    A client for fetching data from the Binance Futures API using the ccxt library.
    It supports both live and testnet environments based on the configuration.
    """
    def __init__(self):
        # 1. Initialize CCXT client
        self.client = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_API_SECRET,
            'options': {
                'defaultType': 'future',
                'recvWindow': 60000,
            },
            # --- START OF FIX: ADDING USER-AGENT HEADER ---
            'headers': {
                # This User-Agent header helps requests bypass blocks 
                # often imposed by CDNs (like CloudFront) on cloud/data center IPs.
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
            # --- END OF FIX ---
            'urls': {
                'api': {
                    'fapiPublic': Config.BINANCE_FUTURES_API_URL,
                    'fapiPrivate': Config.BINANCE_FUTURES_API_URL,
                },
            },
        })
        
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME
        
        try:
            # This call caused the 403 Forbidden error, which should now be resolved
            self.client.load_markets() 
            self.market_info = self.client.market(self.symbol)
            if not self.market_info:
                raise RuntimeError(f"Could not retrieve market info for {self.symbol}")
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT exchange error during market loading: {e}")
            raise RuntimeError(f"Failed to load markets for {self.symbol}") from e

        logger.info(f"Binance Data client initialized for {self.symbol} using ccxt.")

        # 2. Extract precision from market info
        self.price_precision = self.market_info['precision']['price']
        self.amount_precision = self.market_info['precision']['amount']
        logger.info(f"Price precision for {self.symbol}: {self.price_precision}")
        logger.info(f"Amount precision for {self.symbol}: {self.amount_precision}")

    def _round_price(self, price: float) -> float:
        """Rounds price to the correct precision using CCXT's built-in method."""
        return self.client.price_to_precision(self.symbol, price)

    def _round_amount(self, amount: float) -> float:
        """Rounds amount to the correct precision using CCXT's built-in method."""
        return self.client.amount_to_precision(self.symbol, amount)

    def get_historical_klines(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetches historical candlestick data for the configured symbol and timeframe.
        """
        try:
            # fetch_ohlcv returns a list of lists: [timestamp, open, high, low, close, volume]
            ohlcv = self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert columns to numeric types for later calculations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except ccxt.NetworkError as e:
            logger.error(f"CCXT network error fetching klines for {self.symbol}: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT exchange error fetching klines for {self.symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {self.symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float | None:
        """
        Fetches the current ticker price for the configured symbol.
        """
        try:
            ticker = self.client.fetch_ticker(self.symbol)
            return float(ticker['last']) if ticker and 'last' in ticker else None
        except ccxt.NetworkError as e:
            logger.error(f"CCXT network error fetching current price for {self.symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT exchange error fetching current price for {self.symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching current price for {self.symbol}: {e}")
            return None
