import ccxt
from config import Config
import logging
import pandas as pd
import math

logger = logging.getLogger(__name__)

class BinanceDataClient:
    def __init__(self):
        # Initialize CCXT client
        self.client = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_API_SECRET,
            # For futures, specify the options.defaultType
            'options': {'defaultType': 'future'},
        })
        self.symbol = Config.SYMBOL
        self.market_info = self.client.market(self.symbol)
        
        if self.market_info:
            logger.info(f"Binance Data client initialized for {self.symbol} using ccxt.")
        else:
            logger.error(f"Failed to get market info for {self.symbol}. Data client might not function correctly.")
            raise RuntimeError(f"Could not retrieve market info for {self.symbol}")
        
        # CCXT handles precision, you can access it via the market info
        self.price_precision = self.market_info['precision']['price']
        logger.info(f"Price precision for {self.symbol}: {self.price_precision}")

    def _round_price(self, price):
        """Rounds price to the correct precision using CCXT's built-in method."""
        return self.client.price_to_precision(self.symbol, price)

    def get_historical_klines(self, symbol, interval, limit=500):
        """Fetches historical candlestick data for Futures using CCXT."""
        try:
            # CCXT's fetch_ohlcv returns a list of lists
            ohlcv = self.client.fetch_ohlcv(symbol, interval, limit=limit)
            
            # The structure is [timestamp, open, high, low, volume]
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except ccxt.NetworkError as e:
            logger.error(f"CCXT network error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT exchange error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol):
        """Fetches the current ticker price for Futures using CCXT."""
        try:
            ticker = self.client.fetch_ticker(symbol)
            return ticker['last']
        except ccxt.NetworkError as e:
            logger.error(f"CCXT network error fetching current price for {symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT exchange error fetching current price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def get_exchange_info(self, symbol):
        """Fetches exchange information for a specific symbol using CCXT."""
        try:
            # CCXT's load_markets populates the markets dictionary
            self.client.load_markets()
            if symbol in self.client.markets:
                return self.client.markets[symbol]
            return None
        except Exception as e:
            logger.error(f"Error getting exchange info with ccxt: {e}")
            return None

