# main.py
import time
import logging
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List # Added List import
from settings import Config
# Note: Assuming utils.telegram_bot and strategy.consolidated_trend are in your environment
from utils.telegram_bot import send_telegram_message_sync as send_telegram_message
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Import your strategy
from strategy.consolidated_trend import ConsolidatedTrendStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# --- BinanceDataClient Class ---
class BinanceDataClient:
    """Client for fetching real-time and historical data from Binance USD-M Futures."""

    def __init__(self):
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.is_testnet = Config.BINANCE_TESTNET

        if not self.api_key or not self.api_secret:
            logger.warning("⚠️ Binance API Key/Secret not set. Using public endpoints (rate-limited).")

        base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
        
        if self.api_key and self.api_secret:
            self.futures_client = UMFutures(
                key=self.api_key,
                secret=self.api_secret,
                base_url=base_url
            )
        else:
            self.futures_client = UMFutures(base_url=base_url)

        # Store precision data for all configured symbols
        self.price_precisions: Dict[str, int] = {}
        self._get_symbol_precisions() # Fetch precision for all symbols
        logger.info(f"✅ Binance Data Client initialized. Testnet: {self.is_testnet}")


    def _get_symbol_precisions(self):
        """Fetch price precision for all configured symbols from exchange info."""
        # Symbol cleaning check (from previous steps)
        valid_symbols = [s for s in Config.SYMBOLS if s.endswith(Config.QUOTE_ASSET) and s != Config.QUOTE_ASSET]
        if not valid_symbols:
            logger.error(f"❌ No valid symbols found! Check your SYMBOLS and QUOTE_ASSET config.")
            return

        try:
            info = self.futures_client.exchange_info()
            
            # Use the list of symbols from Config
            for symbol in valid_symbols:
                symbol_info = next(
                    (s for s in info['symbols'] if s['symbol'] == symbol),
                    None
                )
                if symbol_info:
                    price_filter = next(
                        (f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'),
                        None
                    )
                    if price_filter:
                        step_size = price_filter['tickSize']
                        precision = len(step_size.split('.')[-1].rstrip('0'))
                        self.price_precisions[symbol] = precision
            
            logger.info(f"ℹ️ Fetched price precisions for {list(self.price_precisions.keys())}")
        except Exception as e:
            logger.error(f"❌ Could not fetch symbol precisions. Error: {e}")

    
    def _round_price(self, symbol: str, price: float) -> float:
        """Rounds price based on symbol-specific precision."""
        precision = self.price_precisions.get(symbol, 2)
        return round(price, precision) if price else 0.0

    
    def get_historical_klines(self, symbol: str, interval: str = None, limit: int = 500) -> pd.DataFrame:
        """Fetch historical klines (OHLCV) for a specific symbol."""
        interval = interval or Config.TIMEFRAME
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
            logger.info(f"📊 Fetched {len(df)} klines for {symbol}.")
            return df
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"❌ Binance Error fetching {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching klines for {symbol}: {e}")
            return pd.DataFrame()


    def get_current_price(self, symbol: str) -> float | None:
        """Fetch current market price for a specific symbol."""
        try:
            ticker = self.futures_client.ticker_price(symbol=symbol)
            return self._round_price(symbol, float(ticker['price']))
        except Exception as e:
            logger.error(f"❌ Failed to fetch current price for {symbol}: {e}")
            return None


def escape_markdown(text: str) -> str:
    """Escape Telegram MarkdownV2-sensitive characters."""
    # List of characters that need escaping in MarkdownV2
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, f'\{char}')
    return text


def safe_send_telegram_message(message: str):
    """Safely send Telegram message with error handling."""
    try:
        escaped_message = escape_markdown(message)
        send_telegram_message(escaped_message)
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram message: {e}")


def format_signal_message(symbol: str, signal_type: str, signal_data: dict, current_price: float) -> str:
    """Format trading signal for Telegram notification, including the symbol."""
    if signal_type == "NO_TRADE":
        return f"⚪ No Trade Signal on *{symbol}* | Price: {current_price}"
    
    entry = signal_data.get('entry_price', current_price)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    risk_factor = signal_data.get('risk_factor', 1.0)
    tdi_value = signal_data.get('tdi_value', 0)
    
    if signal_type == "BUY":
        emoji = "🟢"
        action = "LONG"
    else:  # SELL
        emoji = "🔴" 
        action = "SHORT"
    
    return (
        f"{emoji} *{action} SIGNAL* for *{symbol}* {emoji}\n"
        f"💰 Entry: ${entry:.4f}\n" 
        f"🛑 Stop Loss: ${sl:.4f}\n"
        f"🎯 Take Profit: ${tp:.4f}\n"
        f"📊 Current: ${current_price:.4f}\n"
        f"⚖️ Risk Factor: {risk_factor}x\n"
        f"📈 TDI RSI: {tdi_value:.2f}"
    )


def main():
    """Main entry point for Binance Data Client with Trading Strategy."""
    logger.info("🚀 Starting Binance Data Client with Consolidated Trend Strategy...")

    try:
        # Determine the minimum required data points for the strategy
        MIN_KLINES_REQUIRED = max(Config.BB_PERIOD, Config.TDI_RSI_PERIOD) + 2
        
        # Initialize clients
        client = BinanceDataClient()
        strategy = ConsolidatedTrendStrategy()
        
        safe_send_telegram_message(f"✅ Client & Strategy started. Monitoring: {', '.join(client.price_precisions.keys())} | Min Klines: {MIN_KLINES_REQUIRED}")

        # Track last signal and cooldown PER SYMBOL to avoid spam
        symbol_state: Dict[str, Dict[str, Any]] = {
            symbol: {"last_signal": None, "cooldown": 0} for symbol in client.price_precisions.keys()
        }
        
        symbols_to_monitor = list(client.price_precisions.keys())

        while True:
            for symbol in symbols_to_monitor:
                try:
                    # 1. Get current price and historical data for the specific symbol
                    current_price = client.get_current_price(symbol)
                    historical_data = client.get_historical_klines(symbol, limit=100)
                    state = symbol_state[symbol]
                    
                    if not current_price:
                        logger.warning(f"⚠️ Missing current price for {symbol}. Skipping cycle.")
                        continue
                    
                    if historical_data.empty:
                        logger.warning(f"⚠️ Historical data for {symbol} is empty. Skipping cycle.")
                        continue

                    # Enforce Minimum Klines Check
                    if len(historical_data) < MIN_KLINES_REQUIRED:
                        logger.warning(f"⚠️ Insufficient klines ({len(historical_data)}/{MIN_KLINES_REQUIRED}) for {symbol}. Skipping calculation.")
                        continue

                    # 2. Analyze data
                    analyzed_data = strategy.analyze_data(historical_data)
                    
                    # 🎯 FIX: Changed 'bb_mid' to 'bb_middle' for alignment with indicators.py
                    if not analyzed_data.empty and 'bb_middle' in analyzed_data.columns: 
                        # 3. Generate trading signal
                        signal_type, signal_data = strategy.generate_signal(analyzed_data)
                        
                        # 4. Check for new signal and cooldown status for THIS symbol
                        is_new_signal = signal_type != "NO_TRADE"
                        is_not_spamming = (signal_type != state["last_signal"] or state["cooldown"] <= 0) 
                        
                        if is_new_signal and is_not_spamming:
                            message = format_signal_message(symbol, signal_type, signal_data, current_price)
                            logger.info(f"🎯 Strategy Signal for {symbol}: {signal_type}")
                            safe_send_telegram_message(message)
                            
                            # Update state
                            state["last_signal"] = signal_type
                            state["cooldown"] = 5  # Set cooldown (5 iterations)
                        
                        # Decrement cooldown if active
                        if state["cooldown"] > 0:
                            state["cooldown"] -= 1
                            
                        # Log current status (less verbose)
                        if signal_type == "NO_TRADE":
                            logger.info(f"⚪ No trade signal for {symbol} | Price: {current_price:.4f}")
                    else:
                        # 🎯 FIX: Changed 'bb_mid' to 'bb_middle' in the error log
                        if 'bb_middle' not in analyzed_data.columns:
                             logger.critical(f"❌ CRASH PREVENTED: 'bb_middle' column missing for {symbol}! Strategy implementation error. Check consolidated_trend.py.")
                        else:
                            logger.warning(f"⚠️ Analyzed data for {symbol} is empty after processing")

                except Exception as e:
                    # Localized symbol error handler
                    error_message = f"🔥 CRITICAL ERROR processing {symbol}: {type(e).__name__}: {e}"
                    logger.critical(error_message)
                    safe_send_telegram_message(escape_markdown(error_message))
                    continue
            
            # Sleep after checking ALL symbols
            logger.info(f"😴 Polling cycle complete. Sleeping for {Config.POLLING_INTERVAL_SECONDS}s.")
            time.sleep(Config.POLLING_INTERVAL_SECONDS)

    except Exception as e:
        # Global initialization error handler
        raw_error = f"🔥 Global Critical Error: {type(e).__name__}: {e}"
        logger.critical(raw_error)
        safe_send_telegram_message(escape_markdown(raw_error))


if __name__ == "__main__":
    main()