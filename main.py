# main.py
import time
import logging
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List 
from settings import Config
from utils.indicators import Indicators 
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


# --- BinanceDataClient Class (Code remains the same as previously validated) ---
class BinanceDataClient:
    """Client for fetching real-time and historical data from Binance USD-M Futures."""

    def __init__(self):
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.is_testnet = Config.BINANCE_TESTNET

        base_url = "https://testnet.binancefuture.com" if self.is_testnet else "https://fapi.binance.com"
        
        if self.api_key and self.api_secret:
            self.futures_client = UMFutures(
                key=self.api_key,
                secret=self.api_secret,
                base_url=base_url
            )
        else:
            self.futures_client = UMFutures(base_url=base_url)

        self.price_precisions: Dict[str, int] = {}
        self._get_symbol_precisions() 
        logger.info(f"✅ Binance Data Client initialized. Testnet: {self.is_testnet}")

    def _get_symbol_precisions(self):
        """Fetch price precision for all configured symbols from exchange info."""
        valid_symbols = [s for s in Config.SYMBOLS if s.endswith(Config.QUOTE_ASSET) and s != Config.QUOTE_ASSET]
        if not valid_symbols:
            logger.error(f"❌ No valid symbols found! Check your SYMBOLS and QUOTE_ASSET config.")
            return

        try:
            info = self.futures_client.exchange_info()
            
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

# --- NEW: Function to fetch HTF data ---
def get_htf_data(client, symbol: str) -> pd.DataFrame:
    """Fetches the 1-hour historical data for HTF trend validation."""
    return client.get_historical_klines(symbol, interval="1h", limit=100)


# --- Updated format_signal_message to include HTF trend ---
def format_signal_message(symbol: str, signal_type: str, signal_data: dict, current_price: float, htf_trend: str) -> str:
    """Format trading signal for Telegram notification, incorporating strength and HTF trend."""
    
    signal_strength = signal_data.get('signal_strength', 'SOFT') 
    header, action = Indicators.get_alert_message_header(signal_type, signal_strength, symbol)
    
    if signal_type == "NO_TRADE":
        reason = signal_data.get('reason', 'Conditions not met.')
        # HTF trend is often available even on NO_TRADE
        return (
            f"⚪ No Trade Signal on *{symbol}* | Price: ${current_price:.4f}\n"
            f"*1H Trend*: {htf_trend}\n"
            f"*Reason*: {reason}"
        )
    
    entry = signal_data.get('entry_price', current_price)
    sl = signal_data.get('stop_loss', 0)
    tp = signal_data.get('take_profit', 0)
    risk_factor = signal_data.get('risk_factor', 1.0)
    tdi_slow_ma = signal_data.get('tdi_slow_ma', 0)
    bb_width = signal_data.get('bb_width_percent', 0)
    
    return (
        f"{header}\n"
        f"---"
        f"\n*1H Trend*: **{htf_trend}**" # Highlight the Higher Timeframe Trend
        f"\n*{signal_strength} {action} SIGNAL*\n"
        f"💰 **Entry**: ${entry:.4f} (Current: ${current_price:.4f})\n" 
        f"🛑 **SL**: ${sl:.4f}\n"
        f"🎯 **TP**: ${tp:.4f}\n"
        f"---"
        f"\n*Risk*: {risk_factor:.1f}x | *TDI Slow MA*: {tdi_slow_ma:.2f}\n"
        f"*Volatility*: {bb_width * 100:.3f}%"
    )
# --- END Updated FUNCTION ---


def main():
    """Main entry point for Binance Data Client with Trading Strategy."""
    logger.info("🚀 Starting Binance Data Client with Consolidated Trend Strategy...")

    try:
        MIN_KLINES_REQUIRED = max(Config.BB_PERIOD, Config.TDI_RSI_PERIOD) + 2
        
        client = BinanceDataClient()
        strategy = ConsolidatedTrendStrategy()
        
        safe_send_telegram_message(f"✅ Client & Strategy started. Monitoring: {', '.join(client.price_precisions.keys())} | Min Klines: {MIN_KLINES_REQUIRED}")

        symbol_state: Dict[str, Dict[str, Any]] = {
            symbol: {"last_signal": None, "cooldown": 0} for symbol in client.price_precisions.keys()
        }
        
        symbols_to_monitor = list(client.price_precisions.keys())

        while True:
            for symbol in symbols_to_monitor:
                try:
                    # 1. Fetch 5m and 1H data
                    current_price = client.get_current_price(symbol)
                    historical_data_5m = client.get_historical_klines(symbol, limit=100)
                    historical_data_1h = get_htf_data(client, symbol) 
                    state = symbol_state[symbol]
                    
                    if not current_price or historical_data_5m.empty or historical_data_1h.empty:
                        logger.warning(f"Skipping {symbol}: Insufficient data.")
                        continue

                    # 2. Set HTF Trend and Analyze 5m data
                    strategy.set_htf_trend(historical_data_1h) # Set 1H trend bias
                    htf_trend = strategy.get_strategy_stats()['htf_trend'] # Get the calculated trend for messaging
                    
                    analyzed_data_5m = strategy.analyze_data(historical_data_5m)
                    
                    if not analyzed_data_5m.empty and 'bb_middle' in analyzed_data_5m.columns: 
                        # 3. Generate trading signal (uses HTF filter and debounce internally)
                        signal_type, signal_data = strategy.generate_signal(analyzed_data_5m)
                        
                        # 4. Check for new signal and cooldown status
                        is_new_signal = signal_type != "NO_TRADE"
                        is_not_spamming = (signal_type != state["last_signal"] or state["cooldown"] <= 0) 
                        
                        if is_new_signal and is_not_spamming:
                            message = format_signal_message(symbol, signal_type, signal_data, current_price, htf_trend)
                            logger.info(f"🎯 Strategy Signal for {symbol}: {signal_type} | HTF: {htf_trend}")
                            safe_send_telegram_message(message)
                            
                            state["last_signal"] = signal_type
                            state["cooldown"] = 5  
                        
                        # Decrement cooldown if active
                        if state["cooldown"] > 0:
                            state["cooldown"] -= 1
                            
                        if signal_type == "NO_TRADE":
                             logger.info(f"⚪ No trade signal for {symbol} | HTF: {htf_trend} | Price: {current_price:.4f}")

                    else:
                        if 'bb_middle' not in analyzed_data_5m.columns:
                             logger.critical(f"❌ CRASH PREVENTED: 'bb_middle' column missing for {symbol}!")
                        else:
                            logger.warning(f"⚠️ Analyzed data for {symbol} is empty after processing")

                except Exception as e:
                    error_message = f"🔥 CRITICAL ERROR processing {symbol}: {type(e).__name__}: {e}"
                    logger.critical(error_message)
                    safe_send_telegram_message(escape_markdown(error_message))
                    continue
            
            logger.info(f"😴 Polling cycle complete. Sleeping for {Config.POLLING_INTERVAL_SECONDS}s.")
            time.sleep(Config.POLLING_INTERVAL_SECONDS)

    except Exception as e:
        raw_error = f"🔥 Global Critical Error: {type(e).__name__}: {e}"
        logger.critical(raw_error)
        safe_send_telegram_message(escape_markdown(raw_error))


if __name__ == "__main__":
    # NOTE: Ensure you have 'utils/telegram_bot.py' present with a functional 'send_telegram_message_sync'
    main()
