# main.py
import time
import logging
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List 
from settings import Config # Assumed to exist and contain necessary settings
# Note: Assuming utils.telegram_bot and strategy.consolidated_trend are in your environment
from utils.telegram_bot import send_telegram_message_sync as send_telegram_message
from binance.um_futures import UMFutures
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Import your strategy
from strategy.consolidated_trend import ConsolidatedTrendStrategy # Assumed to exist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# --- BinanceDataClient Class (No changes needed here, copied for completeness) ---
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
    
    # Use data from signal_data or current_price as fallback
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


# --- NEW: Signal Conclusion Logic ---
def check_active_signal_status(symbol: str, current_price: float, state: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """
    Checks if the active signal has hit its Take Profit (TP) or Stop Loss (SL).
    Returns: (new_status, result_price)
    """
    signal_type = state["active_signal_type"]
    sl = state["active_stop_loss"]
    tp = state["active_take_profit"]

    # Check for SHORT signal conclusion (price rise hits SL or price fall hits TP)
    if signal_type == "SELL":
        if current_price >= sl:
            logger.info(f"🛑 SELL SIGNAL CONCLUDED (SL) for {symbol} at ${current_price:.4f}")
            return "LOSS", sl
        elif current_price <= tp:
            logger.info(f"🎯 SELL SIGNAL CONCLUDED (TP) for {symbol} at ${current_price:.4f}")
            return "PROFIT", tp

    # Check for BUY signal conclusion (price fall hits SL or price rise hits TP)
    elif signal_type == "BUY":
        if current_price <= sl:
            logger.info(f"🛑 BUY SIGNAL CONCLUDED (SL) for {symbol} at ${current_price:.4f}")
            return "LOSS", sl
        elif current_price >= tp:
            logger.info(f"🎯 BUY SIGNAL CONCLUDED (TP) for {symbol} at ${current_price:.4f}")
            return "PROFIT", tp
    
    # If active, but neither SL nor TP is hit
    return "ACTIVE", None


def format_conclusion_message(symbol: str, status: str, result_price: float, signal_type: str) -> str:
    """Format the signal conclusion for Telegram notification."""
    emoji = "✅" if status == "PROFIT" else "❌"
    action = "LONG" if signal_type == "BUY" else "SHORT"
    
    return (
        f"{emoji} *{action} SIGNAL CONCLUDED* for *{symbol}* {emoji}\n"
        f"Result: *{status}*\n" 
        f"Price: ${result_price:.4f}"
    )
# ------------------------------------


def main():
    """Main entry point for Binance Data Client with Trading Strategy."""
    logger.info("🚀 Starting Binance Data Client with Consolidated Trend Strategy...")

    try:
        # Determine the minimum required data points for the strategy
        MIN_KLINES_REQUIRED = max(Config.BB_PERIOD, Config.TDI_RSI_PERIOD) + 2
        
        # Initialize clients
        client = BinanceDataClient()
        strategy = ConsolidatedTrendStrategy()
        
        symbols_monitored = list(client.price_precisions.keys())
        safe_send_telegram_message(f"✅ Client & Strategy started. Monitoring: {', '.join(symbols_monitored)} | Min Klines: {MIN_KLINES_REQUIRED}")

        # --- 🎯 ENHANCED STATE TRACKING ---
        symbol_state: Dict[str, Dict[str, Any]] = {
            symbol: {
                "signal_status": "NONE",          # NONE, ACTIVE, PROFIT, LOSS
                "active_signal_type": None,       # BUY or SELL
                "active_entry_price": 0.0,
                "active_stop_loss": 0.0,
                "active_take_profit": 0.0,
                "last_signal_generated": "NONE",  # Last signal generated (can be NO_TRADE)
            } for symbol in symbols_monitored
        }
        # -----------------------------------

        while True:
            for symbol in symbols_monitored:
                try:
                    current_price = client.get_current_price(symbol)
                    historical_data = client.get_historical_klines(symbol, limit=100)
                    state = symbol_state[symbol]
                    
                    # Basic checks
                    if not current_price or historical_data.empty or len(historical_data) < MIN_KLINES_REQUIRED:
                        if len(historical_data) < MIN_KLINES_REQUIRED:
                            logger.warning(f"⚠️ Insufficient klines ({len(historical_data)}/{MIN_KLINES_REQUIRED}) for {symbol}. Skipping cycle.")
                        continue
                    
                    # 1. 🔍 Check Status of Active Signal (if one exists)
                    if state["signal_status"] == "ACTIVE":
                        new_status, result_price = check_active_signal_status(symbol, current_price, state)
                        
                        if new_status in ["PROFIT", "LOSS"]:
                            # Signal concluded, send alert and reset state
                            conclusion_message = format_conclusion_message(
                                symbol, 
                                new_status, 
                                result_price, 
                                state["active_signal_type"]
                            )
                            safe_send_telegram_message(conclusion_message)
                            
                            # RESET the state to allow new signal generation
                            state["signal_status"] = "NONE"
                            state["active_signal_type"] = None
                            logger.info(f"🔄 State reset for {symbol}. Ready for new signal.")
                            
                        else:
                            # Still ACTIVE, log and skip new signal generation
                            logger.info(f"✅ Active {state['active_signal_type']} signal for {symbol}. Price: ${current_price:.4f} (SL: ${state['active_stop_loss']:.4f})")
                            continue # Crucial: Skip to the next symbol

                    
                    # 2. 💡 Generate New Signal (Only if signal_status is NONE)
                    if state["signal_status"] == "NONE":
                        
                        analyzed_data = strategy.analyze_data(historical_data)
                        
                        if 'bb_middle' not in analyzed_data.columns:
                             logger.critical(f"❌ CRASH PREVENTED: 'bb_middle' column missing for {symbol}! Check consolidated_trend.py.")
                             continue
                             
                        signal_type, signal_data = strategy.generate_signal(analyzed_data)
                        
                        # --- Anti-Spam Check: Ensure it's a new trade signal ---
                        is_new_trade_signal = signal_type in ["BUY", "SELL"]
                        
                        if is_new_trade_signal and signal_type != state["last_signal_generated"]:
                            
                            # 3. Send Alert and Update Active State
                            message = format_signal_message(symbol, signal_type, signal_data, current_price)
                            logger.info(f"🎯 Strategy Signal for {symbol}: {signal_type}")
                            safe_send_telegram_message(message)
                            
                            # Update active signal state
                            state["signal_status"] = "ACTIVE"
                            state["active_signal_type"] = signal_type
                            state["active_entry_price"] = signal_data.get('entry_price', current_price)
                            # Ensure SL/TP are available from the signal data (crucial for conclusion)
                            state["active_stop_loss"] = signal_data.get('stop_loss', 0.0) 
                            state["active_take_profit"] = signal_data.get('take_profit', 0.0)

                        # Always update the last *generated* signal for history tracking
                        state["last_signal_generated"] = signal_type
                        
                        if signal_type == "NO_TRADE":
                            logger.info(f"⚪ No trade signal for {symbol} | Price: {current_price:.4f}")

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
