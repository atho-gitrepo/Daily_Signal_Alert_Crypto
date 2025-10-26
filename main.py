import time
import logging
import pandas as pd
import asyncio # Needed for safe_send_telegram_message (even though it's sync)
from typing import Dict, Any, Optional

# Local imports
from settings import Config
from data_fetcher import DataFetcher
from strategy.consolidated_trend import ConsolidatedTrendStrategy
from utils.telegram_bot import send_telegram_message_sync as send_telegram_message # Import the safe sync function
from utils.signal_manager import (
    SignalManager, 
    SignalStatus, 
    TRADE_LIFECYCLE, 
    escape_markdown
)

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize components
data_fetcher = DataFetcher()
strategy = ConsolidatedTrend()
signal_manager = SignalManager()

# --- Utility Functions ---

def format_signal_message(symbol: str, signal_type: str, signal_data: Dict[str, Any]) -> str:
    """Formats the signal and trade details into a Markdown V2 compliant message."""
    
    # Extract data with safe defaults
    entry = signal_data.get('entry_price', 0.0)
    sl = signal_data.get('stop_loss', 0.0)
    tp = signal_data.get('take_profit', 0.0)
    tdi_value = signal_data.get('tdi_value', 0.0)
    signal_strength = signal_data.get('signal_strength', 'N/A')
    
    # Use appropriate emoji and title
    if signal_type == "BUY":
        emoji = "🟢"
        title = f"{emoji} *LONG SIGNAL* for *{symbol}* {emoji}"
    elif signal_type == "SELL":
        emoji = "🔴"
        title = f"{emoji} *SHORT SIGNAL* for *{symbol}* {emoji}"
    else:
        emoji = "⚫"
        title = f"{emoji} *TRADE UPDATE* for *{symbol}* {emoji}"
        
    # Build the message body
    message_parts = [
        title,
        f"💰 Entry: ${entry:.4f}",
        f"🛑 Stop Loss: ${sl:.4f}",
        f"🎯 Take Profit: ${tp:.4f}",
        f"⚖️ Risk Factor: {signal_data.get('risk_factor', 1.0):.1f}x",
        f"📈 TDI Slow MA: {tdi_value:.2f}",
        f"✨ Strength: {signal_strength}",
        f"💬 Note: {signal_data.get('note', 'Consolidated Trend Strategy')}"
    ]
    
    return "\n".join(message_parts)

def format_status_message(symbol: str, status: str, last_candle: pd.Series, price_diff: float) -> str:
    """Formats the trade status update message (PROFIT/LOSS)."""
    
    if status == TRADE_LIFECYCLE.PROFIT:
        emoji = "🎉"
        result_text = "🎯 *TAKE PROFIT HIT*"
    elif status == TRADE_LIFECYCLE.LOSS:
        emoji = "💀"
        result_text = "🛑 *STOP LOSS HIT*"
    else:
        emoji = "ℹ️"
        result_text = "*SIGNAL EXPIRED*"

    message_parts = [
        f"{emoji} *TRADE CONCLUDED* for *{symbol}* {emoji}",
        f"Result: {result_text}",
        f"Current Price: ${last_candle['close']:.4f}",
        f"P/L Pips: {price_diff:.2f}"
    ]
    
    return "\n".join(message_parts)

def safe_send_telegram_message(message: str):
    """
    Safely send Telegram message by escaping Markdown and calling the sync sender.
    This function acts as a wrapper for the now-fixed send_telegram_message_sync.
    """
    try:
        # Escape markdown characters for Telegram's ParseMode.MARKDOWN_V2
        escaped_message = escape_markdown(message)
        
        # Use the fixed synchronous function
        send_telegram_message(escaped_message) 
        
    except Exception as e:
        logger.error(f"❌ Failed to send Telegram message in wrapper: {e}")

# --- Main Logic ---

def process_symbol(symbol: str):
    """Fetches data, runs strategy, and manages the signal lifecycle for one symbol."""
    
    logger.info(f"📊 Fetching {Config.KLINES_LIMIT} klines for {symbol}.")
    df = data_fetcher.fetch_klines(symbol, Config.TIMEFRAME, Config.KLINES_LIMIT)
    
    if df is None or df.empty or len(df) < Config.KLINES_LIMIT:
        logger.warning(f"⚠️ Insufficient data for {symbol}. Skipping.")
        return

    # 1. Run the Strategy on the complete DataFrame
    signal, signal_details = strategy.generate_signal(df)
    last_candle = df.iloc[-1]
    current_price = last_candle['close']
    
    # 2. Check and Manage Active Signal Status (Anti-Spam Logic)
    
    # Check if a previous signal has concluded (TP/SL hit)
    manager_status, price_diff = signal_manager.check_active_signal_status(symbol, current_price, last_candle)
    
    if manager_status in [TRADE_LIFECYCLE.PROFIT, TRADE_LIFECYCLE.LOSS]:
        # An active trade idea has concluded. Alert and clear the signal manager state.
        logger.info(f"{manager_status.upper()} detected for {symbol}. Alerting and clearing state.")
        
        status_message = format_status_message(symbol, manager_status, last_candle, price_diff)
        safe_send_telegram_message(status_message)
        
        signal_manager.clear_signal(symbol)
        
        # DO NOT process a new signal this cycle to avoid clutter (Optional but recommended)
        return
        
    # Check for NEW Signal
    if signal in ["BUY", "SELL"]:
        
        if signal_manager.get_signal_status(symbol) == TRADE_LIFECYCLE.ACTIVE:
            # New signal received, but a trade is already ACTIVE (Anti-Spam Block)
            logger.info(f"🚫 Ignoring new {signal} for {symbol}. Signal is already ACTIVE.")
            return
            
        # --- Process NEW Signal (BUY/SELL) ---
        
        logger.info(f"🎯 Strategy Signal for {symbol}: {signal}")
        
        # Format the signal message using the current price for freshness
        signal_details['entry_price'] = current_price
        message = format_signal_message(symbol, signal, signal_details)
        
        # Alert the user
        safe_send_telegram_message(message)
        
        # Set the trade to ACTIVE in the manager (Start Anti-Spam Cooldown)
        signal_manager.set_active_signal(symbol, signal, current_price, signal_details)
        
    elif signal_manager.get_signal_status(symbol) == TRADE_LIFECYCLE.ACTIVE:
        # No new signal, but an active trade is running. Log status for debug.
        logger.info(f"⚪ Active trade for {symbol} is running. Price: {current_price:.4f}")
        
    else:
        # No signal and no active trade.
        logger.info(f"⚪ No trade signal for {symbol} | Price: {current_price:.4f}")


def main_loop():
    """The main trading loop that runs periodically."""
    logger.info("🤖 Starting Trading Bot...")
    
    while True:
        start_time = time.time()
        
        for symbol in Config.SYMBOLS:
            try:
                process_symbol(symbol)
            except Exception as e:
                logger.error(f"❌ Critical error processing {symbol}: {e}", exc_info=True)
                # Send a Telegram alert for critical errors
                error_message = f"❌ *CRITICAL ERROR* on {symbol} loop: {escape_markdown(str(e))}"
                safe_send_telegram_message(error_message)

        # Calculate time taken and sleep
        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_duration = max(0, Config.POLLING_INTERVAL_SECONDS - elapsed_time)
        
        if sleep_duration > 0:
            logger.info(f"💤 Sleeping for {sleep_duration:.2f} seconds.")
            time.sleep(sleep_duration)
        else:
            logger.warning(f"⚠️ Processing took longer than polling interval! ({elapsed_time:.2f}s)")

if __name__ == "__main__":
    # Ensure Config is ready before starting the loop
    if not Config.SYMBOLS or not Config.TIMEFRAME:
        logger.error("Configuration is incomplete. Please check settings.py.")
    else:
        main_loop()
