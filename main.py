import time
import logging
import asyncio
from datetime import datetime

# Import the new CoinGecko data client
from utils.coingecko_data_client import CoinGeckoDataClient 

# --- FIX: Import the ASYNCHRONOUS Telegram function for use in the async loop ---
from utils.telegram_bot import send_telegram_message_sync, send_telegram_message_async
# --- END FIX ---

from strategy.consolidated_trend import ConsolidatedTrendStrategy
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot.log"), # Log to file
                        logging.StreamHandler()        # Log to console
                    ])
logger = logging.getLogger(__name__)

class NotificationBot:
    def __init__(self):
        try:
            # Use the new CoinGecko data client
            self.data_client = CoinGeckoDataClient() 
        except Exception as e:
            logger.critical(f"Failed to initialize CoinGeckoDataClient: {e}. Exiting bot.")
            # Note: This sync call works here because it's in the synchronous __init__
            send_telegram_message_sync(f"🚨 *CRITICAL ERROR:* Bot failed to initialize CoinGecko Data Client: `{e}`. Bot shutting down.")
            exit() # Exit if data client cannot be initialized

        self.strategy = ConsolidatedTrendStrategy()
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME

        self.last_candle_close_time = None
        
        # Track last sent signal to avoid spamming the same signal repeatedly on the same candle
        self.last_sent_signal = {"type": None, "timestamp": None}

        logger.info(f"Notification bot initialized for {self.symbol} on {self.timeframe} timeframe (CoinGecko Mock Data).")

    async def run(self):
        logger.info("Starting notification bot...")
        
        # --- FIX: Use the ASYNC function for the startup message within the async loop ---
        await send_telegram_message_async(f"🚀 Notification bot started for *{self.symbol}* on *{self.timeframe}* timeframe (CoinGecko Mock Data).")
        # --- END FIX ---

        while True:
            try:
                # 1. Fetch data
                # This now calls the CoinGecko client, which returns mock historical data
                df = self.data_client.get_historical_klines(self.symbol, self.timeframe)
                
                # Ensure enough data for indicator calculation
                min_data_points = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) * 2
                if df.empty or len(df) < min_data_points:
                    logger.warning(f"Not enough historical klines ({len(df)}) for indicators to warm up (min {min_data_points}). Retrying...")
                    time.sleep(Config.POLLING_INTERVAL_SECONDS)
                    continue

                # Check if a new candle has closed
                latest_complete_candle_close_time = df.index[-1].to_pydatetime()

                if self.last_candle_close_time is not None and latest_complete_candle_close_time <= self.last_candle_close_time:
                    logger.info(f"Waiting for new candle close (last processed: {self.last_candle_close_time}). Current candle ends at {latest_complete_candle_close_time}.")
                    time.sleep(Config.POLLING_INTERVAL_SECONDS)
                    continue

                self.last_candle_close_time = latest_complete_candle_close_time
                logger.info(f"Processing new complete candle ending at {latest_complete_candle_close_time}")
                
                # Fetch current price (uses the real-time CoinGecko price)
                current_price = self.data_client.get_current_price()
                if current_price is None:
                    logger.warning("Could not get current mark price. Proceeding with signal generation based on historical close.")
                    current_price = df['close'].iloc[-1] # Use the last close from the mock data
                    

                # 2. Analyze data and generate signal
                # NOTE: Signal generation here will use the MOCK data provided by the client,
                # which will likely produce random/unreliable signals.
                df_indicators = self.strategy.analyze_data(df.copy())
                signal_type, signal_details = self.strategy.generate_signal(df_indicators)

                # 3. Send Notification if a signal is generated and it's new for this candle
                if signal_type in ["BUY", "SELL"] and \
                   (self.last_sent_signal["type"] != signal_type or
                    self.last_sent_signal["timestamp"] != latest_complete_candle_close_time):
                    
                    logger.info(f"New {signal_type} signal detected!")
                    
                    # Round SL/TP to appropriate precision for display
                    # Using .get here in case signal_details is missing a key
                    entry_price_display = self.data_client._round_price(signal_details.get('entry_price'))
                    stop_loss_display = self.data_client._round_price(signal_details.get('stop_loss'))
                    take_profit_display = self.data_client._round_price(signal_details.get('take_profit'))
                    current_price_display = self.data_client._round_price(current_price)


                    message = (
                        f"🔔 *NEW {signal_type} Signal for {self.symbol} ({self.timeframe}) (CoinGecko Mock Data)*\n"
                        f"Proposed Entry: `{entry_price_display:.{self.data_client.price_precision}f}`\n"
                        f"Proposed Stop Loss: `{stop_loss_display:.{self.data_client.price_precision}f}`\n"
                        f"Proposed Take Profit: `{take_profit_display:.{self.data_client.price_precision}f}`\n"
                        f"TDI: `{signal_details.get('tdi_value', 0.0):.2f}`\n"
                        f"Risk Factor: `{signal_details.get('risk_factor', 1.0)}x` (Strategy's internal risk estimate)\n"
                        f"Current Market Price: `{current_price_display:.{self.data_client.price_precision}f}` (Live via CoinGecko)"
                    )
                    
                    # --- FIX: Use the ASYNC function for sending the message ---
                    await send_telegram_message_async(message)
                    # --- END FIX ---
                    
                    self.last_sent_signal["type"] = signal_type
                    self.last_sent_signal["timestamp"] = latest_complete_candle_close_time
                else:
                    logger.info(f"No new signal or signal already notified for candle ending {latest_complete_candle_close_time}.")

            except Exception as e:
                logger.exception(f"An unexpected error occurred in main loop: {e}")
                
                # --- FIX: Use the ASYNC function for the error message ---
                await send_telegram_message_async(f"🚨 *Bot Error!* An unexpected error occurred: `{type(e).__name__}: {e}`")
                # --- END FIX ---

            finally:
                logger.info(f"Sleeping for {Config.POLLING_INTERVAL_SECONDS} seconds...")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    bot = NotificationBot()
    try:
        # The entire bot.run() must be run inside asyncio.run()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
        # Use sync wrapper outside the async loop for the shutdown message
        send_telegram_message_sync("👋 Notification bot stopped by user.")
    except Exception as e:
        logger.exception("Fatal error, bot shutting down.")
        # Use sync wrapper outside the async loop for the fatal error message
        send_telegram_message_sync(f"❌ *Fatal Bot Error!* Bot has shut down: `{type(e).__name__}: {e}`")
