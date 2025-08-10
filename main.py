# main.py
import time
import logging
import asyncio
from datetime import datetime

from utils.binance_data_client import BinanceDataClient # Import the new data client
from utils.telegram_bot import send_telegram_message_sync
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
            self.data_client = BinanceDataClient() # Use the new data client
        except Exception as e:
            logger.critical(f"Failed to initialize BinanceDataClient: {e}. Exiting bot.")
            send_telegram_message_sync(f"🚨 *CRITICAL ERROR:* Bot failed to initialize Binance Data Client: `{e}`. Bot shutting down.")
            exit() # Exit if data client cannot be initialized

        self.strategy = ConsolidatedTrendStrategy()
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME

        self.last_candle_close_time = None
        
        # Track last sent signal to avoid spamming the same signal repeatedly on the same candle
        self.last_sent_signal = {"type": None, "timestamp": None}

        logger.info(f"Notification bot initialized for {self.symbol} on {self.timeframe} timeframe (Futures Data).")

    async def run(self):
        logger.info("Starting notification bot...")
        await send_telegram_message_sync(f"🚀 Notification bot started for *{self.symbol}* on *{self.timeframe}* timeframe (Binance Futures Data).")

        while True:
            try:
                # 1. Fetch data
                df = self.data_client.get_historical_klines(self.symbol, self.timeframe)
                # Ensure enough data for indicator calculation (e.g., for TDI_SLOW_MA_PERIOD or BB_PERIOD)
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
                
                # Fetch current price (optional for signal, but good for context in notification)
                current_price = self.data_client.get_current_price(self.symbol)
                if current_price is None:
                    logger.warning("Could not get current mark price. Proceeding with signal generation based on historical close.")
                    # Optionally, send a Telegram alert if this is persistent

                # 2. Analyze data and generate signal
                df_indicators = self.strategy.analyze_data(df.copy())
                signal_type, signal_details = self.strategy.generate_signal(df_indicators)

                # 3. Send Notification if a signal is generated and it's new for this candle
                if signal_type in ["BUY", "SELL"] and \
                   (self.last_sent_signal["type"] != signal_type or
                    self.last_sent_signal["timestamp"] != latest_complete_candle_close_time):
                    
                    logger.info(f"New {signal_type} signal detected!")
                    
                    # Round SL/TP to appropriate precision for display
                    entry_price_display = self.data_client._round_price(signal_details['entry_price'])
                    stop_loss_display = self.data_client._round_price(signal_details['stop_loss'])
                    take_profit_display = self.data_client._round_price(signal_details['take_profit'])

                    message = (
                        f"🔔 *NEW {signal_type} Signal for {self.symbol} ({self.timeframe}) (Futures Data)*\n"
                        f"Proposed Entry: `{entry_price_display:.{self.data_client.price_precision}f}`\n"
                        f"Proposed Stop Loss: `{stop_loss_display:.{self.data_client.price_precision}f}`\n"
                        f"Proposed Take Profit: `{take_profit_display:.{self.data_client.price_precision}f}`\n"
                        f"TDI: `{signal_details['tdi_value']:.2f}`\n"
                        f"Risk Factor: `{signal_details['risk_factor']}x` (Strategy's internal risk estimate)\n"
                        f"Current Market Price: `{current_price:.{self.data_client.price_precision}f}` (approx.)"
                    )
                    await send_telegram_message_sync(message)
                    self.last_sent_signal["type"] = signal_type
                    self.last_sent_signal["timestamp"] = latest_complete_candle_close_time
                else:
                    logger.info(f"No new signal or signal already notified for candle ending {latest_complete_candle_close_time}.")

            except Exception as e:
                logger.exception(f"An unexpected error occurred in main loop: {e}")
                await send_telegram_message_sync(f"🚨 *Bot Error!* An unexpected error occurred: `{type(e).__name__}: {e}`")

            finally:
                logger.info(f"Sleeping for {Config.POLLING_INTERVAL_SECONDS} seconds...")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    # Ensure you have your Telegram details set in config.py
    # or as environment variables before running the bot.
    # Example (Linux/macOS terminal):
    # export TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
    # export TELEGRAM_CHAT_ID="your_telegram_chat_id_here"
    # python main.py

    bot = NotificationBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
        send_telegram_message_sync("👋 Notification bot stopped by user.")
    except Exception as e:
        logger.exception("Fatal error, bot shutting down.")
        send_telegram_message_sync(f"❌ *Fatal Bot Error!* Bot has shut down: `{type(e).__name__}: {e}`")
