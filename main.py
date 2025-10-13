# main.py
import time
import logging
import asyncio
from datetime import datetime

from utils.binance_data_client import BinanceDataClient 
from utils.telegram_bot import send_telegram_message_sync, send_telegram_message_async
from strategy.consolidated_trend import ConsolidatedTrendStrategy
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Helper function for escaping Markdown V2 special characters in dynamic strings
def escape_markdown_v2(text):
    return text.replace('.', '\.').replace('(', '\(').replace(')', '\)').replace('`', '\`')

class NotificationBot:
    def __init__(self):
        try:
            self.data_client = BinanceDataClient() 
        except Exception as e:
            logger.critical(f"Failed to initialize BinanceDataClient: {e}. Exiting bot.")
            
            error_type = type(e).__name__
            error_message = str(e)
            escaped_error_message = escape_markdown_v2(error_message)

            send_telegram_message_sync(f"🚨 *CRITICAL ERROR:* Bot failed to initialize Binance Data Client: `{error_type}: {escaped_error_message}`\. Bot shutting down\.")
            exit()

        self.strategy = ConsolidatedTrendStrategy()
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME

        self.last_candle_close_time = None
        self.last_sent_signal = {"type": None, "timestamp": None}

        logger.info(f"Notification bot initialized for {self.symbol} on {self.timeframe} timeframe (Binance REAL Data).")

    async def run(self):
        logger.info("Starting notification bot...")
        
        status_msg = f"🚀 Notification bot started for *{self.symbol}* on *{self.timeframe}* timeframe \(Binance REAL Data\)\."
        await send_telegram_message_async(status_msg)

        while True:
            try:
                # 1. Fetch data
                df = self.data_client.get_historical_klines(self.symbol, self.timeframe)
                
                min_data_points = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) * 2
                if df.empty or len(df) < min_data_points:
                    logger.warning(f"Not enough historical klines ({len(df)}) for indicators to warm up (min {min_data_points})\. Retrying\.")
                    time.sleep(Config.POLLING_INTERVAL_SECONDS)
                    continue

                latest_complete_candle_close_time = df.index[-1].to_pydatetime()

                if self.last_candle_close_time is not None and latest_complete_candle_close_time <= self.last_candle_close_time:
                    logger.info(f"Waiting for new candle close (last processed: {self.last_candle_close_time})\. Current candle ends at {latest_complete_candle_close_time}\.")
                    time.sleep(Config.POLLING_INTERVAL_SECONDS)
                    continue

                self.last_candle_close_time = latest_complete_candle_close_time
                logger.info(f"Processing new complete candle ending at {latest_complete_candle_close_time}")
                
                # Fetch current price
                current_price = self.data_client.get_current_price()
                if current_price is None:
                    logger.warning("Could not get current mark price\. Proceeding with signal generation based on historical close\.")
                    current_price = df['close'].iloc[-1] 
                    
                # 2. Analyze data and generate signal
                df_indicators = self.strategy.analyze_data(df.copy())
                signal_type, signal_details = self.strategy.generate_signal(df_indicators)

                # 3. Send Notification
                if signal_type in ["BUY", "SELL"] and \
                   (self.last_sent_signal["type"] != signal_type or
                    self.last_sent_signal["timestamp"] != latest_complete_candle_close_time):
                    
                    logger.info(f"New {signal_type} signal detected!")
                    
                    price_precision = self.data_client.price_precision
                    entry_price_display = self.data_client._round_price(signal_details.get('entry_price'))
                    stop_loss_display = self.data_client._round_price(signal_details.get('stop_loss'))
                    take_profit_display = self.data_client._round_price(signal_details.get('take_profit'))
                    current_price_display = self.data_client._round_price(current_price)

                    message = (
                        f"🔔 *NEW {signal_type} Signal for {self.symbol} ({self.timeframe}) \(Binance REAL Data\)*\n"
                        f"Proposed Entry: `{entry_price_display:.{price_precision}f}`\n"
                        f"Proposed Stop Loss: `{stop_loss_display:.{price_precision}f}`\n"
                        f"Proposed Take Profit: `{take_profit_display:.{price_precision}f}`\n"
                        f"TDI: `{signal_details.get('tdi_value', 0.0):.2f}`\n"
                        f"Risk Factor: `{signal_details.get('risk_factor', 1.0)}x` \(Strategy's internal risk estimate\)\n"
                        f"Current Market Price: `{current_price_display:.{price_precision}f}` \(Live via Binance\)"
                    )
                    
                    await send_telegram_message_async(message)
                    
                    self.last_sent_signal["type"] = signal_type
                    self.last_sent_signal["timestamp"] = latest_complete_candle_close_time
                else:
                    logger.info(f"No new signal or signal already notified for candle ending {latest_complete_candle_close_time}\.")

            except Exception as e:
                logger.exception(f"An unexpected error occurred in main loop: {e}")
                
                error_type = type(e).__name__
                error_message = str(e)
                escaped_error_message = escape_markdown_v2(error_message)

                await send_telegram_message_async(f"🚨 *Bot Error!* An unexpected error occurred: `{error_type}: {escaped_error_message}`")

            finally:
                logger.info(f"Sleeping for {Config.POLLING_INTERVAL_SECONDS} seconds...")
                time.sleep(Config.POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    bot = NotificationBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)\.")
        send_telegram_message_sync("👋 Notification bot stopped by user\.")
    except Exception as e:
        logger.exception("Fatal error, bot shutting down\.")
        
        error_type = type(e).__name__
        error_message = str(e)
        escaped_error_message = escape_markdown_v2(error_message)

        send_telegram_message_sync(f"❌ *Fatal Bot Error!* Bot has shut down: `{error_type}: {escaped_error_message}`")
