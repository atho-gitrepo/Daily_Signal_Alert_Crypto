# utils/telegram_bot.py

import asyncio
import logging
from config import Config
from telegram import Bot
from telegram.constants import ParseMode 
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

# Initialize the bot once at module level
_bot = None
if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
    try:
        _bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        logger.info("Telegram bot initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot: {e}. Telegram notifications will be disabled.")
        _bot = None
else:
    logger.warning("Telegram bot token or chat ID not configured in config.py. Telegram notifications will be disabled.")


async def send_telegram_message_async(message):
    """Asynchronously sends a message to the configured Telegram chat."""
    if _bot:
        try:
            # Using ParseMode.MARKDOWN_V2 as previously agreed
            await _bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID, 
                text=message, 
                parse_mode=ParseMode.MARKDOWN_V2
            )
            logger.debug(f"Telegram message sent: {message}") 
        except TelegramError as e:
            logger.error(f"Telegram API error: {e}. Check chat_id and message format (MarkdownV2 rules). Message: {message}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    else:
        logger.warning(f"Telegram bot not initialized. Could not send message: {message}")

# --- FIX APPLIED HERE: The wrapper no longer attempts to run a new loop ---
def send_telegram_message_sync(message):
    """
    Synchronous wrapper for sending Telegram messages.
    This is ONLY for use in genuinely synchronous contexts (like logging setup or schedule).
    """
    if not _bot:
        logger.warning(f"Telegram bot not initialized. Could not send sync message: {message}")
        return
        
    try:
        # Get the running event loop
        loop = asyncio.get_event_loop()
        
        # We always schedule the task onto the current loop (or a new one if none is running)
        if loop.is_running():
            # If loop is running, schedule the task without blocking
            loop.create_task(send_telegram_message_async(message))
        else:
            # If no loop is running (e.g., initialization phase), run the async function directly
            asyncio.run(send_telegram_message_async(message))
    except RuntimeError as e:
        # Fallback for threads without a loop
        logger.error(f"Failed to run async Telegram message: {e}. If called from a sub-thread, consider using loop.call_soon_threadsafe.")
    except Exception as e:
        logger.error(f"Unexpected error in synchronous Telegram message wrapper: {e}")
