import asyncio
import logging
from config import Config

# --- START OF FIX ---
# Correct imports for modern python-telegram-bot
# We import ParseMode from telegram.constants
from telegram import Bot
from telegram.constants import ParseMode 
from telegram.error import TelegramError
# --- END OF FIX ---

logger = logging.getLogger(__name__)

# Initialize the bot once at module level
_bot = None
if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
    try:
        # Use Bot imported explicitly
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
            # --- FIX APPLIED HERE ---
            # Use the correctly imported ParseMode
            await _bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID, 
                text=message, 
                parse_mode=ParseMode.MARKDOWN_V2 # Correct usage
            )
            logger.debug(f"Telegram message sent: {message}") # Use debug level for individual messages
        except TelegramError as e: # Use TelegramError instead of telegram.error.BadRequest
            logger.error(f"Telegram API error: {e}. Check chat_id and message format (MarkdownV2 rules). Message: {message}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    else:
        logger.warning(f"Telegram bot not initialized. Could not send message: {message}")

def send_telegram_message_sync(message):
    """
    Synchronous wrapper for sending Telegram messages.
    Handles running async code from a synchronous context.
    """
    if not _bot:
        logger.warning(f"Telegram bot not initialized. Could not send sync message: {message}")
        return
        
    try:
        # Check if an event loop is already running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, schedule the task
            loop.create_task(send_telegram_message_async(message))
        else:
            # If no loop is running, run the async function directly
            asyncio.run(send_telegram_message_async(message))
    except RuntimeError as e:
        # This typically happens if the sync function is called from a thread
        # that doesn't have an event loop yet.
        logger.error(f"Failed to run async Telegram message: {e}. If called from a sub-thread, consider using loop.call_soon_threadsafe.")
    except Exception as e:
        logger.error(f"Unexpected error in synchronous Telegram message wrapper: {e}")
