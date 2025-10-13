import asyncio
import logging
import re
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

class TelegramUtils:
    """Utility class for handling Telegram formatting."""
    @staticmethod
    def escape_markdown_v2(text):
        """
        Escapes reserved characters in a string for Telegram's MarkdownV2.
        Reserved characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        # Note: Must escape the backslash first, but we are primarily concerned with
        # characters that break the simple status message: () and .
        # Using a minimal escape for simplicity, but the main fix is in main.py message construction.
        return re.sub(r'([()\.\-])', r'\\\1', text)

async def send_telegram_message_async(message):
    """Asynchronously sends a message to the configured Telegram chat."""
    if _bot:
        try:
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

def send_telegram_message_sync(message):
    """
    Synchronous wrapper for sending Telegram messages.
    Used ONLY for genuinely synchronous contexts (like logging setup or final shutdown).
    """
    if not _bot:
        logger.warning(f"Telegram bot not initialized. Could not send sync message: {message}")
        return
        
    try:
        loop = asyncio.get_event_loop()
        
        if loop.is_running():
            # If loop is running, schedule the task without blocking
            loop.create_task(send_telegram_message_async(message))
        else:
            # If no loop is running, run the async function directly (e.g., initialization)
            asyncio.run(send_telegram_message_async(message))
    except RuntimeError as e:
        logger.error(f"Failed to run async Telegram message: {e}. If called from a sub-thread, consider using loop.call_soon_threadsafe.")
    except Exception as e:
        logger.error(f"Unexpected error in synchronous Telegram message wrapper: {e}")
