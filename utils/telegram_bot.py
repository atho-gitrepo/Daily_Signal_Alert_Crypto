# utils/telegram_bot.py
import asyncio
import logging
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
# Assuming Config is available and properly defined
from settings import Config  

logger = logging.getLogger(__name__)

# --- Global Initialization ---

# Initialize Telegram Bot object globally
_bot = None
if hasattr(Config, 'TELEGRAM_BOT_TOKEN') and Config.TELEGRAM_BOT_TOKEN and \
   hasattr(Config, 'TELEGRAM_CHAT_ID') and Config.TELEGRAM_CHAT_ID:
    try:
        # We initialize the Bot object here; its network calls are made later in the async function
        _bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        logger.info("✅ Telegram bot initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Telegram bot (Check Token format): {e}")
        _bot = None
else:
    logger.warning("⚠️ Telegram bot token or chat ID missing in settings. Notifications disabled.")


# =============== ASYNC SENDER (The Core) ===============
async def send_telegram_message_async(message: str):
    """
    Sends a Telegram message using the asynchronous Bot method.
    This function is a coroutine and must be awaited.
    """
    if not _bot:
        logger.warning("⚠️ Telegram bot not initialized. Message skipped.")
        return

    try:
        # The key asynchronous network call
        await _bot.send_message(
            chat_id=Config.TELEGRAM_CHAT_ID,
            text=message,
            # Ensure the message is formatted as MarkdownV2
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        logger.info("📩 Telegram message sent successfully.")
    except TelegramError as e:
        # Specific handler for API errors (e.g., bad chat ID, bad message format)
        logger.error(f"⚠️ Telegram API error: {e}. Message: {message}")
    except Exception as e:
        # General unexpected error handler
        logger.error(f"❌ Unexpected error sending Telegram message: {e}")


# =============== SYNC WRAPPER (The Fix) ===============
def send_telegram_message_sync(message: str):
    """
    Sends a Telegram message synchronously from a non-async context (like main.py).
    
    This function contains the robust logic to safely execute the async sender,
    preventing the RuntimeError: Event loop is closed error.
    """
    if not _bot:
        logger.warning("⚠️ Telegram bot not initialized. Message skipped.")
        return

    try:
        # --- Robust Event Loop Execution ---
        
        # 1. Attempt to get the current loop. If it's running, we schedule a task.
        # This handles cases where this function might be called from another async context.
        try:
            loop = asyncio.get_event_loop()
            is_running = loop.is_running()
        except RuntimeError:
            # 2. If no event loop is found, create a new one.
            loop = asyncio.new_event_loop()
            is_running = False

        if is_running:
            # If the current loop is running (rare in your main.py), schedule the task.
            loop.create_task(send_telegram_message_async(message))
        else:
            # This is the expected path in your synchronous main.py loop.
            # Run the async function until completion, preventing loop closure errors.
            loop.run_until_complete(send_telegram_message_async(message))

    except Exception as e:
        logger.error(f"❌ Critical error in sync Telegram wrapper: {e}")

# --- Alias for easy use in main.py ---
# Your main.py should import this function under a convenient name.
# For example: from utils.telegram_bot import send_telegram_message_sync as send_telegram_message
