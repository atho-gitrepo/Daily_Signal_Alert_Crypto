# Binance Futures Notification Bot

This is a Python-based bot designed to send Telegram notifications when specific trading conditions are met on Binance USD-M Futures, based on a Consolidated Trend strategy. This bot **does NOT execute any trades**; it is purely for monitoring and alerting.

## Features

* **Binance Futures Data Integration:** Fetches real-time candlestick data from Binance's public API.
* **Consolidated Trend Strategy:** Implements custom logic using Super TDI and Super Bollinger Bands for signal generation.
* **Telegram Notifications:** Sends real-time alerts for:
    * Bot startup/shutdown
    * Trade signals (BUY/SELL) with proposed entry price, Stop Loss (SL), and Take Profit (TP) details.
    * Errors and warnings.

## Getting Started

### Prerequisites

* Python 3.8+
* Telegram Bot Token and Chat ID

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/futures-notification-bot.git](https://github.com/your-username/futures-notification-bot.git)
    cd futures-notification-bot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set up `config.py`:**
    Open `config.py` and configure the following:
    * `SYMBOL`: The trading pair to monitor (e.g., "BTCUSDT").
    * `TIMEFRAME`: The candlestick interval (e.g., "1h").
    * `POLLING_INTERVAL_SECONDS`: How often the bot checks for new candles.
    * **Strategy Parameters:** Adjust the TDI and Bollinger Band parameters as needed.
    * `TELEGRAM_BOT_TOKEN`: Create a new bot with BotFather on Telegram and get its token.
    * `TELEGRAM_CHAT_ID`: Get your Telegram Chat ID (you can use a bot like `@userinfobot` or `@RawDataBot` to find it).

    **Security Note (Optional API Keys):** While API keys are not strictly required for public data, providing `BINANCE_API_KEY` and `BINANCE_API_SECRET` in `config.py` (or via environment variables) can potentially offer higher API rate limits. If you provide them, ensure they are for an account with Futures enabled, but remember this bot will *not* trade.

    ```python
    # In config.py:
    # BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    # BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    # ...and if you use them, set environment variables before running:
    # export BINANCE_API_KEY="your_actual_key"
    # export BINANCE_API_SECRET="your_actual_secret"
    ```

### Running the Bot

```bash
python main.py
