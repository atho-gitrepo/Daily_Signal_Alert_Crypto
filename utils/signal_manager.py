# utils/signal_manager.py
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# --- Trade Lifecycle Constants ---

class TRADE_LIFECYCLE:
    """Constants for signal status tracking."""
    NONE = "NONE"
    ACTIVE = "ACTIVE"
    PROFIT = "PROFIT"
    LOSS = "LOSS"

class SignalStatus(Enum):
    """Enums for the trade status (for internal use, matching TRADE_LIFECYCLE)."""
    NONE = 0
    ACTIVE = 1
    PROFIT = 2
    LOSS = 3

# --- Markdown V2 Escaping Utility ---

def escape_markdown(text: str) -> str:
    """Escapes characters for Telegram's MarkdownV2 parse mode."""
    # List of characters to escape in MarkdownV2
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return "".join(['\\' + char if char in escape_chars else char for char in text])

# --- Signal Manager Class ---

class SignalManager:
    """
    Manages the lifecycle and status of active trade signals for anti-spam logic.
    Tracks which symbols have an open signal idea until its SL/TP is hit.
    """
    
    def __init__(self):
        # Dictionary to store active signals: 
        # { 'symbol': { 'status': str, 'side': str, 'entry': float, 'sl': float, 'tp': float } }
        self.active_signals: Dict[str, Dict[str, Any]] = {}
        logger.info("SignalManager initialized.")

    def set_active_signal(self, symbol: str, signal_side: str, entry_price: float, details: Dict[str, Any]):
        """
        Sets a new signal as ACTIVE, storing its critical trade parameters.
        This immediately activates the anti-spam cooldown for this symbol.
        """
        self.active_signals[symbol] = {
            'status': TRADE_LIFECYCLE.ACTIVE,
            'side': signal_side,
            'entry': entry_price,
            'sl': details.get('stop_loss', 0.0),
            'tp': details.get('take_profit', 0.0),
        }
        logger.info(f"Signal set to ACTIVE for {symbol}.")

    def clear_signal(self, symbol: str):
        """Clears the active signal state for a symbol."""
        if symbol in self.active_signals:
            del self.active_signals[symbol]
            logger.info(f"Signal state cleared for {symbol}.")

    def get_signal_status(self, symbol: str) -> str:
        """Returns the current lifecycle status for a symbol."""
        return self.active_signals.get(symbol, {}).get('status', TRADE_LIFECYCLE.NONE)

    def check_active_signal_status(self, symbol: str, current_price: float, last_candle: pd.Series) -> Tuple[str, float]:
        """
        Checks if an active signal has hit its SL or TP using the current price.
        
        Returns: 
            (TRADE_LIFECYCLE.PROFIT or TRADE_LIFECYCLE.LOSS or TRADE_LIFECYCLE.ACTIVE), price_difference
        """
        
        if self.get_signal_status(symbol) != TRADE_LIFECYCLE.ACTIVE:
            return TRADE_LIFECYCLE.NONE, 0.0

        data = self.active_signals[symbol]
        sl = data['sl']
        tp = data['tp']
        
        # Calculate the potential P/L in points for the alert
        price_diff = 0.0
        
        if data['side'] == "BUY":
            # Check for LOSS: price drops to or below SL
            if last_candle['low'] <= sl:
                price_diff = sl - data['entry'] # Negative diff
                return TRADE_LIFECYCLE.LOSS, price_diff
            # Check for PROFIT: price rises to or above TP
            elif last_candle['high'] >= tp:
                price_diff = tp - data['entry'] # Positive diff
                return TRADE_LIFECYCLE.PROFIT, price_diff
                
        elif data['side'] == "SELL":
            # Check for LOSS: price rises to or above SL
            if last_candle['high'] >= sl:
                price_diff = data['entry'] - sl # Negative diff
                return TRADE_LIFECYCLE.LOSS, price_diff
            # Check for PROFIT: price drops to or below TP
            elif last_candle['low'] <= tp:
                price_diff = data['entry'] - tp # Positive diff
                return TRADE_LIFECYCLE.PROFIT, price_diff

        # If neither SL nor TP was hit
        return TRADE_LIFECYCLE.ACTIVE, 0.0
