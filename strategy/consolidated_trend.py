# strategy/consolidated_trend.py
import pandas as pd
import numpy as np
from utils.indicators import Indicators
from config import Config
import logging

logger = logging.getLogger(__name__)

class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy initialized.")

    def analyze_data(self, df):
        """
        Applies all indicators and prepares data for signal generation.
        Returns the DataFrame with all indicator values.
        """
        if df.empty:
            return df

        # Ensure correct column names (Binance klines return lowercased columns by default)
        df.columns = [col.lower() for col in df.columns]

        # Calculate Super TDI
        df = Indicators.calculate_super_tdi(df.copy())
        # Calculate Super Bollinger Bands
        df = Indicators.calculate_super_bollinger_bands(df.copy())

        # Drop any rows with NaN values introduced by indicator calculations
        # (This is important for the latest candle's complete data)
        df.dropna(inplace=True)

        return df

    def generate_signal(self, df):
        """
        Generates buy/sell signals based on the strategy rules.
        Returns a tuple: (signal_type, signal_details_dict)
        signal_type: "BUY", "SELL", "NO_TRADE"
        signal_details_dict: Contains info like SL, TP, risk_factor, etc.
        """
        if df.empty or len(df) < max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 2:
            logger.warning("DataFrame is too small to generate signal after dropping NaNs.")
            return "NO_TRADE", {}

        # Get the latest candle's data
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Extract indicator values for the latest candle
        current_tdi_rsi = last_candle['rsi']
        current_tdi_fast_ma = last_candle['tdi_fast_ma']
        current_tdi_slow_ma = last_candle['tdi_slow_ma']
        current_bb_lower = last_candle['bb_lower']
        current_bb_upper = last_candle['bb_upper']
        current_close = last_candle['close']
        current_open = last_candle['open']
        current_high = last_candle['high']
        current_low = last_candle['low']

        # Previous values for crossover detection
        prev_tdi_fast_ma = prev_candle['tdi_fast_ma']
        prev_tdi_slow_ma = prev_candle['tdi_slow_ma']


        # --- No Trade Zone ---
        if Config.TDI_NO_TRADE_ZONE_START <= current_tdi_rsi <= Config.TDI_NO_TRADE_ZONE_END:
            # Check if TDI Fast/Slow MA are also flat or crossing within this zone
            if abs(current_tdi_fast_ma - current_tdi_slow_ma) < 0.5: # Small spread indicates stalling
                logger.info(f"TDI ({current_tdi_rsi:.2f}) stalling around 50. No Trade Zone.")
                return "NO_TRADE", {"reason": "TDI stalling around 50."}

        # --- Buy (Long) Setup ---
        is_tdi_in_buyer_zone = (current_tdi_rsi < 50 or
                                (Config.TDI_HARD_BUY_LEVEL <= current_tdi_rsi <= Config.TDI_SOFT_BUY_LEVEL))
        is_tdi_green_above_red_crossover = (current_tdi_fast_ma > current_tdi_slow_ma and
                                            prev_tdi_fast_ma <= prev_tdi_slow_ma)
        is_price_touch_lower_bb = current_close <= current_bb_lower or current_low <= current_bb_lower

        is_bb_rejection_and_reversal_buy = False
        # Look for the previous candle touching/going outside lower BB AND current candle closing above lower BB
        # AND current candle closing higher than previous candle (reversal sign)
        if (prev_candle['low'] <= prev_candle['bb_lower'] or prev_candle['close'] <= prev_candle['bb_lower']) and \
           (current_close > current_bb_lower and current_close > prev_candle['close']):
            is_bb_rejection_and_reversal_buy = True
        # Also incorporate the BB's own basic buy signal if useful
        if last_candle['bb_buy_signal']:
            is_bb_rejection_and_reversal_buy = True


        if (is_tdi_in_buyer_zone and
            is_tdi_green_above_red_crossover and
            is_price_touch_lower_bb and # Price touched or moved outside lower BB
            is_bb_rejection_and_reversal_buy): # Shows rejection and reversal

            # Determine risk factor
            risk_factor = 1 # Not used for trade, but can be part of signal info
            if current_tdi_rsi <= Config.TDI_HARD_BUY_LEVEL: # Hard Buy zone
                risk_factor = 2
                logger.info(f"Hard Buy Signal detected. TDI: {current_tdi_rsi:.2f}")
            elif current_tdi_rsi <= Config.TDI_SOFT_BUY_LEVEL: # Soft Buy zone
                risk_factor = 1
                logger.info(f"Soft Buy Signal detected. TDI: {current_tdi_rsi:.2f}")

            # Calculate Stop Loss (just beyond recent swing low)
            # Find the lowest low in a small lookback window (e.g., last 5-10 candles)
            lookback_period = 10
            # Ensure we don't look beyond available data
            start_index = max(0, len(df) - lookback_period -1) # -1 to exclude current candle, -1 for array indexing
            recent_low = df['low'].iloc[start_index:-1].min() # Exclude current candle
            if recent_low > current_low: # If current candle is the new low, use it
                recent_low = current_low
            stop_loss = recent_low * 0.998 # A small buffer below the swing low
            # Ensure SL is below entry for a long position
            if stop_loss >= current_open:
                stop_loss = current_open * 0.995 # Fallback to a percentage below entry

            # Calculate Take Profit (recent area of rejection/resistance, or 1:1 RRR)
            entry_price = current_open
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + sl_distance

            # Consider recent high as TP, but prioritize RRR if it's better
            recent_high = df['high'].iloc[start_index:-1].max() # Exclude current candle
            if take_profit < recent_high: # If 1:1 RRR is too small, aim for recent resistance
                take_profit = recent_high * 0.998 # A bit below resistance

            logger.info(f"BUY (LONG) Signal generated: TDI: {current_tdi_rsi:.2f}, BB Lower: {current_bb_lower:.4f}")
            return "BUY", {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_factor": risk_factor,
                "tdi_value": current_tdi_rsi
            }

        # --- Sell (Short) Setup ---
        is_tdi_in_seller_zone = (current_tdi_rsi > 50 or
                                (Config.TDI_SOFT_SELL_LEVEL <= current_tdi_rsi <= Config.TDI_HARD_SELL_LEVEL))
        is_tdi_green_below_red_crossover = (current_tdi_fast_ma < current_tdi_slow_ma and
                                            prev_tdi_fast_ma >= prev_tdi_slow_ma)
        is_price_touch_upper_bb = current_close >= current_bb_upper or current_high >= current_bb_upper

        is_bb_rejection_and_reversal_sell = False
        # Look for the previous candle touching/going outside upper BB AND current candle closing below upper BB
        # AND current candle closing lower than previous candle (reversal sign)
        if (prev_candle['high'] >= prev_candle['bb_upper'] or prev_candle['close'] >= prev_candle['bb_upper']) and \
           (current_close < current_bb_upper and current_close < prev_candle['close']):
            is_bb_rejection_and_reversal_sell = True
        # Also incorporate the BB's own basic sell signal if useful
        if last_candle['bb_sell_signal']:
            is_bb_rejection_and_reversal_sell = True


        if (is_tdi_in_seller_zone and
            is_tdi_green_below_red_crossover and
            is_price_touch_upper_bb and
            is_bb_rejection_and_reversal_sell):

            # Determine risk factor
            risk_factor = 1 # Not used for trade, but can be part of signal info
            if current_tdi_rsi >= Config.TDI_HARD_SELL_LEVEL: # Hard Sell zone
                risk_factor = 2
                logger.info(f"Hard Sell Signal detected. TDI: {current_tdi_rsi:.2f}")
            elif current_tdi_rsi >= Config.TDI_SOFT_SELL_LEVEL: # Soft Sell zone
                risk_factor = 1
                logger.info(f"Soft Sell Signal detected. TDI: {current_tdi_rsi:.2f}")

            # Calculate Stop Loss (just beyond recent swing high)
            lookback_period = 10
            start_index = max(0, len(df) - lookback_period - 1)
            recent_high = df['high'].iloc[start_index:-1].max()
            if recent_high < current_high: # If current candle is the new high, use it
                recent_high = current_high
            stop_loss = recent_high * 1.002 # A small buffer above the swing high
            # Ensure SL is above entry for a short position
            if stop_loss <= current_open:
                stop_loss = current_open * 1.005 # Fallback to a percentage above entry

            # Calculate Take Profit (recent area of rejection/support, or 1:1 RRR)
            entry_price = current_open
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - sl_distance

            recent_low = df['low'].iloc[start_index:-1].min()
            if take_profit > recent_low:
                take_profit = recent_low * 1.002 # A bit above support

            logger.info(f"SELL (SHORT) Signal generated: TDI: {current_tdi_rsi:.2f}, BB Upper: {current_bb_upper:.4f}")
            return "SELL", {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_factor": risk_factor,
                "tdi_value": current_tdi_rsi
            }

        logger.info("No valid trade signal found.")
        return "NO_TRADE", {}
