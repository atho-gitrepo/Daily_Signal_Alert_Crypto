# strategy/consolidated_trend.py
import pandas as pd
import numpy as np
from utils.indicators import Indicators
from config import Config
import logging

logger = logging.getLogger(__name__)

RRR_RATIO = 1.5 

class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy initialized.")

    def analyze_data(self, df):
        """
        Applies all indicators and prepares data for signal generation.
        """
        if df.empty:
            return df

        df.columns = [col.lower() for col in df.columns]

        df = Indicators.calculate_super_tdi(df.copy())
        df = Indicators.calculate_super_bollinger_bands(df.copy())

        df.dropna(inplace=True)

        return df

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """
        Calculates structural Stop Loss and Take Profit based on recent swing high/low 
        and a fixed RRR_RATIO.
        """
        entry_price = last_candle['close'] 
        lookback_period = 10
        start_index = max(0, len(df) - lookback_period) 
        lookback_df = df.iloc[start_index:]

        if signal_type == "BUY":
            # Structural Stop Loss: Lowest low of the lookback period, with a small buffer
            stop_loss = lookback_df['low'].min() * 0.998
            
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995 # Fallback

            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * RRR_RATIO * risk_factor)

        elif signal_type == "SELL":
            # Structural Stop Loss: Highest high of the lookback period, with a small buffer
            stop_loss = lookback_df['high'].max() * 1.002

            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005 # Fallback
            
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * RRR_RATIO * risk_factor)

        else:
            return entry_price, 0, 0 

        return entry_price, stop_loss, take_profit


    def generate_signal(self, df):
        """
        Generates buy/sell signals based on the strategy rules.
        """
        min_data = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 2
        if df.empty or len(df) < min_data:
            logger.warning(f"DataFrame is too small to generate signal after dropping NaNs (Need >{min_data} rows).")
            return "NO_TRADE", {}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        current_tdi_rsi = last_candle['rsi']
        current_tdi_fast_ma = last_candle['tdi_fast_ma']
        current_tdi_slow_ma = last_candle['tdi_slow_ma']
        current_bb_lower = last_candle['bb_lower']
        current_bb_upper = last_candle['bb_upper']
        current_close = last_candle['close']
        
        prev_tdi_fast_ma = prev_candle['tdi_fast_ma']
        prev_tdi_slow_ma = prev_candle['tdi_slow_ma']
        
        risk_factor = 1.0

        is_consolidating_tdi = (Config.TDI_NO_TRADE_ZONE_START <= current_tdi_rsi <= Config.TDI_NO_TRADE_ZONE_END) and \
                               (abs(current_tdi_fast_ma - current_tdi_slow_ma) < 0.5)

        # --- Buy (Long) Signal Check ---
        is_tdi_bullish_crossover = (current_tdi_fast_ma > current_tdi_slow_ma and
                                    prev_tdi_fast_ma <= prev_tdi_slow_ma)
        is_tdi_in_buy_zone = (current_tdi_rsi < 50)
        is_bb_rejection_buy = (prev_candle['low'] <= prev_candle['bb_lower'] or last_candle['low'] <= current_bb_lower) and \
                              (current_close > prev_candle['close']) and \
                              (current_close > current_bb_lower)
        
        if is_tdi_bullish_crossover and is_tdi_in_buy_zone and is_bb_rejection_buy:

            if current_tdi_rsi <= Config.TDI_HARD_BUY_LEVEL: 
                risk_factor = 1.5 
                logger.info(f"Hard Buy Signal (TDI < {Config.TDI_HARD_BUY_LEVEL}) detected!")
            elif current_tdi_rsi <= Config.TDI_SOFT_BUY_LEVEL: 
                risk_factor = 1.2
                logger.info(f"Soft Buy Signal detected. TDI: {current_tdi_rsi:.2f}")

            entry_price, stop_loss, take_profit = self._calculate_structural_sl_tp(
                df, last_candle, "BUY", risk_factor
            )
            
            logger.info(f"BUY (LONG) Signal generated: TDI: {current_tdi_rsi:.2f}")
            return "BUY", {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_factor": risk_factor,
                "tdi_value": current_tdi_rsi
            }

        # --- Sell (Short) Signal Check ---
        is_tdi_bearish_crossover = (current_tdi_fast_ma < current_tdi_slow_ma and
                                    prev_tdi_fast_ma >= prev_tdi_slow_ma)
        is_tdi_in_sell_zone = (current_tdi_rsi > 50)
        is_bb_rejection_sell = (prev_candle['high'] >= prev_candle['bb_upper'] or last_candle['high'] >= current_bb_upper) and \
                               (current_close < prev_candle['close']) and \
                               (current_close < current_bb_upper)

        if is_tdi_bearish_crossover and is_tdi_in_sell_zone and is_bb_rejection_sell:

            if current_tdi_rsi >= Config.TDI_HARD_SELL_LEVEL: 
                risk_factor = 1.5 
                logger.info(f"Hard Sell Signal (TDI > {Config.TDI_HARD_SELL_LEVEL}) detected!")
            elif current_tdi_rsi >= Config.TDI_SOFT_SELL_LEVEL: 
                risk_factor = 1.2
                logger.info(f"Soft Sell Signal detected. TDI: {current_tdi_rsi:.2f}")

            entry_price, stop_loss, take_profit = self._calculate_structural_sl_tp(
                df, last_candle, "SELL", risk_factor
            )

            logger.info(f"SELL (SHORT) Signal generated: TDI: {current_tdi_rsi:.2f}")
            return "SELL", {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_factor": risk_factor,
                "tdi_value": current_tdi_rsi
            }
        
        # --- Final NO_TRADE Check ---
        if is_consolidating_tdi:
            logger.info(f"TDI ({current_tdi_rsi:.2f}) stalling around 50. No Trade Zone.")
            return "NO_TRADE", {"reason": "TDI stalling around 50."}


        logger.info("No valid trade signal found.")
        return "NO_TRADE", {}
