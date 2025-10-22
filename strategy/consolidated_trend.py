import pandas as pd
import numpy as np
import logging
# Assuming 'Indicators' and 'Config' are properly defined in your project
from utils.indicators import Indicators
from settings import Config 

logger = logging.getLogger(__name__)

# --- Configuration ---
RRR_RATIO = 1.5
# NEW: Minimum BB Width as a percentage of the price (e.g., 0.005 means 0.5% min width)
MIN_BB_WIDTH_PERCENT = 0.003  # Adjusted for 15m crypto volatility (e.g., 0.3%)

# --- Strategy Class ---
class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy (Volatility Filter) initialized.")

    def analyze_data(self, df):
        """
        Applies all indicators and prepares data for signal generation.
        Calculates BB Width for the new filter.
        """
        if df.empty:
            return df

        df.columns = [col.lower() for col in df.columns]

        # Calculate all necessary indicators
        df = Indicators.calculate_super_tdi(df.copy())
        df = Indicators.calculate_super_bollinger_bands(df.copy())

        # NEW: Calculate BB Width Percentage
        df['bb_width_percent'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        df.dropna(inplace=True)

        return df

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """
        Calculates structural Stop Loss and Take Profit based on recent swing high/low 
        and a fixed RRR_RATIO. Lookback period reduced to 7 for 15m sharpness.
        """
        entry_price = last_candle['close']
        lookback_period = 7 # Sharper lookback for 15m
        start_index = max(0, len(df) - lookback_period)
        lookback_df = df.iloc[start_index:-1] # Exclude current candle for SL/TP reference

        if signal_type == "BUY":
            # Using the minimum of the lookback lows as structural SL
            stop_loss = lookback_df['low'].min() * 0.998
            
            # Fallback SL (if structural SL is too close or above entry)
            if entry_price - stop_loss < entry_price * 0.002: # Minimum 0.2% risk
                stop_loss = entry_price * 0.998
                
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * RRR_RATIO * risk_factor)

        elif signal_type == "SELL":
            # Using the maximum of the lookback highs as structural SL
            stop_loss = lookback_df['high'].max() * 1.002
            
            # Fallback SL (if structural SL is too close or below entry)
            if stop_loss - entry_price < entry_price * 0.002: # Minimum 0.2% risk
                stop_loss = entry_price * 1.002
                
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * RRR_RATIO * risk_factor)

        else:
            return entry_price, 0, 0

        return entry_price, stop_loss, take_profit

    def generate_signal(self, df):
        """
        Generates buy/sell signals based on the strategy rules, incorporating 
        the Volatility Filter.
        """
        min_data = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 5
        if df.empty or len(df) < min_data:
            logger.warning(f"DataFrame too small for signal generation (Need >{min_data} rows).")
            return "NO_TRADE", {}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        rsi = last_candle['rsi']
        fast_ma = last_candle['tdi_fast_ma']
        slow_ma = last_candle['tdi_slow_ma']
        close = last_candle['close']
        bb_lower = last_candle['bb_lower']
        bb_upper = last_candle['bb_upper']
        low = last_candle['low']
        high = last_candle['high']
        bb_width_percent = last_candle['bb_width_percent'] # NEW

        risk_factor = 1.0

        # --- NEW: Volatility Filter Check ---
        if bb_width_percent < MIN_BB_WIDTH_PERCENT:
            logger.info(f"VOLATILITY FILTER: BB Width ({bb_width_percent:.4f}) too small. NO TRADE.")
            return "NO_TRADE", {"reason": "Low Volatility (Consolidation)."}

        # --- Crossover Check (Simplified & Faster) ---
        bullish_crossover = (fast_ma > slow_ma) and (prev_candle['tdi_fast_ma'] <= prev_candle['tdi_slow_ma'])
        bearish_crossover = (fast_ma < slow_ma) and (prev_candle['tdi_fast_ma'] >= prev_candle['tdi_slow_ma'])

        # --- BB Rejection Check (Refined) ---
        # Rejection: candle closed back inside BB after touching/breaking the edge
        bb_rejection_buy = (prev_candle['low'] <= prev_candle['bb_lower']) and (close > prev_candle['close']) and (close > bb_lower)
        bb_rejection_sell = (prev_candle['high'] >= prev_candle['bb_upper']) and (close < prev_candle['close']) and (close < bb_upper)

        # --- BUY Signal ---
        if bullish_crossover and rsi < 55 and bb_rejection_buy:
            # Risk factor logic remains, but now only triggers with volatility confirmation
            if rsi <= Config.TDI_HARD_BUY_LEVEL:
                risk_factor = 1.5
                logger.info(f"HARD BUY: Volatility confirmed, TDI Crossover, BB Rejection, and low RSI.")
            elif rsi <= Config.TDI_SOFT_BUY_LEVEL:
                risk_factor = 1.2
                logger.info(f"SOFT BUY: Volatility confirmed, TDI Crossover, BB Rejection.")

            entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "BUY", risk_factor)
            logger.info(f"*** BUY SIGNAL GENERATED *** RSI={rsi:.2f}, SL={sl:.4f}, TP={tp:.4f}")
            return "BUY", {
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_factor": risk_factor,
                "tdi_value": rsi
            }

        # --- SELL Signal ---
        if bearish_crossover and rsi > 45 and bb_rejection_sell:
            if rsi >= Config.TDI_HARD_SELL_LEVEL:
                risk_factor = 1.5
                logger.info(f"HARD SELL: Volatility confirmed, TDI Crossover, BB Rejection, and high RSI.")
            elif rsi >= Config.TDI_SOFT_SELL_LEVEL:
                risk_factor = 1.2
                logger.info(f"SOFT SELL: Volatility confirmed, TDI Crossover, BB Rejection.")

            entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "SELL", risk_factor)
            logger.info(f"*** SELL SIGNAL GENERATED *** RSI={rsi:.2f}, SL={sl:.4f}, TP={tp:.4f}")
            return "SELL", {
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_factor": risk_factor,
                "tdi_value": rsi
            }

        # --- NO TRADE (Fallback) ---
        logger.info("No valid trade signal found, but Volatility is sufficient.")
        return "NO_TRADE", {}
