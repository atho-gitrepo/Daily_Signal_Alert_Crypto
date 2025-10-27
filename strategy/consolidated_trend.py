# strategy/consolidated_trend.py (Complete and Fixed)
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import math 

# Assuming 'Indicators' and 'Config' are properly defined in your project
from utils.indicators import Indicators
from settings import Config 

logger = logging.getLogger(__name__)

# --- Strategy Constants (Hard-coded for strict rule enforcement) ---
RRR_RATIO = 1.0 # Target RRR of 1:1
MIN_BB_WIDTH_PERCENT = 0.003 
MAX_BB_WIDTH_PERCENT = 0.03
TDI_NO_TRADE_RANGE = 3 # Margin around 50 for No Trade Zone (i.e., 47 to 53)
MIN_RISK_PERCENT = 0.0015 # 0.15% minimum SL distance relative to entry price
MAX_RISK_PERCENT = 0.02 # 2% maximum SL distance relative to entry price

# --- Strategy Class ---
class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy (Strict 15min) initialized.")
        # Ensure minimum data required for all indicators is known
        self.MIN_KLINES_REQUIRED = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 10
        
        # Robust config check for TDI levels
        required_config = ['TDI_CENTER_LINE', 'TDI_HARD_BUY_LEVEL', 'TDI_SOFT_BUY_LEVEL', 'TDI_HARD_SELL_LEVEL', 'TDI_SOFT_SELL_LEVEL']
        if not all(hasattr(Config, key) for key in required_config):
             logger.error(f"Missing required TDI levels in settings.Config: {required_config}")

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all indicators and prepares data for signal generation.
        Includes robust checks to prevent KeyError if indicators fail.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty in analyze_data. Returning empty DataFrame.")
            return df
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        # --- List of critical columns needed for signal generation ---
        REQUIRED_INDICATORS = [
            'bb_width_percent', 'tdi_strength', 'tdi_slow_ma', 'tdi_fast_ma',
            'bb_rejection_buy', 'bb_rejection_sell', 'atr' # ATR is useful for buffer
        ]

        try:
            # Step 1: Calculate indicators
            df = Indicators.calculate_all_indicators(df) 
            
            # Step 2: Check for missing indicator columns (Critical Fix for KeyError)
            missing_cols = [col for col in REQUIRED_INDICATORS if col not in df.columns]
            if missing_cols:
                logger.error(f"Indicators failed to generate required columns: {missing_cols}")
                return pd.DataFrame() 
                
            # Step 3: Additional market state/strength calculations
            df['market_state'] = np.select(
                [df['bb_width_percent'] < MIN_BB_WIDTH_PERCENT, df['bb_width_percent'] > MAX_BB_WIDTH_PERCENT],
                ['CONSOLIDATION', 'HIGH_VOL'],
                default='NORMAL'
            )
            
            # Step 4: Drop initial indicator NaNs
            df.dropna(inplace=True) 

            # Step 5: Final data size check
            if df.empty:
                logger.warning("DataFrame became empty after dropping NaNs (insufficient data).")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error during indicator calculation: {str(e)}", exc_info=True)
            return pd.DataFrame() # Return empty on calculation failure

        return df

    def _calculate_structural_sl_tp(self, df: pd.DataFrame, last_candle: pd.Series, signal_type: str, risk_factor: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculates SL/TP based on recent swing structure, 1:1 RRR, and risk constraints.
        
        Ensures min/max risk constraints are respected while prioritizing the structural SL point.
        """
        entry_price = last_candle['close']
        
        lookback_period = 10 
        # Lookback DF for structural high/low, excluding the current (last_candle)
        lookback_df = df.iloc[max(0, len(df) - lookback_period - 1):-1] 
        
        default_atr_buffer = entry_price * 0.001 
        # Use ATR for buffer if available
        atr_buffer = last_candle.get('atr', default_atr_buffer) * 0.5 

        # 2. Define risk constraints
        min_sl_distance = entry_price * MIN_RISK_PERCENT
        max_sl_distance = entry_price * MAX_RISK_PERCENT

        final_stop_loss = 0.0
        
        if signal_type == "BUY":
            # Structural SL: Below recent swing low
            swing_low = lookback_df['low'].min()
            structural_stop_loss = swing_low - atr_buffer 
            
            min_risk_sl = entry_price - min_sl_distance
            max_risk_sl = entry_price - max_sl_distance
            
            final_stop_loss = structural_stop_loss
            
            # If Structural SL is too tight (i.e., higher/closer than the minimum required risk SL)
            if final_stop_loss > min_risk_sl:
                final_stop_loss = min_risk_sl
                
            # If Structural SL is too wide (i.e., lower/further than the maximum allowed risk SL)
            if final_stop_loss < max_risk_sl:
                final_stop_loss = max_risk_sl
            
            # Final safety check
            final_stop_loss = min(final_stop_loss, entry_price * 0.999) 
            
        elif signal_type == "SELL":
            # Structural SL: Above recent swing high
            swing_high = lookback_df['high'].max()
            structural_stop_loss = swing_high + atr_buffer
            
            min_risk_sl = entry_price + min_sl_distance
            max_risk_sl = entry_price + max_sl_distance
            
            final_stop_loss = structural_stop_loss
            
            # If Structural SL is too tight (i.e., lower/closer than the minimum required risk SL)
            if final_stop_loss < min_risk_sl:
                final_stop_loss = min_risk_sl
                
            # If Structural SL is too wide (i.e., higher/further than the maximum allowed risk SL)
            if final_stop_loss > max_risk_sl:
                final_stop_loss = max_risk_sl
                
            # Final safety check
            final_stop_loss = max(final_stop_loss, entry_price * 1.001)

        else:
            return entry_price, 0.0, 0.0 

        # 3. Calculate Take Profit based on the FINAL SL distance and RRR
        if signal_type == "BUY":
            sl_distance_final = entry_price - final_stop_loss
            # Apply risk_factor to the RRR (i.e., wider TP for 2x risk)
            take_profit = entry_price + (sl_distance_final * RRR_RATIO * risk_factor) 
            
        elif signal_type == "SELL":
            sl_distance_final = final_stop_loss - entry_price
            # Apply risk_factor to the RRR
            take_profit = entry_price - (sl_distance_final * RRR_RATIO * risk_factor)
        
        # Round final values to a sensible precision
        round_digits = 8 
        final_stop_loss = round(final_stop_loss, round_digits)
        take_profit = round(take_profit, round_digits)
        entry_price = round(entry_price, round_digits)

        return entry_price, final_stop_loss, take_profit

    def _check_volatility_filter(self, bb_width_percent: float) -> bool:
        """Check if volatility is suitable (within MIN/MAX BB width)."""
        return MIN_BB_WIDTH_PERCENT <= bb_width_percent <= MAX_BB_WIDTH_PERCENT

    def _check_tdi_zone(self, tdi_slow_ma: float, signal_type: str) -> Tuple[bool, float]:
        """
        Check TDI zone (1) and assign risk factor (6).
        """
        center = Config.TDI_CENTER_LINE if hasattr(Config, 'TDI_CENTER_LINE') else 50
        
        if signal_type == "BUY":
            if tdi_slow_ma < center:
                if tdi_slow_ma <= Config.TDI_HARD_BUY_LEVEL: 
                    return True, 2.0  # Hard Buy (2x Risk)
                elif tdi_slow_ma <= Config.TDI_SOFT_BUY_LEVEL: 
                    return True, 1.0  # Soft Buy (1x Risk)
                return True, 1.0 
                
        elif signal_type == "SELL":
            if tdi_slow_ma > center:
                if tdi_slow_ma >= Config.TDI_HARD_SELL_LEVEL: 
                    return True, 2.0  # Hard Sell (2x Risk)
                elif tdi_slow_ma >= Config.TDI_SOFT_SELL_LEVEL: 
                    return True, 1.0  # Soft Sell (1x Risk)
                return True, 1.0 

        return False, 1.0
        
    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Implements the Consolidated/Trend Trading Strategy strictly.
        """
        if df.empty or len(df) < self.MIN_KLINES_REQUIRED:
            return "NO_TRADE", {"reason": "Insufficient data to calculate indicators or DataFrame is empty."}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Extract key indicators (These are guaranteed to exist by analyze_data checks)
        tdi_slow_ma = last_candle['tdi_slow_ma']     
        fast_ma = last_candle['tdi_fast_ma']         
        bb_width_percent = last_candle['bb_width_percent']

        # --- 3. No Trade Zone Check ---
        center = Config.TDI_CENTER_LINE if hasattr(Config, 'TDI_CENTER_LINE') else 50
        if abs(tdi_slow_ma - center) <= TDI_NO_TRADE_RANGE:
            return "NO_TRADE", {"reason": "TDI SlowMA stalling near 50 (Indecision Zone)."}

        # --- Volatility Filter Check ---
        if not self._check_volatility_filter(bb_width_percent):
            return "NO_TRADE", {"reason": "Volatility Filter: BB Width outside optimal range."}

        # --- Crossover Check (Rule 2) ---
        bullish_crossover = (fast_ma > tdi_slow_ma) and (prev_candle['tdi_fast_ma'] <= prev_candle['tdi_slow_ma'])
        bearish_crossover = (fast_ma < tdi_slow_ma) and (prev_candle['tdi_fast_ma'] >= prev_candle['tdi_slow_ma'])
        
        # --- BB Rejection Check (Rule 3 & 4) ---
        bb_rejection_buy = last_candle['bb_rejection_buy'] 
        bb_rejection_sell = last_candle['bb_rejection_sell'] 

        # --- FINAL BUY Signal Check ---
        if bullish_crossover and bb_rejection_buy:
            zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "BUY")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "BUY", risk_factor)
                
                if sl != 0.0 and tp != 0.0:
                    signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                    
                    signal_details = {
                        "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                        "risk_factor": risk_factor, "tdi_slow_ma": tdi_slow_ma,
                        "signal_strength": signal_strength,
                        "note": f"Risk {risk_factor:.1f}x ({signal_strength}). RRR {RRR_RATIO}:1."
                    }
                    return "BUY", signal_details

        # --- FINAL SELL Signal Check ---
        if bearish_crossover and bb_rejection_sell:
            zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "SELL")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "SELL", risk_factor)
                
                if sl != 0.0 and tp != 0.0:
                    signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                    
                    signal_details = {
                        "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                        "risk_factor": risk_factor, "tdi_slow_ma": tdi_slow_ma,
                        "signal_strength": signal_strength,
                        "note": f"Risk {risk_factor:.1f}x ({signal_strength}). RRR {RRR_RATIO}:1."
                    }
                    return "SELL", signal_details

        return "NO_TRADE", {"reason": "Conditions for TDI crossover, BB rejection, or TDI zone were not met."}
