# strategy/consolidated_trend.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
# Assuming 'Indicators' and 'Config' are properly defined in your project
from utils.indicators import Indicators
from settings import Config 

logger = logging.getLogger(__name__)

# --- Strategy Constants (Hard-coded for strict rule enforcement) ---
RRR_RATIO = 1.0 # Target RRR of 1:1
MIN_BB_WIDTH_PERCENT = 0.003 
MAX_BB_WIDTH_PERCENT = 0.03
TDI_NO_TRADE_RANGE = 3 # Margin around 50 for No Trade Zone (i.e., 47 to 53)
MIN_RISK_PERCENT = 0.0015 # 0.15% minimum SL distance
MAX_RISK_PERCENT = 0.02 # 2% maximum SL distance

# --- Strategy Class ---
class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy (Strict 15min) initialized.")
        # Ensure minimum data required for all indicators is known
        self.MIN_KLINES_REQUIRED = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 10
        
        self.last_signal = "NO_TRADE"
        self.consecutive_signals = 0

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all indicators and prepares data for signal generation."""
        if df.empty:
            return df
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        try:
            df = Indicators.calculate_all_indicators(df)
            
            # Additional market state/strength calculations for filters
            if all(col in df.columns for col in ['bb_width_percent', 'tdi_strength']):
                df['market_state'] = np.select(
                    [df['bb_width_percent'] < MIN_BB_WIDTH_PERCENT, df['bb_width_percent'] > MAX_BB_WIDTH_PERCENT],
                    ['CONSOLIDATION', 'HIGH_VOL'],
                    default='NORMAL'
                )
            
            df.dropna(inplace=True)

        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            return pd.DataFrame()

        return df

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """
        Calculates SL/TP based on recent swing structure and 1:1 RRR.
        """
        entry_price = last_candle['close']
        
        # Use 10 bars for swing low/high approximation
        lookback_period = 10 
        lookback_df = df.iloc[max(0, len(df) - lookback_period):-1]
        
        atr_buffer = last_candle.get('atr', entry_price * 0.001) * 0.5 # Small buffer

        if signal_type == "BUY":
            # SL: Just beyond the recent swing low
            swing_low = lookback_df['low'].min()
            stop_loss = swing_low - atr_buffer 
            
            # Apply min/max risk constraints
            sl_distance = entry_price - stop_loss
            if sl_distance < entry_price * MIN_RISK_PERCENT:
                stop_loss = entry_price - entry_price * MIN_RISK_PERCENT
            if sl_distance > entry_price * MAX_RISK_PERCENT:
                stop_loss = entry_price - entry_price * MAX_RISK_PERCENT
                
            sl_distance_final = entry_price - stop_loss
            take_profit = entry_price + (sl_distance_final * RRR_RATIO)
            
        elif signal_type == "SELL":
            # SL: Just beyond the recent swing high
            swing_high = lookback_df['high'].max()
            stop_loss = swing_high + atr_buffer
            
            # Apply min/max risk constraints
            sl_distance = stop_loss - entry_price
            if sl_distance < entry_price * MIN_RISK_PERCENT:
                stop_loss = entry_price + entry_price * MIN_RISK_PERCENT
            if sl_distance > entry_price * MAX_RISK_PERCENT:
                stop_loss = entry_price + entry_price * MAX_RISK_PERCENT
                
            sl_distance_final = stop_loss - entry_price
            take_profit = entry_price - (sl_distance_final * RRR_RATIO)

        else:
            return entry_price, 0, 0

        return entry_price, stop_loss, take_profit

    def _check_volatility_filter(self, bb_width_percent: float) -> bool:
        """Check if volatility is suitable."""
        if bb_width_percent < MIN_BB_WIDTH_PERCENT:
            return False
            
        if bb_width_percent > MAX_BB_WIDTH_PERCENT:
            return False
            
        return True

    def _check_tdi_zone(self, tdi_slow_ma: float, signal_type: str) -> Tuple[bool, float]:
        """
        Check TDI zone (1) and assign risk factor (6).
        """
        if signal_type == "BUY":
            # Rule 1: Price enters the Buyer Zone (TDI below 50)
            if tdi_slow_ma < Config.TDI_CENTER_LINE:
                if tdi_slow_ma <= Config.TDI_HARD_BUY_LEVEL: # Near 25
                    return True, 2.0  # Hard Buy (2x Risk)
                elif tdi_slow_ma <= Config.TDI_SOFT_BUY_LEVEL: # Near 35
                    return True, 1.0  # Soft Buy (1x Risk)
                else:
                    return True, 1.0 # Default in buyer zone
                
        elif signal_type == "SELL":
            # Rule 1: Price enters the Seller Zone (TDI above 50)
            if tdi_slow_ma > Config.TDI_CENTER_LINE:
                if tdi_slow_ma >= Config.TDI_HARD_SELL_LEVEL: # Near 75
                    return True, 2.0  # Hard Sell (2x Risk)
                elif tdi_slow_ma >= Config.TDI_SOFT_SELL_LEVEL: # Near 65
                    return True, 1.0  # Soft Sell (1x Risk)
                else:
                    return True, 1.0 # Default in seller zone

        return False, 1.0
        
    def generate_signal(self, df):
        """
        Implements the Consolidated/Trend Trading Strategy strictly.
        """
        if df.empty or len(df) < self.MIN_KLINES_REQUIRED:
            return "NO_TRADE", {}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Extract key indicators
        tdi_slow_ma = last_candle['tdi_slow_ma']     # Red Line (Bears MA)
        fast_ma = last_candle['tdi_fast_ma']         # Green Line (Bulls MA)
        bb_width_percent = last_candle['bb_width_percent']

        # --- 3. No Trade Zone Check ---
        # When the TDI (SlowMA) is stalling around the 50 level (centerline).
        if abs(tdi_slow_ma - Config.TDI_CENTER_LINE) <= TDI_NO_TRADE_RANGE:
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": "TDI SlowMA stalling near 50 (Indecision Zone)."}

        # --- Volatility Filter Check (Advanced Tips 7) ---
        if not self._check_volatility_filter(bb_width_percent):
            return "NO_TRADE", {"reason": "Volatility Filter: BB Width outside optimal range."}

        # --- Crossover Check (Rule 2) ---
        bullish_crossover = (fast_ma > tdi_slow_ma) and (prev_candle['tdi_fast_ma'] <= prev_candle['tdi_slow_ma'])
        bearish_crossover = (fast_ma < tdi_slow_ma) and (prev_candle['tdi_fast_ma'] >= prev_candle['tdi_slow_ma'])
        
        # --- BB Rejection Check (Rule 3 & 4) ---
        # Checks if previous bar touched/moved outside BB and current bar closed back inside/above/below.
        bb_rejection_buy = last_candle['bb_rejection_buy'] 
        bb_rejection_sell = last_candle['bb_rejection_sell'] 

        signal_details = {}

        # --- FINAL BUY Signal Check ---
        if bullish_crossover and bb_rejection_buy:
            zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "BUY")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "BUY", risk_factor)
                signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                
                signal_details = {
                    "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                    "risk_factor": risk_factor, "tdi_slow_ma": tdi_slow_ma,
                    "signal_strength": signal_strength,
                    "note": f"Risk {risk_factor:.1f}x (Hard/Soft). RRR 1:1."
                }
                
                self._update_signal_state("BUY")
                return "BUY", signal_details

        # --- FINAL SELL Signal Check ---
        if bearish_crossover and bb_rejection_sell:
            zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "SELL")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "SELL", risk_factor)
                signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                
                signal_details = {
                    "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                    "risk_factor": risk_factor, "tdi_slow_ma": tdi_slow_ma,
                    "signal_strength": signal_strength,
                    "note": f"Risk {risk_factor:.1f}x (Hard/Soft). RRR 1:1."
                }
                
                self._update_signal_state("SELL")
                return "SELL", signal_details

        self._update_signal_state("NO_TRADE")
        return "NO_TRADE", {"reason": "Conditions for TDI crossover, BB rejection, or TDI zone were not met."}

    def _update_signal_state(self, current_signal: str):
        """Track consecutive signals to avoid overtrading"""
        if current_signal == self.last_signal and current_signal != "NO_TRADE":
            self.consecutive_signals += 1
        else:
            self.consecutive_signals = 1
            self.last_signal = current_signal

    def get_strategy_stats(self) -> Dict:
        """Get current strategy statistics"""
        return {
            "last_signal": self.last_signal,
            "consecutive_signals": self.consecutive_signals,
            "min_bb_width": MIN_BB_WIDTH_PERCENT,
            "max_bb_width": MAX_BB_WIDTH_PERCENT,
            "rrr_ratio": RRR_RATIO
        }