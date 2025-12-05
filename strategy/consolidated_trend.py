# strategy/consolidated_trend.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional, Any
from utils.indicators import Indicators 
from settings import Config 

logger = logging.getLogger(__name__)

# --- Strategy Constants ---
RRR_RATIO = 1.0 
MIN_BB_WIDTH_PERCENT = 0.003 
MAX_BB_WIDTH_PERCENT = 0.03
TDI_NO_TRADE_RANGE = 5 
MIN_RISK_PERCENT = 0.0015 
MAX_RISK_PERCENT = 0.02 
HTF_TREND_PERIOD = 20 # 20 EMA on 1H chart for trend bias

# --- Strategy Class ---
class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy (5m/1H) initialized.")
        self.MIN_KLINES_REQUIRED = max(
            getattr(Config, 'TDI_RSI_PERIOD', 13),
            getattr(Config, 'TDI_BB_LENGTH', 34),
            getattr(Config, 'BB_PERIOD', 20),
            getattr(Config, 'TDI_SLOW_MA_PERIOD', 7)
        ) + 20 
        
        # CRITICAL: Debounce State
        self.last_signal = "NO_TRADE" 
        self.consecutive_signals = 0
        self.signal_history = []
        
        # NEW: HTF State
        self.HTF_TREND = "NEUTRAL" 

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all indicators and prepares data for signal generation."""
        if df.empty:
            logger.warning("Empty dataframe provided to analyze_data")
            return df
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        try:
            df = Indicators.calculate_all_indicators(df)
            
            if all(col in df.columns for col in ['bb_width_percent', 'tdi_strength']):
                df['market_state'] = np.select(
                    [df['bb_width_percent'] < MIN_BB_WIDTH_PERCENT, df['bb_width_percent'] > MAX_BB_WIDTH_PERCENT],
                    ['CONSOLIDATION', 'HIGH_VOL'],
                    default='NORMAL'
                )
            
            df.dropna(inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def set_htf_trend(self, htf_data: pd.DataFrame):
        """
        Calculates and sets the 1H trend (Bull/Bear/Neutral) based on 20 EMA.
        """
        if htf_data.empty:
            self.HTF_TREND = "NEUTRAL"
            return

        htf_data = Indicators.calculate_ema(htf_data, 'close', HTF_TREND_PERIOD)
        
        last_close = htf_data['close'].iloc[-1]
        last_ema = htf_data[f'close_ema_{HTF_TREND_PERIOD}'].iloc[-1]
        
        if pd.isna(last_ema):
            self.HTF_TREND = "NEUTRAL"
        elif last_close > last_ema:
            self.HTF_TREND = "BULL"
        elif last_close < last_ema:
            self.HTF_TREND = "BEAR"
        else:
            self.HTF_TREND = "NEUTRAL"

        logger.info(f"Higher Timeframe (1H) Trend set to: {self.HTF_TREND}")

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """Calculates SL/TP based on recent swing structure and 1:1 RRR."""
        try:
            entry_price = last_candle['close']
            if entry_price <= 0: return entry_price, 0, 0

            lookback_period = 30 
            start_idx = max(0, len(df) - lookback_period)
            lookback_df = df.iloc[start_idx:-1] 
            
            if lookback_df.empty: return entry_price, 0, 0

            atr_buffer = last_candle.get('atr', entry_price * 0.001) * 0.5 

            if signal_type == "BUY":
                swing_low = lookback_df['low'].min()
                stop_loss = swing_low - atr_buffer 
                sl_distance = entry_price - stop_loss
                min_sl_distance = entry_price * MIN_RISK_PERCENT
                max_sl_distance = entry_price * MAX_RISK_PERCENT
                
                if sl_distance < min_sl_distance: stop_loss = entry_price - min_sl_distance
                elif sl_distance > max_sl_distance: stop_loss = entry_price - max_sl_distance
                    
                sl_distance_final = entry_price - stop_loss
                take_profit = entry_price + (sl_distance_final * RRR_RATIO)

            elif signal_type == "SELL":
                swing_high = lookback_df['high'].max()
                stop_loss = swing_high + atr_buffer
                sl_distance = stop_loss - entry_price
                min_sl_distance = entry_price * MIN_RISK_PERCENT
                max_sl_distance = entry_price * MAX_RISK_PERCENT
                
                if sl_distance < min_sl_distance: stop_loss = entry_price + min_sl_distance
                elif sl_distance > max_sl_distance: stop_loss = entry_price + max_sl_distance
                    
                sl_distance_final = stop_loss - entry_price
                take_profit = entry_price - (sl_distance_final * RRR_RATIO)

            else:
                return entry_price, 0, 0

            if stop_loss <= 0 or take_profit <= 0: return entry_price, 0, 0

            return entry_price, stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error in _calculate_structural_sl_tp: {str(e)}")
            return last_candle['close'], 0, 0

    def _check_volatility_filter(self, bb_width_percent: float) -> bool:
        """Check if volatility is suitable."""
        if pd.isna(bb_width_percent): return False
        if bb_width_percent < MIN_BB_WIDTH_PERCENT: return False
        if bb_width_percent > MAX_BB_WIDTH_PERCENT: return False
        return True

    def _check_tdi_zone(self, tdi_slow_ma: float, signal_type: str) -> Tuple[bool, float]:
        """Check TDI zone and assign risk factor based on HARD/SOFT levels."""
        if pd.isna(tdi_slow_ma): return False, 1.0

        center_line = getattr(Config, 'TDI_CENTER_LINE', 50)
        hard_buy_level = getattr(Config, 'TDI_HARD_BUY_LEVEL', 25)
        soft_buy_level = getattr(Config, 'TDI_SOFT_BUY_LEVEL', 35)
        soft_sell_level = getattr(Config, 'TDI_SOFT_SELL_LEVEL', 65)
        hard_sell_level = getattr(Config, 'TDI_HARD_SELL_LEVEL', 75)

        if signal_type == "BUY":
            if tdi_slow_ma < center_line:
                if tdi_slow_ma <= hard_buy_level: return True, 2.0  # Hard Buy
                elif tdi_slow_ma <= soft_buy_level: return True, 1.0  # Soft Buy
                else: return True, 1.0 
        elif signal_type == "SELL":
            if tdi_slow_ma > center_line:
                if tdi_slow_ma >= hard_sell_level: return True, 2.0  # Hard Sell
                elif tdi_slow_ma >= soft_sell_level: return True, 1.0  # Soft Sell
                else: return True, 1.0 

        return False, 1.0

    def _validate_signal_conditions(self, df) -> Tuple[bool, str]:
        """Validate overall conditions AND align with HTF Trend."""
        if df.empty or len(df) < self.MIN_KLINES_REQUIRED:
            return False, f"Insufficient data. Need {self.MIN_KLINES_REQUIRED}, got {len(df)}"
            
        required_indicators = ['tdi_slow_ma', 'tdi_fast_ma', 'bb_width_percent', 
                              'bb_rejection_buy', 'bb_rejection_sell']
        
        for indicator in required_indicators:
            if indicator not in df.columns:
                return False, f"Missing indicator: {indicator}"
            if df[indicator].iloc[-3:].isna().any():
                return False, f"NaN values in {indicator}"
                
        # CRITICAL: HTF Trend Alignment Filter
        if self.HTF_TREND == "NEUTRAL":
            return False, "HTF Trend Filter: 1H trend is NEUTRAL (20 EMA flat or unconfirmed)."

        return True, "All conditions valid"

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """
        Implements the Consolidated/Trend Trading Strategy with HTF filter and CRITICAL DEBOUNCE.
        """
        conditions_ok, conditions_msg = self._validate_signal_conditions(df)
        if not conditions_ok:
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": conditions_msg}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        try:
            tdi_slow_ma = last_candle['tdi_slow_ma']     
            fast_ma = last_candle['tdi_fast_ma']         
            bb_width_percent = last_candle['bb_width_percent']
        except KeyError as e:
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": f"Missing indicator: {e}"}

        center_line = getattr(Config, 'TDI_CENTER_LINE', 50)
        if abs(tdi_slow_ma - center_line) <= TDI_NO_TRADE_RANGE:
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": f"TDI SlowMA stalling near {center_line} (Indecision Zone)."}

        if not self._check_volatility_filter(bb_width_percent):
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": "Volatility Filter: BB Width outside optimal range."}

        bullish_crossover = (fast_ma > tdi_slow_ma) and (prev_candle['tdi_fast_ma'] <= prev_candle['tdi_slow_ma'])
        bearish_crossover = (fast_ma < tdi_slow_ma) and (prev_candle['tdi_fast_ma'] >= prev_candle['tdi_slow_ma'])
        
        bb_rejection_buy = last_candle['bb_rejection_buy'] 
        bb_rejection_sell = last_candle['bb_rejection_sell'] 

        signal_details = {}

        # --- FINAL BUY Signal Check ---
        if bullish_crossover and bb_rejection_buy:
            # 1. HTF Trend Alignment Filter
            if self.HTF_TREND != "BULL":
                self._update_signal_state("NO_TRADE")
                return "NO_TRADE", {"reason": f"BUY rejected: HTF Trend Mismatch ({self.HTF_TREND})."}
            
            # 2. Debounce Check
            if self.last_signal == "BUY":
                self._update_signal_state("NO_TRADE") 
                return "NO_TRADE", {"reason": "BUY signal suppressed (Debounced)."}
                
            # 3. TDI Zone & Risk Calculation
            zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "BUY")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "BUY", risk_factor)
                
                if sl > 0 and tp > 0:
                    signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                    signal_details = {
                        "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                        "risk_factor": risk_factor, "tdi_slow_ma": tdi_slow_ma,
                        "signal_strength": signal_strength, "bb_width_percent": bb_width_percent,
                        "market_state": last_candle.get('market_state', 'UNKNOWN'),
                    }
                    self._update_signal_state("BUY") 
                    return "BUY", signal_details

        # --- FINAL SELL Signal Check ---
        if bearish_crossover and bb_rejection_sell:
            # 1. HTF Trend Alignment Filter
            if self.HTF_TREND != "BEAR":
                self._update_signal_state("NO_TRADE")
                return "NO_TRADE", {"reason": f"SELL rejected: HTF Trend Mismatch ({self.HTF_TREND})."}

            # 2. Debounce Check
            if self.last_signal == "SELL":
                self._update_signal_state("NO_TRADE") 
                return "NO_TRADE", {"reason": "SELL signal suppressed (Debounced)."}
                
            # 3. TDI Zone & Risk Calculation
            zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "SELL")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "SELL", risk_factor)
                
                if sl > 0 and tp > 0:
                    signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                    signal_details = {
                        "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                        "risk_factor": risk_factor, "tdi_slow_ma": tdi_slow_ma,
                        "signal_strength": signal_strength, "bb_width_percent": bb_width_percent,
                        "market_state": last_candle.get('market_state', 'UNKNOWN'),
                    }
                    self._update_signal_state("SELL")
                    return "SELL", signal_details

        # No trade conditions
        no_trade_reason = "Conditions for TDI crossover, BB rejection, or TDI zone were not met."
        if not (bullish_crossover or bearish_crossover):
            no_trade_reason = "No TDI crossover detected."
        elif not (bb_rejection_buy or bb_rejection_sell):
            no_trade_reason = "No BB rejection detected."
            
        self._update_signal_state("NO_TRADE")
        return "NO_TRADE", {"reason": no_trade_reason}

    def _update_signal_state(self, current_signal: str):
        """Track last signal for debouncing and consecutive count."""
        if current_signal == self.last_signal and current_signal != "NO_TRADE":
            self.consecutive_signals += 1
        else:
            self.consecutive_signals = 1
            self.last_signal = current_signal
            
        self.signal_history.append({'signal': current_signal, 'timestamp': pd.Timestamp.now(), 'consecutive': self.consecutive_signals})
        self.signal_history = self.signal_history[-50:] 

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        return {
            "last_signal": self.last_signal,
            "consecutive_signals": self.consecutive_signals,
            "min_klines_required": self.MIN_KLINES_REQUIRED,
            "htf_trend": self.HTF_TREND
        }
