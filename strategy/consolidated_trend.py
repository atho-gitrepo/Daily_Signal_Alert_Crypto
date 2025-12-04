# strategy/consolidated_trend.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from utils.indicators import Indicators # Assuming this is the correct import
from settings import Config 

logger = logging.getLogger(__name__)

# --- Strategy Constants (Hard-coded for strict rule enforcement) ---
RRR_RATIO = 1.0 # Target RRR of 1:1
MIN_BB_WIDTH_PERCENT = 0.003 
MAX_BB_WIDTH_PERCENT = 0.03
TDI_NO_TRADE_RANGE = 5 # Margin around 50 for No Trade Zone (i.e., 47 to 53)
MIN_RISK_PERCENT = 0.0015 # 0.15% minimum SL distance
MAX_RISK_PERCENT = 0.02 # 2% maximum SL distance

# --- Strategy Class ---
class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy (Super BB + TDI) initialized.")
        # Ensure minimum data required for all indicators is known
        self.MIN_KLINES_REQUIRED = max(
            getattr(Config, 'TDI_RSI_PERIOD', 14),
            getattr(Config, 'TDI_BB_LENGTH', 34),
            getattr(Config, 'BB_PERIOD', 20),
            getattr(Config, 'TDI_SLOW_MA_PERIOD', 5),
            200 # For 200 EMA
        ) + 20  # Added buffer for stable calculations
        
        self.last_signal = "NO_TRADE"
        self.consecutive_signals = 0
        self.signal_history = []

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all indicators and prepares data for signal generation."""
        if df.empty:
            logger.warning("Empty dataframe provided to analyze_data")
            return df
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        try:
            logger.info(f"Analyzing data with {len(df)} candles...")
            
            # Calculate all indicators
            df = Indicators.calculate_all_indicators(df)
            
            # Additional market state/strength calculations for filters
            if all(col in df.columns for col in ['bb_width_percent', 'tdi_strength']):
                df['market_state'] = np.select(
                    [df['bb_width_percent'] < MIN_BB_WIDTH_PERCENT, df['bb_width_percent'] > MAX_BB_WIDTH_PERCENT],
                    ['CONSOLIDATION', 'HIGH_VOL'],
                    default='NORMAL'
                )
                
                # Log market state for current candle
                if len(df) > 0:
                    last_state = df['market_state'].iloc[-1]
                    last_bb_width = df['bb_width_percent'].iloc[-1] * 100
                    logger.info(f"Market State: {last_state}, BB Width: {last_bb_width:.3f}%")
            
            # Validate we have enough data after indicator calculation
            if len(df) < 15:  # Need at least 5 candles for reliable signals
                logger.warning(f"Insufficient data after indicator calculation: {len(df)} candles")
                return pd.DataFrame()
                
            df.dropna(inplace=True)
            
            logger.info(f"Data analysis completed. Final shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """
        Calculates SL/TP based on recent swing structure and 1:1 RRR.
        Enhanced with better error handling.
        """
        try:
            entry_price = last_candle['close']
            
            if entry_price <= 0:
                logger.error(f"Invalid entry price: {entry_price}")
                return entry_price, 0, 0

            # Use 30 bars for swing low/high approximation
            lookback_period = 30 
            start_idx = max(0, len(df) - lookback_period)
            lookback_df = df.iloc[start_idx:-1]  # Exclude current candle
            
            if lookback_df.empty:
                logger.warning("No data available for swing structure analysis")
                return entry_price, 0, 0

            atr_buffer = last_candle.get('atr', entry_price * 0.001) * 0.5  # Small buffer

            if signal_type == "BUY":
                # SL: Just beyond the recent swing low
                swing_low = lookback_df['low'].min()
                stop_loss = swing_low - atr_buffer 
                
                # Apply min/max risk constraints
                sl_distance = entry_price - stop_loss
                min_sl_distance = entry_price * MIN_RISK_PERCENT
                max_sl_distance = entry_price * MAX_RISK_PERCENT
                
                if sl_distance < min_sl_distance:
                    stop_loss = entry_price - min_sl_distance
                    logger.debug(f"SL adjusted to minimum risk: {stop_loss:.4f}")
                elif sl_distance > max_sl_distance:
                    stop_loss = entry_price - max_sl_distance
                    logger.debug(f"SL adjusted to maximum risk: {stop_loss:.4f}")
                    
                sl_distance_final = entry_price - stop_loss
                take_profit = entry_price + (sl_distance_final * RRR_RATIO)
                
                logger.info(f"BUY - Entry: {entry_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}, Distance: {sl_distance_final:.4f}")

            elif signal_type == "SELL":
                # SL: Just beyond the recent swing high
                swing_high = lookback_df['high'].max()
                stop_loss = swing_high + atr_buffer
                
                # Apply min/max risk constraints
                sl_distance = stop_loss - entry_price
                min_sl_distance = entry_price * MIN_RISK_PERCENT
                max_sl_distance = entry_price * MAX_RISK_PERCENT
                
                if sl_distance < min_sl_distance:
                    stop_loss = entry_price + min_sl_distance
                    logger.debug(f"SL adjusted to minimum risk: {stop_loss:.4f}")
                elif sl_distance > max_sl_distance:
                    stop_loss = entry_price + max_sl_distance
                    logger.debug(f"SL adjusted to maximum risk: {stop_loss:.4f}")
                    
                sl_distance_final = stop_loss - entry_price
                take_profit = entry_price - (sl_distance_final * RRR_RATIO)

                logger.info(f"SELL - Entry: {entry_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}, Distance: {sl_distance_final:.4f}")

            else:
                return entry_price, 0, 0

            # Final validation
            if stop_loss <= 0 or take_profit <= 0:
                logger.error(f"Invalid SL/TP calculated: SL={stop_loss:.4f}, TP={take_profit:.4f}")
                return entry_price, 0, 0

            return entry_price, stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error in _calculate_structural_sl_tp: {str(e)}")
            return last_candle['close'], 0, 0

    def _check_volatility_filter(self, bb_width_percent: float) -> bool:
        """Check if volatility is suitable."""
        if pd.isna(bb_width_percent):
            logger.warning("BB width percent is NaN")
            return False
            
        if bb_width_percent < MIN_BB_WIDTH_PERCENT:
            logger.debug(f"Volatility too low: {bb_width_percent:.4f} < {MIN_BB_WIDTH_PERCENT}")
            return False
            
        if bb_width_percent > MAX_BB_WIDTH_PERCENT:
            logger.debug(f"Volatility too high: {bb_width_percent:.4f} > {MAX_BB_WIDTH_PERCENT}")
            return False
            
        return True

    def _check_tdi_zone(self, tdi_slow_ma: float, signal_type: str) -> Tuple[bool, float]:
        """
        Check TDI zone (1) and assign risk factor (6).
        Enhanced with NaN protection.
        """
        if pd.isna(tdi_slow_ma):
            logger.warning("TDI Slow MA is NaN in zone check")
            return False, 1.0

        center_line = getattr(Config, 'TDI_CENTER_LINE', 50)
        hard_buy_level = getattr(Config, 'TDI_HARD_BUY_LEVEL', 25)
        soft_buy_level = getattr(Config, 'TDI_SOFT_BUY_LEVEL', 35)
        soft_sell_level = getattr(Config, 'TDI_SOFT_SELL_LEVEL', 65)
        hard_sell_level = getattr(Config, 'TDI_HARD_SELL_LEVEL', 75)

        if signal_type == "BUY":
            # Rule 1: Price enters the Buyer Zone (TDI below 50)
            if tdi_slow_ma < center_line:
                if tdi_slow_ma <= hard_buy_level: # Near 25
                    logger.debug(f"HARD BUY zone: TDI Slow MA = {tdi_slow_ma:.2f}")
                    return True, 2.0  # Hard Buy (2x Risk)
                elif tdi_slow_ma <= soft_buy_level: # Near 35
                    logger.debug(f"SOFT BUY zone: TDI Slow MA = {tdi_slow_ma:.2f}")
                    return True, 1.0  # Soft Buy (1x Risk)
                else:
                    logger.debug(f"BUY ZONE: TDI Slow MA = {tdi_slow_ma:.2f}")
                    return True, 1.0 # Default in buyer zone
                
        elif signal_type == "SELL":
            # Rule 1: Price enters the Seller Zone (TDI above 50)
            if tdi_slow_ma > center_line:
                if tdi_slow_ma >= hard_sell_level: # Near 75
                    logger.debug(f"HARD SELL zone: TDI Slow MA = {tdi_slow_ma:.2f}")
                    return True, 2.0  # Hard Sell (2x Risk)
                elif tdi_slow_ma >= soft_sell_level: # Near 65
                    logger.debug(f"SOFT SELL zone: TDI Slow MA = {tdi_slow_ma:.2f}")
                    return True, 1.0  # Soft Sell (1x Risk)
                else:
                    logger.debug(f"SELL ZONE: TDI Slow MA = {tdi_slow_ma:.2f}")
                    return True, 1.0 # Default in seller zone

        logger.debug(f"TDI zone not suitable for {signal_type}: TDI Slow MA = {tdi_slow_ma:.2f}")
        return False, 1.0

    def _validate_signal_conditions(self, df) -> Tuple[bool, str]:
        """Validate overall conditions before signal generation."""
        if df.empty:
            return False, "Empty dataframe"
            
        if len(df) < self.MIN_KLINES_REQUIRED:
            return False, f"Insufficient data. Need {self.MIN_KLINES_REQUIRED}, got {len(df)}"
            
        # Check if indicators are properly calculated
        required_indicators = ['tdi_slow_ma', 'tdi_fast_ma', 'bb_width_percent', 
                              'bb_upper', 'bb_lower', 'bb_middle', 'ema_200',
                              'stoch_k', 'stoch_d'] # Added new BB/MA/Stoch checks
        
        for indicator in required_indicators:
            if indicator not in df.columns:
                return False, f"Missing indicator: {indicator}"
                
            # Check last few candles for NaN values
            last_values = df[indicator].iloc[-3:]  # Check last 3 candles
            if last_values.isna().any():
                return False, f"NaN values in {indicator}"
                
        return True, "All conditions valid"
        
    # --- NEW: BB Signal Checks ---
    def _check_bb_long_breakout(self, current, prev) -> bool:
        """Price closes above Upper Band (breakout)."""
        # Close above Upper Band, confirming the breakout.
        is_breakout = (current['close'] > current['bb_upper'])
        
        # Optional: Check if the previous candle was not already above the upper band
        # is_fresh_breakout = (prev['close'] <= prev['bb_upper']) 

        return is_breakout

    def _check_bb_long_pullback(self, current, prev) -> bool:
        """Price pulls back to Middle Band (20 SMA) and holds."""
        middle_band = current['bb_middle']
        
        # 1. Price is currently above Middle Band (holding the support)
        is_above_middle = current['close'] > middle_band
        
        # 2. Price touched or came near the Middle Band recently (pullback)
        is_pullback_touch = (current['low'] <= middle_band)
        
        # 3. Confirmation: The current candle is bullish (close > open)
        is_bullish_hold = current['close'] > current['open']
        
        # A strong pullback signal is when price touches MB and closes bullishly above it
        return is_above_middle and is_pullback_touch and is_bullish_hold

    def _check_bb_short_breakdown(self, current, prev) -> bool:
        """Price closes below Lower Band (breakdown)."""
        is_breakdown = (current['close'] < current['bb_lower'])
        return is_breakdown

    def _check_bb_short_rejection(self, current, prev) -> bool:
        """Price rallies to Middle Band and rejects."""
        middle_band = current['bb_middle']

        # 1. Price is currently below Middle Band (rejecting the resistance)
        is_below_middle = current['close'] < middle_band

        # 2. Price touched or came near the Middle Band recently (rejection)
        is_rejection_touch = (current['high'] >= middle_band)

        # 3. Confirmation: The current candle is bearish (close < open)
        is_bearish_rejection = current['close'] < current['open']

        # A strong rejection signal is when price touches MB and closes bearishly below it
        return is_below_middle and is_rejection_touch and is_bearish_rejection
        
    # --- NEW: TDI Signal Checks ---
    def _check_tdi_long_signal(self, current) -> bool:
        """Checks RSI > Bulls Line AND Stochastic rising from below 30."""
        
        # 1. RSI > Bulls Line (Using the Fast MA as the 'RSI' signal line, and the 50 line as the Bulls threshold for strong momentum)
        # Note: The rule 'RSI > Bulls Line (2)' is unconventional. We use Fast MA > Slow MA and > 50 for strong trend momentum.
        is_tdi_momentum = (current['tdi_fast_ma'] > current['tdi_slow_ma']) and (current['tdi_fast_ma'] > 50)
        
        # 2. Stochastic %K > %D and rising from below 30
        is_stoch_confirm = (current['stoch_k'] > current['stoch_d'])
        is_stoch_oversold_rising = is_stoch_confirm and (current['stoch_k'] < 30)
        
        # For a long, we want bullish momentum and either a clean stochastic confirmation OR a rising from oversold confirmation
        return is_tdi_momentum and (is_stoch_confirm or is_stoch_oversold_rising)

    def _check_tdi_short_signal(self, current) -> bool:
        """Checks RSI < Bears Line AND Stochastic falling from above 70."""

        # 1. RSI < Bears Line 
        is_tdi_momentum = (current['tdi_fast_ma'] < current['tdi_slow_ma']) and (current['tdi_fast_ma'] < 50)
        
        # 2. Stochastic %K < %D and falling from above 70
        is_stoch_confirm = (current['stoch_k'] < current['stoch_d'])
        is_stoch_overbought_falling = is_stoch_confirm and (current['stoch_k'] > 70)
        
        # For a short, we want bearish momentum and either a clean stochastic confirmation OR a falling from overbought confirmation
        return is_tdi_momentum and (is_stoch_confirm or is_stoch_overbought_falling)
        
    # --- NEW: Trend Filter Check ---
    def _check_trend_filter(self, current, signal_type: str) -> bool:
        """Checks Price > 200 EMA for LONG, Price < 200 EMA for SHORT."""
        ema_200 = current['ema_200']
        price = current['close']
        
        if pd.isna(ema_200):
            logger.warning("200 EMA is NaN, trend filter skipped.")
            return True # Skip filter if data is missing

        if signal_type == "BUY":
            is_uptrend = price > ema_200
            if not is_uptrend:
                logger.debug(f"Trend Filter: Price ({price:.4f}) not above 200 EMA ({ema_200:.4f})")
            return is_uptrend
        
        elif signal_type == "SELL":
            is_downtrend = price < ema_200
            if not is_downtrend:
                logger.debug(f"Trend Filter: Price ({price:.4f}) not below 200 EMA ({ema_200:.4f})")
            return is_downtrend
            
        return False


    def generate_signal(self, df):
        """
        Implements the Combined Super BB + Super TDI Trading System rules.
        """
        logger.info("=== GENERATING SIGNAL (Super BB + TDI) ===")
        
        # Validate overall conditions
        conditions_ok, conditions_msg = self._validate_signal_conditions(df)
        if not conditions_ok:
            logger.warning(f"Signal conditions not met: {conditions_msg}")
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": conditions_msg}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Extract key indicators
        try:
            tdi_slow_ma = last_candle['tdi_slow_ma']
            bb_width_percent = last_candle['bb_width_percent']
        except KeyError as e:
            logger.error(f"Missing essential indicator column: {e}")
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": f"Missing indicator: {e}"}

        # --- Filters (1, 2) ---
        if abs(tdi_slow_ma - 50) <= TDI_NO_TRADE_RANGE:
            logger.debug(f"TDI in No-Trade Zone: {tdi_slow_ma:.2f} near 50")
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": "TDI SlowMA stalling near 50 (Indecision Zone)."}

        if not self._check_volatility_filter(bb_width_percent):
            self._update_signal_state("NO_TRADE")
            return "NO_TRADE", {"reason": "Volatility Filter: BB Width outside optimal range."}


        signal = "NO_TRADE"
        signal_reason = "No entry conditions met."
        risk_factor = 1.0 # Initialize risk factor
        
        # --- LONG ENTRY CHECK ---
        is_long_breakout = self._check_bb_long_breakout(last_candle, prev_candle)
        is_long_pullback = self._check_bb_long_pullback(last_candle, prev_candle)
        
        if is_long_breakout or is_long_pullback:
            
            tdi_long_ok = self._check_tdi_long_signal(last_candle)
            
            if ti_long_ok:
                trend_ok = self._check_trend_filter(last_candle, "BUY")
                zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "BUY")
                
                if trend_ok and zone_ok:
                    # NOTE: Volume check (Rule 3) is a placeholder, as volume data is not available.
                    signal = "BUY"
                    entry_type = 'Breakout' if is_long_breakout else 'Pullback'
                    signal_reason = f"LONG ENTRY: BB {entry_type} + TDI Momentum + Trend Filter. Risk Factor: {risk_factor}x"


        # --- SHORT ENTRY CHECK (Only check if no LONG signal was found) ---
        if signal == "NO_TRADE": 
            is_short_breakdown = self._check_bb_short_breakdown(last_candle, prev_candle)
            is_short_rejection = self._check_bb_short_rejection(last_candle, prev_candle)
            
            if is_short_breakdown or is_short_rejection:
                
                tdi_short_ok = self._check_tdi_short_signal(last_candle)
                
                if tdi_short_ok:
                    trend_ok = self._check_trend_filter(last_candle, "SELL")
                    zone_ok, risk_factor = self._check_tdi_zone(tdi_slow_ma, "SELL")
                    
                    if trend_ok and zone_ok:
                        # NOTE: Volume check (Rule 3) is a placeholder.
                        signal = "SELL"
                        entry_type = 'Breakdown' if is_short_breakdown else 'Rejection'
                        signal_reason = f"SHORT ENTRY: BB {entry_type} + TDI Momentum + Trend Filter. Risk Factor: {risk_factor}x"


        # --- FINAL SIGNAL EXECUTION ---
        if signal != "NO_TRADE":
            entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, signal, risk_factor)
            
            if sl > 0 and tp > 0:
                signal_strength = "HARD" if risk_factor == 2.0 else "SOFT"
                
                signal_details = {
                    "entry_price": entry, 
                    "stop_loss": sl, 
                    "take_profit": tp,
                    "risk_factor": risk_factor, 
                    "tdi_slow_ma": tdi_slow_ma,
                    "signal_strength": signal_strength,
                    "bb_width_percent": bb_width_percent,
                    "market_state": last_candle.get('market_state', 'UNKNOWN'),
                    "note": signal_reason,
                    "timestamp": pd.Timestamp.now()
                }
                
                logger.info(f"🔴 {signal} SIGNAL GENERATED - {signal_reason}")
                self._update_signal_state(signal)
                return signal, signal_details
            else:
                logger.warning(f"{signal} signal rejected: Invalid SL/TP calculation")
                signal = "NO_TRADE"
                signal_reason = "Invalid SL/TP calculation."

        
        # No trade conditions
        logger.debug(f"No trade: {signal_reason}")
        self._update_signal_state("NO_TRADE")
        return "NO_TRADE", {"reason": signal_reason}

    def _update_signal_state(self, current_signal: str):
        """Track consecutive signals to avoid overtrading"""
        if current_signal == self.last_signal and current_signal != "NO_TRADE":
            self.consecutive_signals += 1
        else:
            self.consecutive_signals = 1
            self.last_signal = current_signal
            
        # Record signal history (keep last 50 signals)
        self.signal_history.append({
            'signal': current_signal,
            'timestamp': pd.Timestamp.now(),
            'consecutive': self.consecutive_signals
        })
        self.signal_history = self.signal_history[-50:]  # Keep only last 50

    def get_strategy_stats(self) -> Dict:
        """Get current strategy statistics"""
        return {
            "last_signal": self.last_signal,
            "consecutive_signals": self.consecutive_signals,
            "total_signals_generated": len([s for s in self.signal_history if s['signal'] != 'NO_TRADE']),
            "min_bb_width": MIN_BB_WIDTH_PERCENT,
            "max_bb_width": MAX_BB_WIDTH_PERCENT,
            "rrr_ratio": RRR_RATIO,
            "min_risk_percent": MIN_RISK_PERCENT * 100,
            "max_risk_percent": MAX_RISK_PERCENT * 100,
            "min_klines_required": self.MIN_KLINES_REQUIRED
        }

    def get_current_market_state(self, df: pd.DataFrame) -> Dict:
        """Get current market state analysis for debugging"""
        if df.empty or len(df) < 2:
            return {"error": "Insufficient data"}
            
        last_candle = df.iloc[-1]
        
        return {
            "price": last_candle['close'],
            "tdi_slow_ma": last_candle.get('tdi_slow_ma', 'N/A'),
            "tdi_fast_ma": last_candle.get('tdi_fast_ma', 'N/A'),
            "bb_width_percent": last_candle.get('bb_width_percent', 'N/A'),
            "market_state": last_candle.get('market_state', 'UNKNOWN'),
            "tdi_zone": last_candle.get('tdi_zone', 'UNKNOWN'),
            "bb_rejection_buy": last_candle.get('bb_rejection_buy', False),
            "bb_rejection_sell": last_candle.get('bb_rejection_sell', False)
        }

# --- Utility function for quick testing ---
def test_strategy_signal(df: pd.DataFrame) -> Dict:
    """
    Test function to quickly check strategy signal with given data.
    """
    strategy = ConsolidatedTrendStrategy()
    
    # Analyze data
    analyzed_df = strategy.analyze_data(df)
    
    if analyzed_df.empty:
        return {"error": "Data analysis failed"}
    
    # Generate signal
    signal, details = strategy.generate_signal(analyzed_df)
    
    # Get market state
    market_state = strategy.get_current_market_state(analyzed_df)
    
    return {
        "signal": signal,
        "signal_details": details,
        "market_state": market_state,
        "strategy_stats": strategy.get_strategy_stats()
    }
