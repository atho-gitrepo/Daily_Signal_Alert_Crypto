# strategy/consolidated_trend.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
# Assuming 'Indicators' and 'Config' are properly defined in your project
from utils.indicators import Indicators
from settings import Config 

logger = logging.getLogger(__name__)

# --- Configuration ---
RRR_RATIO = 1.5
# Minimum BB Width as a percentage of the price (e.g., 0.003 means 0.3% min width)
MIN_BB_WIDTH_PERCENT = 0.003 
# Maximum BB Width to avoid extreme volatility
MAX_BB_WIDTH_PERCENT = 0.03

# --- Strategy Class ---
class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy (Volatility Filter) initialized.")
        # Ensure minimum data required for all indicators is known
        self.MIN_KLINES_REQUIRED = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 10
        
        # Strategy state tracking
        self.last_signal = "NO_TRADE"
        self.consecutive_signals = 0

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all indicators and prepares data for signal generation.
        Calculates BB Width for the volatility filter.
        
        Enhanced with additional trend confirmation and market state detection.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to analyze_data")
            return df

        # Make a copy to avoid modifying original data
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        try:
            # Calculate all necessary indicators
            df = Indicators.calculate_super_tdi(df)
            df = Indicators.calculate_super_bollinger_bands(df)

            # 🎯 Calculate BB Width Percentage for volatility filter
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                df['bb_width_percent'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                # Add BB position relative to price
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            else:
                logger.error("Bollinger Band columns missing after calculation!")
                return pd.DataFrame()
            
            # Additional trend strength indicators
            df = self._calculate_trend_strength(df)
            df = self._calculate_market_state(df)
            
            df.dropna(inplace=True)
            
            if df.empty:
                logger.warning("All data removed after dropna()")

        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}")
            return pd.DataFrame()

        return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional trend strength metrics"""
        # TDI trend strength
        df['tdi_trend'] = df['tdi_fast_ma'] - df['tdi_slow_ma']
        df['tdi_strength'] = abs(df['tdi_trend']) / df['tdi_slow_ma']
        
        # Price momentum relative to BB
        if 'bb_middle' in df.columns:
            df['price_vs_bb'] = (df['close'] - df['bb_middle']) / df['bb_middle']
        
        # Volume confirmation (if available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df

    def _calculate_market_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market state (trending, ranging, volatile)"""
        # Price volatility
        df['price_range'] = (df['high'] - df['low']) / df['low']
        df['range_ma'] = df['price_range'].rolling(window=20).mean()
        df['volatility_ratio'] = df['price_range'] / df['range_ma']
        
        # Market state classification
        conditions = [
            (df['bb_width_percent'] < MIN_BB_WIDTH_PERCENT),  # Consolidation
            (df['bb_width_percent'] > MAX_BB_WIDTH_PERCENT),  # High Volatility
            (df['tdi_strength'] > df['tdi_strength'].rolling(20).mean()),  # Strong Trend
        ]
        choices = ['CONSOLIDATION', 'HIGH_VOL', 'TRENDING']
        df['market_state'] = np.select(conditions, choices, default='NORMAL')
        
        return df

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """
        Enhanced SL/TP calculation with multiple confirmation methods.
        """
        entry_price = last_candle['close']
        
        # Dynamic lookback based on market volatility
        volatility_lookback = min(20, max(5, int(last_candle.get('volatility_ratio', 1) * 10)))
        lookback_period = volatility_lookback
        
        start_index = max(0, len(df) - lookback_period)
        lookback_df = df.iloc[start_index:-1]

        if signal_type == "BUY":
            # Method 1: Recent swing low
            swing_low_sl = lookback_df['low'].min() * 0.998
            
            # Method 2: BB-based SL
            bb_sl = last_candle.get('bb_lower', entry_price * 0.99) * 0.995
            
            # Use the more conservative (higher) SL
            stop_loss = max(swing_low_sl, bb_sl)
            
            # Minimum risk check (0.15% minimum)
            min_risk_distance = entry_price * 0.0015
            if entry_price - stop_loss < min_risk_distance:
                stop_loss = entry_price - min_risk_distance
                
            # Maximum risk check (2% maximum)  
            max_risk_distance = entry_price * 0.02
            if entry_price - stop_loss > max_risk_distance:
                stop_loss = entry_price - max_risk_distance
                
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * RRR_RATIO * risk_factor)

        elif signal_type == "SELL":
            # Method 1: Recent swing high
            swing_high_sl = lookback_df['high'].max() * 1.002
            
            # Method 2: BB-based SL
            bb_sl = last_candle.get('bb_upper', entry_price * 1.01) * 1.005
            
            # Use the more conservative (lower) SL
            stop_loss = min(swing_high_sl, bb_sl)
            
            # Minimum risk check (0.15% minimum)
            min_risk_distance = entry_price * 0.0015
            if stop_loss - entry_price < min_risk_distance:
                stop_loss = entry_price + min_risk_distance
                
            # Maximum risk check (2% maximum)
            max_risk_distance = entry_price * 0.02
            if stop_loss - entry_price > max_risk_distance:
                stop_loss = entry_price + max_risk_distance
                
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * RRR_RATIO * risk_factor)

        else:
            return entry_price, 0, 0

        logger.debug(f"SL/TP Calculation: Entry={entry_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}, Risk={abs(entry_price-stop_loss)/entry_price*100:.2f}%")
        return entry_price, stop_loss, take_profit

    def _check_volatility_filter(self, bb_width_percent: float) -> bool:
        """
        Check if volatility conditions are suitable for trading.
        """
        if bb_width_percent < MIN_BB_WIDTH_PERCENT:
            logger.debug(f"Volatility too low: {bb_width_percent:.4f} < {MIN_BB_WIDTH_PERCENT}")
            return False
            
        if bb_width_percent > MAX_BB_WIDTH_PERCENT:
            logger.debug(f"Volatility too high: {bb_width_percent:.4f} > {MAX_BB_WIDTH_PERCENT}")
            return False
            
        return True

    def _check_tdi_zone(self, rsi: float, signal_type: str) -> Tuple[bool, float]:
        """
        Check TDI zone and determine risk factor.
        """
        if signal_type == "BUY":
            if rsi <= Config.TDI_HARD_BUY_LEVEL:
                return True, 1.5  # Hard buy
            elif rsi <= Config.TDI_SOFT_BUY_LEVEL:
                return True, 1.2  # Soft buy
            elif rsi < 50:  # Buyer zone but not at key levels
                return True, 1.0
                
        elif signal_type == "SELL":
            if rsi >= Config.TDI_HARD_SELL_LEVEL:
                return True, 1.5  # Hard sell
            elif rsi >= Config.TDI_SOFT_SELL_LEVEL:
                return True, 1.2  # Soft sell
            elif rsi > 50:  # Seller zone but not at key levels
                return True, 1.0
                
        return False, 1.0

    def _check_additional_filters(self, df: pd.DataFrame, current_index: int) -> bool:
        """
        Apply additional filters to improve signal quality.
        """
        if len(df) < 5:
            return True
            
        # Avoid trading during extreme volatility
        current_volatility = df.iloc[current_index].get('volatility_ratio', 1)
        if current_volatility > 2.0:
            logger.debug("Skipping trade: Extreme volatility")
            return False
            
        # Check if market is in strong trend (good for our strategy)
        market_state = df.iloc[current_index].get('market_state', 'NORMAL')
        if market_state == 'CONSOLIDATION':
            logger.debug("Skipping trade: Market in consolidation")
            return False
            
        # Volume confirmation (if available)
        if 'volume_ratio' in df.columns:
            volume_ratio = df.iloc[current_index]['volume_ratio']
            if volume_ratio < 0.8:  # Low volume
                logger.debug("Skipping trade: Low volume")
                return False
                
        return True

    def generate_signal(self, df):
        """
        Enhanced signal generation with multiple confirmation layers.
        """
        if df.empty or len(df) < self.MIN_KLINES_REQUIRED:
            logger.warning(f"DataFrame too small for signal generation (Need >{self.MIN_KLINES_REQUIRED} rows).")
            return "NO_TRADE", {}

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Extract key indicators
        rsi = last_candle['rsi']
        fast_ma = last_candle['tdi_fast_ma']
        slow_ma = last_candle['tdi_slow_ma']
        close = last_candle['close']
        bb_lower = last_candle['bb_lower']
        bb_upper = last_candle['bb_upper']
        bb_width_percent = last_candle['bb_width_percent']
        prev_low = prev_candle['low']
        prev_high = prev_candle['high']
        prev_close = prev_candle['close']

        # --- Volatility Filter Check ---
        if not self._check_volatility_filter(bb_width_percent):
            logger.info(f"VOLATILITY FILTER: BB Width ({bb_width_percent:.4f}) outside optimal range. NO TRADE.")
            return "NO_TRADE", {"reason": f"Volatility out of range: {bb_width_percent:.4f}"}

        # --- Crossover Check ---
        bullish_crossover = (fast_ma > slow_ma) and (prev_candle['tdi_fast_ma'] <= prev_candle['tdi_slow_ma'])
        bearish_crossover = (fast_ma < slow_ma) and (prev_candle['tdi_fast_ma'] >= prev_candle['tdi_slow_ma'])

        # --- BB Rejection Check (Enhanced) ---
        bb_rejection_buy = (
            (prev_low <= prev_candle['bb_lower']) and 
            (close > prev_close) and 
            (close > bb_lower) and
            (last_candle['close'] > last_candle['open'])  # Bullish candle confirmation
        )
        
        bb_rejection_sell = (
            (prev_high >= prev_candle['bb_upper']) and 
            (close < prev_close) and 
            (close < bb_upper) and
            (last_candle['close'] < last_candle['open'])  # Bearish candle confirmation
        )

        # --- Additional Filters ---
        if not self._check_additional_filters(df, -1):
            return "NO_TRADE", {"reason": "Additional filters failed"}

        signal_details = {}

        # --- BUY Signal ---
        if bullish_crossover and bb_rejection_buy:
            zone_ok, risk_factor = self._check_tdi_zone(rsi, "BUY")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "BUY", risk_factor)
                
                signal_details = {
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "risk_factor": risk_factor,
                    "tdi_value": rsi,
                    "bb_width": bb_width_percent,
                    "market_state": last_candle.get('market_state', 'UNKNOWN'),
                    "signal_strength": "HARD" if risk_factor == 1.5 else "SOFT" if risk_factor == 1.2 else "NORMAL"
                }
                
                logger.info(f"*** BUY SIGNAL *** RSI={rsi:.2f}, Risk={risk_factor:.1f}x, SL={sl:.4f}, TP={tp:.4f}")
                self._update_signal_state("BUY")
                return "BUY", signal_details

        # --- SELL Signal ---
        if bearish_crossover and bb_rejection_sell:
            zone_ok, risk_factor = self._check_tdi_zone(rsi, "SELL")
            
            if zone_ok:
                entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "SELL", risk_factor)
                
                signal_details = {
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "risk_factor": risk_factor,
                    "tdi_value": rsi,
                    "bb_width": bb_width_percent,
                    "market_state": last_candle.get('market_state', 'UNKNOWN'),
                    "signal_strength": "HARD" if risk_factor == 1.5 else "SOFT" if risk_factor == 1.2 else "NORMAL"
                }
                
                logger.info(f"*** SELL SIGNAL *** RSI={rsi:.2f}, Risk={risk_factor:.1f}x, SL={sl:.4f}, TP={tp:.4f}")
                self._update_signal_state("SELL")
                return "SELL", signal_details

        # --- NO TRADE ---
        logger.debug(f"No trade: BullishX={bullish_crossover}, BearishX={bearish_crossover}, BBRejBuy={bb_rejection_buy}, BBRejSell={bb_rejection_sell}")
        self._update_signal_state("NO_TRADE")
        return "NO_TRADE", {"reason": "No valid crossover + rejection combination"}

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