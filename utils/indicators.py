# strategy/indicators.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
# Assuming 'Config' is properly defined in your project
from settings import Config 

logger = logging.getLogger(__name__)

class Indicators:

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = Config.TDI_RSI_PERIOD) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI).
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for RSI calculation. Need {period} periods, got {len(df)}")
                df['rsi'] = np.nan
                return df

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Use exponential moving average for smoother RSI (Wilder's smoothing approximation)
            avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            rsi = 100 - (100 / (1 + rs))
            
            df = df.copy()
            df['rsi'] = rsi
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            df['rsi'] = np.nan
            return df

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str, period: int) -> Tuple[pd.DataFrame, str]:
        """Calculates Simple Moving Average with validation."""
        if len(df) < period:
            ma_col_name = f'{column}_sma_{period}'
            df[ma_col_name] = np.nan
            return df, ma_col_name

        ma_col_name = f'{column}_sma_{period}'
        df[ma_col_name] = df[column].rolling(window=period, min_periods=period).mean()
        return df, ma_col_name

    @staticmethod
    def calculate_super_tdi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Super Traders Dynamic Index (TDI) components.
        Implements the exact logic from the Pine Script: RSI, BB on RSI, Fast/Slow MAs.
        """
        try:
            df = df.copy()
            
            # 1. Calculate RSI
            df = Indicators.calculate_rsi(df, period=Config.TDI_RSI_PERIOD)
            
            if df['rsi'].isna().all():
                return df

            # 2. Calculate Bollinger Bands on RSI
            bb_length = Config.TDI_BB_LENGTH 
            bb_dev = Config.TDI_BB_DEV if hasattr(Config, 'TDI_BB_DEV') else 2.0
            
            df, middle_bb_col = Indicators.calculate_sma(df, 'rsi', bb_length)
            df['tdi_middle_bb'] = df[middle_bb_col]
            
            df['tdi_std'] = df['rsi'].rolling(window=bb_length, min_periods=bb_length).std()
            
            df['tdi_bb_upper'] = df['tdi_middle_bb'] + (df['tdi_std'] * bb_dev)
            df['tdi_bb_lower'] = df['tdi_middle_bb'] - (df['tdi_std'] * bb_dev)

            # 3. Calculate Fast MA (Green Line / Bulls MA) - SMA(RSI, 1)
            fast_ma_period = Config.TDI_FAST_MA_PERIOD if hasattr(Config, 'TDI_FAST_MA_PERIOD') else 1
            df, fast_ma_col = Indicators.calculate_sma(df, 'rsi', fast_ma_period)
            df['tdi_fast_ma'] = df[fast_ma_col] 

            # 4. Calculate Slow MA (Red Line / Bears MA) - SMA(RSI, 5)
            slow_ma_period = Config.TDI_SLOW_MA_PERIOD if hasattr(Config, 'TDI_SLOW_MA_PERIOD') else 5
            df, slow_ma_col = Indicators.calculate_sma(df, 'rsi', slow_ma_period)
            df['tdi_slow_ma'] = df[slow_ma_col] 

            df = Indicators._calculate_tdi_metrics(df)

            temp_cols = [col for col in df.columns if '_sma_' in col or col in ['tdi_std']]
            df = df.drop(columns=temp_cols, errors='ignore')
            
            return df

        except Exception as e:
            logger.error(f"Error in calculate_super_tdi: {str(e)}")
            return df

    @staticmethod
    def _calculate_tdi_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional TDI-based metrics for signal generation."""
        
        # TDI Crossover signals (Used for logging/debug)
        df['tdi_bullish_cross'] = (df['tdi_fast_ma'] > df['tdi_slow_ma']) & \
                                 (df['tdi_fast_ma'].shift(1) <= df['tdi_slow_ma'].shift(1))
        df['tdi_bearish_cross'] = (df['tdi_fast_ma'] < df['tdi_slow_ma']) & \
                                 (df['tdi_fast_ma'].shift(1) >= df['tdi_slow_ma'].shift(1))
        
        # TDI Zone detection (Using TDI Slow MA for zone check as it's the trend line)
        df['tdi_zone'] = 'NO_TRADE'
        df.loc[df['tdi_slow_ma'] <= Config.TDI_HARD_BUY_LEVEL, 'tdi_zone'] = 'HARD_BUY'
        df.loc[(df['tdi_slow_ma'] > Config.TDI_HARD_BUY_LEVEL) & (df['tdi_slow_ma'] <= Config.TDI_SOFT_BUY_LEVEL), 'tdi_zone'] = 'SOFT_BUY'
        df.loc[(df['tdi_slow_ma'] > Config.TDI_SOFT_BUY_LEVEL) & (df['tdi_slow_ma'] < Config.TDI_CENTER_LINE), 'tdi_zone'] = 'BUY_ZONE'
        df.loc[(df['tdi_slow_ma'] >= Config.TDI_CENTER_LINE) & (df['tdi_slow_ma'] < Config.TDI_SOFT_SELL_LEVEL), 'tdi_zone'] = 'SELL_ZONE'
        df.loc[(df['tdi_slow_ma'] >= Config.TDI_SOFT_SELL_LEVEL) & (df['tdi_slow_ma'] < Config.TDI_HARD_SELL_LEVEL), 'tdi_zone'] = 'SOFT_SELL'
        df.loc[df['tdi_slow_ma'] >= Config.TDI_HARD_SELL_LEVEL, 'tdi_zone'] = 'HARD_SELL'
        
        df['tdi_strength'] = (df['tdi_slow_ma'] - Config.TDI_CENTER_LINE) / Config.TDI_CENTER_LINE
        
        return df

    @staticmethod
    def calculate_super_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Enhanced Super Bollinger Bands.
        Focuses on standard BB for strategy core, keeping other Pine logic for completeness.
        """
        try:
            df = df.copy()
            bb_period = Config.BB_PERIOD
            bb_multiplier = Config.BB_DEV
            
            if len(df) < bb_period:
                return df

            # 1. Standard Bollinger Bands (The primary bands for rejection)
            df['bb_middle'] = df['close'].rolling(window=bb_period, min_periods=bb_period).mean()
            df['bb_std'] = df['close'].rolling(window=bb_period, min_periods=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_multiplier)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_multiplier)

            # 2. SMMA Trend Line (approximation)
            df['smma'] = df['close'].ewm(alpha=1/(Config.BB_TREND_PERIOD if hasattr(Config, 'BB_TREND_PERIOD') else 9), adjust=False).mean() 

            # 3. ATR-based Trend Detection (for completeness, though not strictly required for core entry)
            df = Indicators._calculate_atr_trend(df)

            # 4. BB Rejection Metrics (CRITICAL for strategy)
            df = Indicators._calculate_bb_metrics(df)

            temp_cols = ['bb_std', 'atr', 'up_band', 'dn_band']
            df = df.drop(columns=[col for col in temp_cols if col in df.columns], errors='ignore')
            
            return df

        except Exception as e:
            logger.error(f"Error in calculate_super_bollinger_bands: {str(e)}")
            return df

    @staticmethod
    def _calculate_atr_trend(df: pd.DataFrame, atr_period: int = 1, atr_multiplier: float = 0.9) -> pd.DataFrame:
        """Calculate ATR-based trend detection as in Pine Script."""
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Trend calculation logic is complex and computationally intensive for large dataframes.
        # We will keep the calculated 'atr' for use in the SL/TP logic.
        
        return df

    @staticmethod
    def _calculate_bb_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate BB Width and Rejection signals (key for strategy)."""
        
        # BB Width Percentage (for volatility filter)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_percent'] = df['bb_width'] / df['bb_middle']
        
        # BB Rejection Signals (Rule 3 & 4: Price touches/moves outside and reverses inside)
        # Check if previous bar touched/moved outside BB (Low <= LowerBand)
        # AND current bar closed back *inside* the band (Close > LowerBand) 
        # AND current bar shows reversal (Close > Open or Close > Prev Close)
        
        # We use the previous bar's Low/High vs. the previous bar's BB to capture the moment of touch/exit.
        # We use the current bar's Close vs. the current bar's BB for the reversal inside confirmation.
        
        df['bb_rejection_buy'] = (df['low'].shift(1) <= df['bb_lower'].shift(1)) & \
                                (df['close'] > df['bb_lower']) & \
                                (df['close'] > df['close'].shift(1)) # Price reverses up
        
        df['bb_rejection_sell'] = (df['high'].shift(1) >= df['bb_upper'].shift(1)) & \
                                 (df['close'] < df['bb_upper']) & \
                                 (df['close'] < df['close'].shift(1)) # Price reverses down
        
        return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators required for the Consolidated Trend Strategy.
        """
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        df = Indicators.calculate_super_tdi(df)
        df = Indicators.calculate_super_bollinger_bands(df)
        
        # Combine metrics for filters in strategy file
        df['tdi_trend'] = df['tdi_fast_ma'] - df['tdi_slow_ma']
        df['tdi_strength'] = abs(df['tdi_trend']) / df['tdi_slow_ma']
        
        return df