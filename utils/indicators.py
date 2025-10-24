# strategy/indicators.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from settings import Config

logger = logging.getLogger(__name__)

class Indicators:

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = Config.TDI_RSI_PERIOD) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI) using the standard formula.
        Enhanced with better NaN handling and validation.
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for RSI calculation. Need {period} periods, got {len(df)}")
                df['rsi'] = np.nan
                return df

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Use exponential moving average for smoother RSI
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
            logger.warning(f"Not enough data for SMA {period}. Need {period} periods, got {len(df)}")
            ma_col_name = f'{column}_sma_{period}'
            df[ma_col_name] = np.nan
            return df, ma_col_name

        ma_col_name = f'{column}_sma_{period}'
        df[ma_col_name] = df[column].rolling(window=period, min_periods=period).mean()
        return df, ma_col_name

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str, period: int) -> Tuple[pd.DataFrame, str]:
        """Calculates Exponential Moving Average for smoother trends."""
        if len(df) < period:
            logger.warning(f"Not enough data for EMA {period}. Need {period} periods, got {len(df)}")
            ema_col_name = f'{column}_ema_{period}'
            df[ema_col_name] = np.nan
            return df, ema_col_name

        ema_col_name = f'{column}_ema_{period}'
        df[ema_col_name] = df[column].ewm(span=period, min_periods=period).mean()
        return df, ema_col_name

    @staticmethod
    def calculate_super_tdi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Super Traders Dynamic Index (TDI) components according to the strategy.
        Implements the exact logic from the Pine Script:
        - RSI with configurable period
        - Bollinger Bands on RSI
        - Fast and Slow MAs on RSI
        """
        try:
            df = df.copy()
            
            # 1. Calculate RSI (Primary momentum indicator)
            df = Indicators.calculate_rsi(df, period=Config.TDI_RSI_PERIOD)
            
            if df['rsi'].isna().all():
                logger.error("RSI calculation failed - all NaN values")
                return df

            # 2. Calculate Bollinger Bands on RSI (as in Pine Script)
            bb_length = Config.TDI_BB_LENGTH if hasattr(Config, 'TDI_BB_LENGTH') else 20
            
            # Middle band (SMA of RSI)
            df, middle_bb_col = Indicators.calculate_sma(df, 'rsi', bb_length)
            df['tdi_middle_bb'] = df[middle_bb_col]
            
            # Standard deviation for BB
            df['tdi_std'] = df['rsi'].rolling(window=bb_length, min_periods=bb_length).std()
            
            # Upper and Lower bands (2.0 * std as in Pine Script)
            bb_dev = Config.TDI_BB_DEV if hasattr(Config, 'TDI_BB_DEV') else 2.0
            df['tdi_bb_upper'] = df['tdi_middle_bb'] + (df['tdi_std'] * bb_dev)
            df['tdi_bb_lower'] = df['tdi_middle_bb'] - (df['tdi_std'] * bb_dev)

            # 3. Calculate Fast MA (Bulls MA) - SMA of RSI with length 1 (from Pine Script)
            fast_ma_period = Config.TDI_FAST_MA_PERIOD if hasattr(Config, 'TDI_FAST_MA_PERIOD') else 1
            df, fast_ma_col = Indicators.calculate_sma(df, 'rsi', fast_ma_period)
            df['tdi_fast_ma'] = df[fast_ma_col]  # This is the Green line (Bulls)

            # 4. Calculate Slow MA (Bears MA) - SMA of RSI with length 5 (from Pine Script)
            slow_ma_period = Config.TDI_SLOW_MA_PERIOD if hasattr(Config, 'TDI_SLOW_MA_PERIOD') else 5
            df, slow_ma_col = Indicators.calculate_sma(df, 'rsi', slow_ma_period)
            df['tdi_slow_ma'] = df[slow_ma_col]  # This is the Red line (Bears)

            # 5. Calculate additional TDI metrics for strategy
            df = Indicators._calculate_tdi_metrics(df)

            # Clean up temporary columns
            temp_cols = [col for col in df.columns if '_sma_' in col or col in ['tdi_std']]
            df = df.drop(columns=temp_cols, errors='ignore')
            
            logger.debug("Super TDI calculation completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in calculate_super_tdi: {str(e)}")
            return df

    @staticmethod
    def _calculate_tdi_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional TDI-based metrics for signal generation."""
        
        # TDI Crossover signals
        df['tdi_bullish_cross'] = (df['tdi_fast_ma'] > df['tdi_slow_ma']) & \
                                 (df['tdi_fast_ma'].shift(1) <= df['tdi_slow_ma'].shift(1))
        df['tdi_bearish_cross'] = (df['tdi_fast_ma'] < df['tdi_slow_ma']) & \
                                 (df['tdi_fast_ma'].shift(1) >= df['tdi_slow_ma'].shift(1))
        
        # TDI Zone detection
        df['tdi_zone'] = 'NO_TRADE'
        df.loc[df['rsi'] <= Config.TDI_HARD_BUY_LEVEL, 'tdi_zone'] = 'HARD_BUY'
        df.loc[(df['rsi'] > Config.TDI_HARD_BUY_LEVEL) & (df['rsi'] <= Config.TDI_SOFT_BUY_LEVEL), 'tdi_zone'] = 'SOFT_BUY'
        df.loc[(df['rsi'] > Config.TDI_SOFT_BUY_LEVEL) & (df['rsi'] < 50), 'tdi_zone'] = 'BUY_ZONE'
        df.loc[(df['rsi'] >= 50) & (df['rsi'] < Config.TDI_SOFT_SELL_LEVEL), 'tdi_zone'] = 'SELL_ZONE'
        df.loc[(df['rsi'] >= Config.TDI_SOFT_SELL_LEVEL) & (df['rsi'] < Config.TDI_HARD_SELL_LEVEL), 'tdi_zone'] = 'SOFT_SELL'
        df.loc[df['rsi'] >= Config.TDI_HARD_SELL_LEVEL, 'tdi_zone'] = 'HARD_SELL'
        
        # TDI Strength (distance from middle)
        df['tdi_strength'] = (df['rsi'] - 50) / 50
        
        return df

    @staticmethod
    def calculate_super_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Enhanced Super Bollinger Bands with ATR-based trend detection.
        Implements the logic from the Pine Script including:
        - Standard BB with configurable period and multiplier
        - SMMA trend line
        - ATR-based trend detection
        """
        try:
            df = df.copy()
            bb_period = Config.BB_PERIOD
            bb_multiplier = Config.BB_DEV
            
            if len(df) < bb_period:
                logger.warning(f"Not enough data for BB calculation. Need {bb_period} periods, got {len(df)}")
                return df

            # 1. Standard Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=bb_period, min_periods=bb_period).mean()
            df['bb_std'] = df['close'].rolling(window=bb_period, min_periods=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_multiplier)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_multiplier)

            # 2. SMMA Trend Line (as in Pine Script)
            df = Indicators._calculate_smma(df, period=Config.BB_TREND_PERIOD if hasattr(Config, 'BB_TREND_PERIOD') else 9)

            # 3. ATR-based Trend Detection (as in Pine Script)
            df = Indicators._calculate_atr_trend(df)

            # 4. BB Position and Signals
            df = Indicators._calculate_bb_metrics(df)

            # Clean up temporary columns
            temp_cols = ['bb_std', 'atr', 'up_band', 'dn_band', 'up1', 'dn1']
            df = df.drop(columns=[col for col in temp_cols if col in df.columns], errors='ignore')
            
            logger.debug("Super Bollinger Bands calculation completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in calculate_super_bollinger_bands: {str(e)}")
            return df

    @staticmethod
    def _calculate_smma(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        """Calculate Smoothed Moving Average (SMMA) as in Pine Script."""
        # SMMA formula from Pine Script: na(smma[1]) ? ta.ema(src, len) : (smma[1] * (len - 1) + src) / len
        df['smma'] = df['close'].ewm(alpha=1/period, adjust=False).mean()  # Using EMA as initial approximation
        return df

    @staticmethod
    def _calculate_atr_trend(df: pd.DataFrame, atr_period: int = 1, atr_multiplier: float = 0.9) -> pd.DataFrame:
        """Calculate ATR-based trend detection as in Pine Script."""
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Calculate up/dn bands
        df['up_band'] = df['close'] - atr_multiplier * df['atr']
        df['dn_band'] = df['close'] + atr_multiplier * df['atr']
        
        # Initialize trend
        df['bb_trend'] = 1
        
        # Calculate trend according to Pine Script logic
        for i in range(1, len(df)):
            prev_trend = df['bb_trend'].iloc[i-1]
            close_prev = df['close'].iloc[i-1]
            up_prev = df['up_band'].iloc[i-1] if not np.isnan(df['up_band'].iloc[i-1]) else df['up_band'].iloc[i]
            dn_prev = df['dn_band'].iloc[i-1] if not np.isnan(df['dn_band'].iloc[i-1]) else df['dn_band'].iloc[i]
            close_curr = df['close'].iloc[i]
            
            if prev_trend == -1 and close_curr > dn_prev:
                df.loc[df.index[i], 'bb_trend'] = 1
            elif prev_trend == 1 and close_curr < up_prev:
                df.loc[df.index[i], 'bb_trend'] = -1
            else:
                df.loc[df.index[i], 'bb_trend'] = prev_trend
        
        # Buy/Sell signals based on trend changes
        df['bb_buy_signal'] = (df['bb_trend'] == 1) & (df['bb_trend'].shift(1) == -1)
        df['bb_sell_signal'] = (df['bb_trend'] == -1) & (df['bb_trend'].shift(1) == 1)
        
        return df

    @staticmethod
    def _calculate_bb_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional Bollinger Band metrics for strategy."""
        
        # BB Position (0-1 scale where 0=lower band, 1=upper band)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].clip(0, 1)  # Ensure within bounds
        
        # Band width and squeeze indicators
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_pct'] = df['bb_width'] / df['bb_middle']
        
        # Price relative to bands
        df['bb_upper_dist_pct'] = (df['bb_upper'] - df['close']) / df['close']
        df['bb_lower_dist_pct'] = (df['close'] - df['bb_lower']) / df['close']
        
        # Band touch signals
        df['bb_touch_upper'] = (df['high'] >= df['bb_upper']) & (df['high'].shift(1) < df['bb_upper'].shift(1))
        df['bb_touch_lower'] = (df['low'] <= df['bb_lower']) & (df['low'].shift(1) > df['bb_lower'].shift(1))
        
        # Rejection signals (price moves outside then back inside)
        df['bb_rejection_buy'] = (df['low'].shift(1) <= df['bb_lower'].shift(1)) & \
                                (df['close'] > df['bb_lower']) & \
                                (df['close'] > df['close'].shift(1))
        
        df['bb_rejection_sell'] = (df['high'].shift(1) >= df['bb_upper'].shift(1)) & \
                                 (df['close'] < df['bb_upper']) & \
                                 (df['close'] < df['close'].shift(1))
        
        return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators required for the Consolidated Trend Strategy.
        Returns DataFrame with all indicator columns.
        """
        logger.info("Calculating all indicators for Consolidated Trend Strategy...")
        
        # Calculate Super TDI first
        df = Indicators.calculate_super_tdi(df)
        
        # Calculate Super Bollinger Bands
        df = Indicators.calculate_super_bollinger_bands(df)
        
        # Calculate additional combined metrics
        df = Indicators._calculate_combined_metrics(df)
        
        logger.info(f"Indicator calculation complete. DataFrame shape: {df.shape}")
        return df

    @staticmethod
    def _calculate_combined_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics that combine TDI and BB for signal generation."""
        
        # Combined signal strength
        if all(col in df.columns for col in ['tdi_strength', 'bb_position']):
            df['combined_strength'] = df['tdi_strength'] * (1 - abs(df['bb_position'] - 0.5) * 2)
        
        # Market condition classification
        conditions = [
            (df['bb_width_pct'] < 0.003),  # Low volatility
            (df['bb_width_pct'] > 0.02),   # High volatility
            (abs(df['tdi_strength']) > 0.3),  # Strong trend
            (abs(df['tdi_strength']) < 0.1)   # Weak trend
        ]
        choices = ['LOW_VOL', 'HIGH_VOL', 'TRENDING', 'RANGING']
        df['market_condition'] = np.select(conditions, choices, default='NORMAL')
        
        return df

    @staticmethod
    def validate_indicators(df: pd.DataFrame) -> bool:
        """
        Validate that all required indicators are present and contain valid data.
        """
        required_columns = [
            'rsi', 'tdi_fast_ma', 'tdi_slow_ma', 'tdi_bb_upper', 'tdi_bb_lower',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width_pct'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required indicator columns: {missing_columns}")
            return False
        
        # Check for sufficient non-NaN values
        valid_data_ratio = df[required_columns].notna().mean().min()
        if valid_data_ratio < 0.8:
            logger.warning(f"Low valid data ratio in indicators: {valid_data_ratio:.2%}")
            return False
            
        return True