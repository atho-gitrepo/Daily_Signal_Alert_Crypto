# strategy/indicators.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
# Assuming settings.py has been configured with new settings
from settings import Config 

logger = logging.getLogger(__name__)

class Indicators:

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = Config.TDI_RSI_PERIOD) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI) with enhanced error handling.
        """
        try:
            if len(df) < period:
                logger.warning(f"Not enough data for RSI calculation. Need {period} periods, got {len(df)}")
                df['rsi'] = np.nan
                return df

            # Ensure we have valid price data
            if df['close'].isna().any() or (df['close'] <= 0).any():
                logger.warning("Invalid price data detected in RSI calculation")
                # Replace invalid values
                df['close'] = df['close'].replace(0, 0.0001)
                df['close'] = np.where(df['close'] <= 0, 0.0001, df['close'])
                df['close'] = df['close'].ffill()

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Use exponential moving average for smoother RSI (Wilder's smoothing approximation)
            avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

            # Avoid division by zero with robust handling
            avg_loss_replaced = avg_loss.replace(0, np.finfo(float).eps)
            rs = avg_gain / avg_loss_replaced
            rsi = 100 - (100 / (1 + rs))
            
            df = df.copy()
            df['rsi'] = rsi
            
            # Validate RSI output
            if df['rsi'].isna().all():
                logger.error("RSI calculation resulted in all NaN values")
            elif (df['rsi'] == 0).all():
                logger.error("RSI calculation resulted in all zero values")
            else:
                logger.debug(f"RSI calculated successfully. Range: {df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}", exc_info=True)
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
    def calculate_ema(df: pd.DataFrame, column: str, period: int) -> Tuple[pd.DataFrame, str]:
        """Calculates Exponential Moving Average (EMA) with validation."""
        if len(df) < period:
            ma_col_name = f'{column}_ema_{period}'
            df[ma_col_name] = np.nan
            return df, ma_col_name

        ma_col_name = f'{column}_ema_{period}'
        # Using EWM for EMA calculation
        df[ma_col_name] = df[column].ewm(span=period, adjust=False, min_periods=period).mean()
        return df, ma_col_name
        
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smoothing: int = 3) -> pd.DataFrame:
        """
        Calculates Stochastic %K and %D based on price (used for confirmation).
        """
        try:
            df = df.copy()
            if len(df) < k_period or len(df) < d_period or len(df) < smoothing:
                logger.warning(f"Insufficient data for Stochastic calculation. Needed at least {max(k_period, d_period, smoothing)}.")
                df['stoch_k'] = np.nan
                df['stoch_d'] = np.nan
                return df
                
            # High/Low over the K-period
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            # Calculate %K
            # Handle potential division by zero (high_max - low_min)
            range_diff = (high_max - low_min).replace(0, np.finfo(float).eps)
            df['stoch_k_raw'] = 100 * ((df['close'] - low_min) / range_diff)
            
            # Smooth %K
            df['stoch_k'] = df['stoch_k_raw'].rolling(window=smoothing).mean()
            
            # Calculate %D (SMA of %K)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}", exc_info=True)
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan
            return df

    @staticmethod
    def calculate_super_tdi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the Super Traders Dynamic Index (TDI) components.
        FIXED VERSION with comprehensive error handling and logging.
        """
        try:
            df = df.copy()
            
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for TDI: {missing_columns}")
                # Initialize TDI columns with NaN to prevent strategy errors
                tdi_columns = ['rsi', 'tdi_middle_bb', 'tdi_bb_upper', 'tdi_bb_lower', 
                              'tdi_fast_ma', 'tdi_slow_ma', 'tdi_zone', 'tdi_strength']
                for col in tdi_columns:
                    df[col] = np.nan
                return df
            
            # Clean price data
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan).ffill()
                    df[col] = np.where(df[col] <= 0, np.nan, df[col])
                    df[col] = df[col].ffill()
            
            # Check if we have any valid data left
            if df['close'].isna().all():
                logger.error("No valid price data after cleaning")
                return df

            # 1. Calculate RSI
            rsi_period = getattr(Config, 'TDI_RSI_PERIOD', 14)
            logger.info(f"Calculating TDI RSI with period {rsi_period}, data length: {len(df)}")
            
            df = Indicators.calculate_rsi(df, period=rsi_period)
            
            # Check if RSI calculation was successful
            if df['rsi'].isna().all() or (df['rsi'] == 0).all():
                logger.error("RSI calculation failed - cannot proceed with TDI")
                # Initialize required TDI columns with NaN
                tdi_cols = ['tdi_middle_bb', 'tdi_bb_upper', 'tdi_bb_lower', 'tdi_fast_ma', 'tdi_slow_ma']
                for col in tdi_cols:
                    df[col] = np.nan
                return df

            logger.info(f"RSI calculated successfully. Sample values: {df['rsi'].dropna().head().tolist()}")

            # 2. Calculate Bollinger Bands on RSI
            bb_length = getattr(Config, 'TDI_BB_LENGTH', 34)
            bb_dev = getattr(Config, 'TDI_BB_DEV', 2.0)
            
            if len(df) < bb_length:
                logger.warning(f"Insufficient data for TDI BB. Need {bb_length}, got {len(df)}")
                df['tdi_middle_bb'] = np.nan
                df['tdi_bb_upper'] = np.nan
                df['tdi_bb_lower'] = np.nan
            else:
                df, middle_bb_col = Indicators.calculate_sma(df, 'rsi', bb_length)
                df['tdi_middle_bb'] = df[middle_bb_col]
                
                df['tdi_std'] = df['rsi'].rolling(window=bb_length, min_periods=bb_length).std()
                
                # Calculate BB bands with safety checks
                df['tdi_bb_upper'] = df['tdi_middle_bb'] + (df['tdi_std'] * bb_dev)
                df['tdi_bb_lower'] = df['tdi_middle_bb'] - (df['tdi_std'] * bb_dev)

            # 3. Calculate Fast MA (Green Line / Bulls MA) - SMA(RSI, 1)
            fast_ma_period = getattr(Config, 'TDI_FAST_MA_PERIOD', 1)
            df, fast_ma_col = Indicators.calculate_sma(df, 'rsi', fast_ma_period)
            df['tdi_fast_ma'] = df[fast_ma_col] 

            # 4. Calculate Slow MA (Red Line / Bears MA) - SMA(RSI, 5)
            slow_ma_period = getattr(Config, 'TDI_SLOW_MA_PERIOD', 5)
            df, slow_ma_col = Indicators.calculate_sma(df, 'rsi', slow_ma_period)
            df['tdi_slow_ma'] = df[slow_ma_col] 

            # CRITICAL FIX: Replace zeros and handle NaN values properly
            df['tdi_slow_ma'] = df['tdi_slow_ma'].replace(0, np.nan).ffill()
            df['tdi_fast_ma'] = df['tdi_fast_ma'].replace(0, np.nan).ffill()
            
            # Only calculate additional metrics if we have valid TDI data
            valid_tdi_data = not df['tdi_slow_ma'].isna().all() and not df['tdi_fast_ma'].isna().all()
            
            if valid_tdi_data:
                df = Indicators._calculate_tdi_metrics(df)
                logger.info(f"TDI calculation completed. Slow MA range: {df['tdi_slow_ma'].min():.2f} - {df['tdi_slow_ma'].max():.2f}")
            else:
                logger.warning("Insufficient valid TDI data for metric calculation")
                # Initialize missing metrics
                df['tdi_zone'] = 'NO_TRADE'
                df['tdi_strength'] = 0.0

            # Clean up temporary columns
            temp_cols = [col for col in df.columns if '_sma_' in col or col in ['tdi_std']]
            df = df.drop(columns=temp_cols, errors='ignore')
            
            return df

        except Exception as e:
            logger.error(f"Critical error in calculate_super_tdi: {str(e)}", exc_info=True)
            # Ensure all required columns are present even on error
            required_tdi_cols = ['rsi', 'tdi_middle_bb', 'tdi_bb_upper', 'tdi_bb_lower', 
                               'tdi_fast_ma', 'tdi_slow_ma', 'tdi_zone', 'tdi_strength']
            for col in required_tdi_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df['tdi_zone'] = df.get('tdi_zone', 'NO_TRADE')
            return df

    @staticmethod
    def _calculate_tdi_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional TDI-based metrics for signal generation."""
        try:
            # TDI Crossover signals (Used for logging/debug)
            df['tdi_bullish_cross'] = (df['tdi_fast_ma'] > df['tdi_slow_ma']) & \
                                     (df['tdi_fast_ma'].shift(1) <= df['tdi_slow_ma'].shift(1))
            df['tdi_bearish_cross'] = (df['tdi_fast_ma'] < df['tdi_slow_ma']) & \
                                     (df['tdi_fast_ma'].shift(1) >= df['tdi_slow_ma'].shift(1))
            
            # TDI Zone detection (Using TDI Slow MA for zone check as it's the trend line)
            df['tdi_zone'] = 'NO_TRADE'
            
            # Safely get configuration values with defaults
            center_line = getattr(Config, 'TDI_CENTER_LINE', 50)
            hard_buy = getattr(Config, 'TDI_HARD_BUY_LEVEL', 25)
            soft_buy = getattr(Config, 'TDI_SOFT_BUY_LEVEL', 35)
            soft_sell = getattr(Config, 'TDI_SOFT_SELL_LEVEL', 65)
            hard_sell = getattr(Config, 'TDI_HARD_SELL_LEVEL', 75)
            
            # Apply zone logic with NaN safety
            valid_slow_ma = df['tdi_slow_ma'].notna()
            
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] <= hard_buy), 'tdi_zone'] = 'HARD_BUY'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] > hard_buy) & (df['tdi_slow_ma'] <= soft_buy), 'tdi_zone'] = 'SOFT_BUY'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] > soft_buy) & (df['tdi_slow_ma'] < center_line), 'tdi_zone'] = 'BUY_ZONE'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] >= center_line) & (df['tdi_slow_ma'] < soft_sell), 'tdi_zone'] = 'SELL_ZONE'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] >= soft_sell) & (df['tdi_slow_ma'] < hard_sell), 'tdi_zone'] = 'SOFT_SELL'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] >= hard_sell), 'tdi_zone'] = 'HARD_SELL'
            
            # Calculate TDI strength with division safety
            df['tdi_strength'] = (df['tdi_slow_ma'] - center_line) / center_line
            
            # Replace any infinite values resulting from division
            df['tdi_strength'] = df['tdi_strength'].replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in _calculate_tdi_metrics: {str(e)}")
            # Ensure basic columns exist
            if 'tdi_zone' not in df.columns:
                df['tdi_zone'] = 'NO_TRADE'
            if 'tdi_strength' not in df.columns:
                df['tdi_strength'] = 0.0
            return df

    @staticmethod
    def calculate_super_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Enhanced Super Bollinger Bands.
        FIXED VERSION with comprehensive error handling.
        """
        try:
            df = df.copy()
            bb_period = getattr(Config, 'BB_PERIOD', 20)
            bb_multiplier = getattr(Config, 'BB_DEV', 2.0)
            
            if len(df) < bb_period:
                logger.warning(f"Not enough data for BB calculation. Need {bb_period} periods, got {len(df)}")
                # Initialize BB columns with NaN
                bb_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_percent', 
                          'bb_rejection_buy', 'bb_rejection_sell', 'atr']
                for col in bb_cols:
                    df[col] = np.nan
                return df

            # 1. Standard Bollinger Bands (The primary bands for rejection)
            df['bb_middle'] = df['close'].rolling(window=bb_period, min_periods=bb_period).mean()
            df['bb_std'] = df['close'].rolling(window=bb_period, min_periods=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_multiplier)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_multiplier)

            # 2. SMMA Trend Line (approximation)
            trend_period = getattr(Config, 'BB_TREND_PERIOD', 9)
            df['smma'] = df['close'].ewm(alpha=1/trend_period, adjust=False).mean() 

            # 3. ATR-based Trend Detection
            df = Indicators._calculate_atr_trend(df)

            # 4. BB Rejection Metrics (CRITICAL for strategy)
            df = Indicators._calculate_bb_metrics(df)

            # Clean up temporary columns
            temp_cols = ['bb_std']
            df = df.drop(columns=[col for col in temp_cols if col in df.columns], errors='ignore')
            
            logger.debug("Bollinger Bands calculated successfully")
            
            return df

        except Exception as e:
            logger.error(f"Error in calculate_super_bollinger_bands: {str(e)}", exc_info=True)
            # Ensure required columns exist even on error
            required_bb_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_percent', 
                               'bb_rejection_buy', 'bb_rejection_sell', 'atr']
            for col in required_bb_cols:
                if col not in df.columns:
                    df[col] = np.nan
            return df

    @staticmethod
    def _calculate_atr_trend(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Calculate ATR with proper error handling."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            df['atr'] = true_range.rolling(window=atr_period).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error in _calculate_atr_trend: {str(e)}")
            df['atr'] = np.nan
            return df

    @staticmethod
    def _calculate_bb_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate BB Width and Rejection signals with error handling."""
        try:
            # BB Width Percentage (for volatility filter)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # Safe division for width percentage
            df['bb_width_percent'] = df['bb_width'] / df['bb_middle'].replace(0, np.nan)
            df['bb_width_percent'] = df['bb_width_percent'].fillna(0)
            
            # BB Rejection Signals with NaN safety
            df['bb_rejection_buy'] = False
            df['bb_rejection_sell'] = False
            
            # Only calculate if we have sufficient data
            if len(df) > 1:
                df['bb_rejection_buy'] = (
                    (df['low'].shift(1) <= df['bb_lower'].shift(1)) & 
                    (df['close'] > df['bb_lower']) & 
                    (df['close'] > df['close'].shift(1))
                )
                
                df['bb_rejection_sell'] = (
                    (df['high'].shift(1) >= df['bb_upper'].shift(1)) & 
                    (df['close'] < df['bb_upper']) & 
                    (df['close'] < df['close'].shift(1))
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error in _calculate_bb_metrics: {str(e)}")
            df['bb_width'] = np.nan
            df['bb_width_percent'] = np.nan
            df['bb_rejection_buy'] = False
            df['bb_rejection_sell'] = False
            return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators required for the Combined Trading Strategy.
        Modified to include 200 EMA and Stochastic.
        """
        try:
            df = df.copy()
            df.columns = [col.lower() for col in df.columns]

            logger.info("Starting indicator calculation...")
            
            # Calculate TDI indicators
            df = Indicators.calculate_super_tdi(df)
            
            # Calculate Bollinger Bands
            df = Indicators.calculate_super_bollinger_bands(df)
            
            # Calculate 200 EMA (for Trend Filter)
            df, ema_col = Indicators.calculate_ema(df, 'close', 200)
            df['ema_200'] = df[ema_col]
            
            # Calculate Stochastic (for additional momentum check/confirmation)
            df = Indicators.calculate_stochastic(df, 
                                                k_period=getattr(Config, 'STOCH_PERIOD', 14), 
                                                d_period=getattr(Config, 'STOCH_LENGTH', 3), 
                                                smoothing=getattr(Config, 'STOCH_SMOOTHING', 3))
            
            # Combine metrics for filters in strategy file
            df['tdi_trend'] = df['tdi_fast_ma'] - df['tdi_slow_ma']
            
            # Safe division for TDI strength
            valid_slow_ma = df['tdi_slow_ma'].replace(0, np.nan)
            df['tdi_strength'] = abs(df['tdi_trend']) / valid_slow_ma
            df['tdi_strength'] = df['tdi_strength'].fillna(0)
            
            # CRITICAL: Drop any rows where essential indicators are completely NaN
            essential_cols = ['tdi_slow_ma', 'tdi_fast_ma', 'bb_width_percent', 'ema_200', 'stoch_k']
            before_drop = len(df)
            # Use 'any' drop to be safer, but only on essential columns
            df.dropna(subset=essential_cols, how='any', inplace=True)
            after_drop = len(df)
            
            if before_drop != after_drop:
                logger.info(f"Dropped {before_drop - after_drop} rows with missing essential indicators")
            
            logger.info(f"Indicator calculation completed. Final data shape: {df.shape}")
            
            return df

        except Exception as e:
            logger.error(f"Critical error in calculate_all_indicators: {str(e)}", exc_info=True)
            # Return the original dataframe with minimal indicator initialization
            required_cols = ['tdi_slow_ma', 'tdi_fast_ma', 'tdi_zone', 'tdi_strength', 
                           'bb_width_percent', 'bb_rejection_buy', 'bb_rejection_sell', 
                           'ema_200', 'stoch_k', 'stoch_d']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df['tdi_zone'] = df.get('tdi_zone', 'NO_TRADE')
            return df

    @staticmethod
    def validate_indicators(df: pd.DataFrame) -> bool:
        """
        Validate that all required indicators are properly calculated.
        """
        try:
            required_indicators = [
                'tdi_slow_ma', 'tdi_fast_ma', 'tdi_zone', 'tdi_strength',
                'bb_width_percent', 'bb_rejection_buy', 'bb_rejection_sell', 'atr',
                'ema_200', 'stoch_k', 'stoch_d' # Added new indicators
            ]
            
            for indicator in required_indicators:
                if indicator not in df.columns:
                    logger.error(f"Missing required indicator: {indicator}")
                    return False
                
                # Check if indicator has valid values (not all NaN)
                if df[indicator].isna().all():
                    logger.error(f"Indicator {indicator} has all NaN values")
                    return False
            
            # Check if we have at least some valid TDI values
            if df['tdi_slow_ma'].isna().all() or df['tdi_fast_ma'].isna().all():
                logger.error("TDI indicators have no valid values")
                return False
                
            logger.info("All indicators validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating indicators: {str(e)}")
            return False
