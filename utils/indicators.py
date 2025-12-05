# strategy/indicators.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from settings import Config 

logger = logging.getLogger(__name__)

class Indicators:

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = Config.TDI_RSI_PERIOD) -> pd.DataFrame:
        """Calculates the Relative Strength Index (RSI)."""
        try:
            if len(df) < period:
                df['rsi'] = np.nan
                return df
            df['close'] = df['close'].ffill().replace(0, np.finfo(float).eps)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
            avg_loss_replaced = avg_loss.replace(0, np.finfo(float).eps)
            rs = avg_gain / avg_loss_replaced
            rsi = 100 - (100 / (1 + rs))
            df = df.copy()
            df['rsi'] = rsi
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}", exc_info=True)
            df['rsi'] = np.nan
            return df

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str, period: int) -> Tuple[pd.DataFrame, str]:
        """Calculates Simple Moving Average with validation."""
        ma_col_name = f'{column}_sma_{period}'
        if len(df) < period:
            df[ma_col_name] = np.nan
            return df, ma_col_name
        df[ma_col_name] = df[column].rolling(window=period, min_periods=period).mean()
        return df, ma_col_name

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
        """
        Calculates the Exponential Moving Average (EMA).
        """
        ema_col_name = f'{column}_ema_{period}'
        if len(df) < period:
            df[ema_col_name] = np.nan
            return df
        
        df[ema_col_name] = df[column].ewm(span=period, adjust=False, min_periods=period).mean()
        return df

    @staticmethod
    def calculate_super_tdi(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Super Traders Dynamic Index (TDI) components."""
        try:
            df = df.copy()
            rsi_period = getattr(Config, 'TDI_RSI_PERIOD', 13)
            df = Indicators.calculate_rsi(df, period=rsi_period)
            
            if df['rsi'].isna().all():
                tdi_cols = ['tdi_middle_bb', 'tdi_bb_upper', 'tdi_bb_lower', 'tdi_fast_ma', 'tdi_slow_ma']
                for col in tdi_cols: df[col] = np.nan
                return df

            bb_length = getattr(Config, 'TDI_BB_LENGTH', 34)
            bb_dev = getattr(Config, 'TDI_BB_DEV', 2.0)
            
            if len(df) >= bb_length:
                df, middle_bb_col = Indicators.calculate_sma(df, 'rsi', bb_length)
                df['tdi_middle_bb'] = df[middle_bb_col]
                df['tdi_std'] = df['rsi'].rolling(window=bb_length, min_periods=bb_length).std()
                df['tdi_bb_upper'] = df['tdi_middle_bb'] + (df['tdi_std'] * bb_dev)
                df['tdi_bb_lower'] = df['tdi_middle_bb'] - (df['tdi_std'] * bb_dev)
            else:
                df['tdi_middle_bb'] = df['tdi_bb_upper'] = df['tdi_bb_lower'] = np.nan

            fast_ma_period = getattr(Config, 'TDI_FAST_MA_PERIOD', 1)
            df, fast_ma_col = Indicators.calculate_sma(df, 'rsi', fast_ma_period)
            df['tdi_fast_ma'] = df[fast_ma_col] 

            slow_ma_period = getattr(Config, 'TDI_SLOW_MA_PERIOD', 7)
            df, slow_ma_col = Indicators.calculate_sma(df, 'rsi', slow_ma_period)
            df['tdi_slow_ma'] = df[slow_ma_col] 

            if not df['tdi_slow_ma'].isna().all() and not df['tdi_fast_ma'].isna().all():
                df = Indicators._calculate_tdi_metrics(df)
            else:
                df['tdi_zone'] = 'NO_TRADE'
                df['tdi_strength'] = 0.0

            temp_cols = [col for col in df.columns if '_sma_' in col or col in ['tdi_std']]
            df = df.drop(columns=temp_cols, errors='ignore')
            
            return df

        except Exception as e:
            logger.error(f"Critical error in calculate_super_tdi: {str(e)}", exc_info=True)
            required_tdi_cols = ['rsi', 'tdi_middle_bb', 'tdi_bb_upper', 'tdi_bb_lower', 
                               'tdi_fast_ma', 'tdi_slow_ma', 'tdi_zone', 'tdi_strength']
            for col in required_tdi_cols:
                if col not in df.columns: df[col] = np.nan
            df['tdi_zone'] = df.get('tdi_zone', 'NO_TRADE')
            return df

    @staticmethod
    def _calculate_tdi_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional TDI-based metrics for signal generation."""
        try:
            center_line = getattr(Config, 'TDI_CENTER_LINE', 50)
            hard_buy = getattr(Config, 'TDI_HARD_BUY_LEVEL', 25)
            soft_buy = getattr(Config, 'TDI_SOFT_BUY_LEVEL', 35)
            soft_sell = getattr(Config, 'TDI_SOFT_SELL_LEVEL', 65)
            hard_sell = getattr(Config, 'TDI_HARD_SELL_LEVEL', 75)
            
            df['tdi_zone'] = 'NO_TRADE'
            valid_slow_ma = df['tdi_slow_ma'].notna()
            
            # Apply zone logic
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] <= hard_buy), 'tdi_zone'] = 'HARD_BUY'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] > hard_buy) & (df['tdi_slow_ma'] <= soft_buy), 'tdi_zone'] = 'SOFT_BUY'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] > soft_buy) & (df['tdi_slow_ma'] < center_line), 'tdi_zone'] = 'BUY_ZONE'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] >= center_line) & (df['tdi_slow_ma'] < soft_sell), 'tdi_zone'] = 'SELL_ZONE'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] >= soft_sell) & (df['tdi_slow_ma'] < hard_sell), 'tdi_zone'] = 'SOFT_SELL'
            df.loc[valid_slow_ma & (df['tdi_slow_ma'] >= hard_sell), 'tdi_zone'] = 'HARD_SELL'
            
            df['tdi_strength'] = (df['tdi_slow_ma'] - center_line) / center_line
            df['tdi_strength'] = df['tdi_strength'].replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in _calculate_tdi_metrics: {str(e)}")
            if 'tdi_zone' not in df.columns: df['tdi_zone'] = 'NO_TRADE'
            if 'tdi_strength' not in df.columns: df['tdi_strength'] = 0.0
            return df

    @staticmethod
    def calculate_super_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Enhanced Super Bollinger Bands."""
        try:
            df = df.copy()
            bb_period = getattr(Config, 'BB_PERIOD', 20)
            bb_multiplier = getattr(Config.BB_DEV, 'BB_DEV', 2.0)
            
            if len(df) < bb_period:
                bb_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width_percent', 
                          'bb_rejection_buy', 'bb_rejection_sell', 'atr']
                for col in bb_cols: df[col] = np.nan
                return df

            df['bb_middle'] = df['close'].rolling(window=bb_period, min_periods=bb_period).mean()
            df['bb_std'] = df['close'].rolling(window=bb_period, min_periods=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_multiplier)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_multiplier)

            df = Indicators._calculate_atr_trend(df)
            df = Indicators._calculate_bb_metrics(df)

            df = df.drop(columns=['bb_std'], errors='ignore')
            
            return df

        except Exception as e:
            logger.error(f"Error in calculate_super_bollinger_bands: {str(e)}", exc_info=True)
            required_bb_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width_percent', 
                               'bb_rejection_buy', 'bb_rejection_sell', 'atr']
            for col in required_bb_cols:
                if col not in df.columns: df[col] = np.nan
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
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_width_percent'] = df['bb_width'] / df['bb_middle'].replace(0, np.nan)
            df['bb_width_percent'] = df['bb_width_percent'].fillna(0)
            
            df['bb_rejection_buy'] = False
            df['bb_rejection_sell'] = False
            
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
            df['bb_width_percent'] = np.nan
            df['bb_rejection_buy'] = False
            df['bb_rejection_sell'] = False
            return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators required for the Consolidated Trend Strategy."""
        try:
            df = df.copy()
            df.columns = [col.lower() for col in df.columns]
            
            df = Indicators.calculate_super_tdi(df)
            df = Indicators.calculate_super_bollinger_bands(df)
            
            df['tdi_trend'] = df['tdi_fast_ma'] - df['tdi_slow_ma']
            
            valid_slow_ma = df['tdi_slow_ma'].replace(0, np.nan)
            df['tdi_strength'] = abs(df['tdi_trend']) / valid_slow_ma
            df['tdi_strength'] = df['tdi_strength'].fillna(0)
            
            essential_cols = ['tdi_slow_ma', 'tdi_fast_ma', 'bb_width_percent']
            df = df.dropna(subset=essential_cols, how='all')
            
            return df

        except Exception as e:
            logger.error(f"Critical error in calculate_all_indicators: {str(e)}", exc_info=True)
            required_cols = ['tdi_slow_ma', 'tdi_fast_ma', 'tdi_zone', 'tdi_strength', 
                           'bb_width_percent', 'bb_rejection_buy', 'bb_rejection_sell']
            for col in required_cols:
                if col not in df.columns: df[col] = np.nan
            df['tdi_zone'] = df.get('tdi_zone', 'NO_TRADE')
            return df

    @staticmethod
    def get_alert_message_header(signal: str, strength: str, symbol: str) -> Tuple[str, str]:
        """Generates the header and action based on signal type and strength."""
        signal = signal.upper()
        strength = strength.upper()
        
        if signal == 'BUY':
            action = "LONG"
            if strength == 'HARD':
                header = f"🚨 **HARD BUY SIGNAL** | LONG *{symbol}* 🟢"
            elif strength == 'SOFT':
                header = f"🟢 Soft Buy Signal | LONG *{symbol}* 📊"
            else:
                header = f"🟢 BUY Signal | LONG *{symbol}* 📈"
            
        elif signal == 'SELL':
            action = "SHORT"
            if strength == 'HARD':
                header = f"🚨 **HARD SELL SIGNAL** | SHORT *{symbol}* 🔴"
            elif strength == 'SOFT':
                header = f"🔴 Soft Sell Signal | SHORT *{symbol}* 📉"
            else:
                header = f"🔴 SELL Signal | SHORT *{symbol}* 📉"
        else:
            header = f"ℹ️ Market State: NO TRADE on *{symbol}*"
            action = "NO_TRADE"
            
        return header, action
