# utils/indicators.py (formerly indicator.py)
import pandas as pd
import numpy as np
from config import Config

class Indicators:

    @staticmethod
    def calculate_rsi(df, period=Config.TDI_RSI_PERIOD):
        """Calculates the Relative Strength Index (RSI)."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use ewm (Exponentially Weighted Moving Average) for RS, which is more common 
        # and more accurate for RSI than simple rolling mean, but your rolling mean logic is maintained for consistency:
        avg_gain = gain.rolling(window=period, min_periods=period).mean() 
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi
        return df

    @staticmethod
    def calculate_sma(df, column, period):
        """Calculates Simple Moving Average."""
        # Use a consistent naming convention that doesn't rely on f-string in return
        ma_col_name = f'{column}_sma_{period}'
        df[ma_col_name] = df[column].rolling(window=period, min_periods=period).mean()
        return df, ma_col_name

    @staticmethod
    def calculate_super_tdi(df):
        """
        Calculates the Super Traders Dynamic Index (TDI) components.
        Adds 'rsi', 'tdi_price_ma', 'tdi_fast_ma', 'tdi_slow_ma', 'tdi_bb_upper', 'tdi_bb_lower' to DataFrame.
        """
        # 1. Calculate RSI
        df = Indicators.calculate_rsi(df, period=Config.TDI_RSI_PERIOD)

        # 2. Calculate Price Line (Green Line) - SMA of RSI
        df, price_ma_col = Indicators.calculate_sma(df, 'rsi', Config.TDI_PRICE_MA_PERIOD)
        df['tdi_price_ma'] = df[price_ma_col]

        # 3. Calculate Fast MA (Yellow Line) - SMA of Price Line
        df, fast_ma_col = Indicators.calculate_sma(df, 'tdi_price_ma', Config.TDI_FAST_MA_PERIOD)
        df['tdi_fast_ma'] = df[fast_ma_col]

        # 4. Calculate Slow MA (Red Line) - SMA of Fast MA
        df, slow_ma_col = Indicators.calculate_sma(df, 'tdi_fast_ma', Config.TDI_SLOW_MA_PERIOD)
        df['tdi_slow_ma'] = df[slow_ma_col]

        # 5. Calculate Volatility Bands (Bollinger Bands on RSI)
        # --- FIX APPLIED HERE: Using TDI_RSI_PERIOD for the volatility bands ---
        bb_period = Config.TDI_RSI_PERIOD # Standard TDI uses the RSI period for its BB
        
        # Calculate Middle Band (SMA of RSI)
        df, middle_bb_col = Indicators.calculate_sma(df, 'rsi', bb_period)
        df['tdi_middle_bb'] = df[middle_bb_col]
        
        # Calculate Standard Deviation of RSI for BB
        df['tdi_std'] = df['rsi'].rolling(window=bb_period, min_periods=bb_period).std()
        
        # Calculate Upper/Lower Bands
        df['tdi_bb_upper'] = df['tdi_middle_bb'] + (df['tdi_std'] * Config.BB_DEV)
        df['tdi_bb_lower'] = df['tdi_middle_bb'] - (df['tdi_std'] * Config.BB_DEV)

        # Drop temporary columns
        cols_to_drop = [col for col in df.columns if '_sma_' in col or 'tdi_std' in col]
        return df.drop(columns=cols_to_drop, errors='ignore')


    @staticmethod
    def calculate_super_bollinger_bands(df):
        """
        Calculates Super Bollinger Bands.
        Adds 'bb_middle', 'bb_upper', 'bb_lower', 'bb_buy_signal', 'bb_sell_signal' to DataFrame.
        """
        bb_period = Config.BB_PERIOD
        
        df['bb_middle'] = df['close'].rolling(window=bb_period, min_periods=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period, min_periods=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * Config.BB_DEV)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * Config.BB_DEV)

        # Basic BB Buy/Sell signals based on candle close relative to bands
        df['bb_buy_signal'] = (df['close'] < df['bb_lower']) & \
                              (df['close'].shift(1) >= df['bb_lower'].shift(1)) # Close below lower band from above
        df['bb_sell_signal'] = (df['close'] > df['bb_upper']) & \
                               (df['close'].shift(1) <= df['bb_upper'].shift(1)) # Close above upper band from below

        return df.drop(columns=['bb_std'], errors='ignore')
