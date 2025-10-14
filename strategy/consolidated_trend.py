import pandas as pd
import numpy as np
from utils.indicators import Indicators
from settings import Config
import logging

logger = logging.getLogger(__name__)

RRR_RATIO = 1.5

class ConsolidatedTrendStrategy:
    def __init__(self):
        logger.info("Consolidated/Trend Trading Strategy initialized.")

    def analyze_data(self, df):
        """
        Applies all indicators and prepares data for signal generation.
        """
        if df.empty:
            return df

        df.columns = [col.lower() for col in df.columns]

        df = Indicators.calculate_super_tdi(df.copy())
        df = Indicators.calculate_super_bollinger_bands(df.copy())

        df.dropna(inplace=True)

        return df

    def _calculate_structural_sl_tp(self, df, last_candle, signal_type, risk_factor=1.0):
        """
        Calculates structural Stop Loss and Take Profit based on recent swing high/low 
        and a fixed RRR_RATIO.
        """
        entry_price = last_candle['close']
        lookback_period = 10
        start_index = max(0, len(df) - lookback_period)
        lookback_df = df.iloc[start_index:]

        if signal_type == "BUY":
            stop_loss = lookback_df['low'].min() * 0.998
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995
            sl_distance = entry_price - stop_loss
            take_profit = entry_price + (sl_distance * RRR_RATIO * risk_factor)

        elif signal_type == "SELL":
            stop_loss = lookback_df['high'].max() * 1.002
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005
            sl_distance = stop_loss - entry_price
            take_profit = entry_price - (sl_distance * RRR_RATIO * risk_factor)

        else:
            return entry_price, 0, 0

        return entry_price, stop_loss, take_profit

    def generate_signal(self, df):
        """
        Generates buy/sell signals based on the strategy rules.
        """
        min_data = max(Config.TDI_RSI_PERIOD, Config.BB_PERIOD, Config.TDI_SLOW_MA_PERIOD) + 5
        if df.empty or len(df) < min_data:
            logger.warning(f"DataFrame too small for signal generation (Need >{min_data} rows).")
            return "NO_TRADE", {}

        last_candle = df.iloc[-1]
        recent_df = df.iloc[-3:]  # Lookback for crossover flexibility

        rsi = last_candle['rsi']
        fast_ma = last_candle['tdi_fast_ma']
        slow_ma = last_candle['tdi_slow_ma']
        close = last_candle['close']
        bb_lower = last_candle['bb_lower']
        bb_upper = last_candle['bb_upper']
        low = last_candle['low']
        high = last_candle['high']

        risk_factor = 1.0

        logger.debug(f"RSI: {rsi:.2f}, Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f}")
        logger.debug(f"BB Lower: {bb_lower:.2f}, BB Upper: {bb_upper:.2f}, Close: {close:.2f}")

        bullish_crossover = any(
            recent_df.iloc[i]['tdi_fast_ma'] > recent_df.iloc[i]['tdi_slow_ma'] and
            recent_df.iloc[i - 1]['tdi_fast_ma'] <= recent_df.iloc[i - 1]['tdi_slow_ma']
            for i in range(1, len(recent_df))
        )
        bearish_crossover = any(
            recent_df.iloc[i]['tdi_fast_ma'] < recent_df.iloc[i]['tdi_slow_ma'] and
            recent_df.iloc[i - 1]['tdi_fast_ma'] >= recent_df.iloc[i - 1]['tdi_slow_ma']
            for i in range(1, len(recent_df))
        )

        bb_rejection_buy = (low <= bb_lower or df.iloc[-2]['low'] <= df.iloc[-2]['bb_lower']) and \
                           (close > df.iloc[-2]['close']) and (close > bb_lower)
        bb_rejection_sell = (high >= bb_upper or df.iloc[-2]['high'] >= df.iloc[-2]['bb_upper']) and \
                            (close < df.iloc[-2]['close']) and (close < bb_upper)

        consolidating_tdi = (Config.TDI_NO_TRADE_ZONE_START <= rsi <= Config.TDI_NO_TRADE_ZONE_END) and \
                            (abs(fast_ma - slow_ma) < 0.5)

        # --- BUY Signal ---
        if bullish_crossover and rsi < 55 and bb_rejection_buy:
            if rsi <= Config.TDI_HARD_BUY_LEVEL:
                risk_factor = 1.5
                logger.info(f"Hard Buy Signal (RSI < {Config.TDI_HARD_BUY_LEVEL})")
            elif rsi <= Config.TDI_SOFT_BUY_LEVEL:
                risk_factor = 1.2
                logger.info(f"Soft Buy Signal (RSI < {Config.TDI_SOFT_BUY_LEVEL})")

            entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "BUY", risk_factor)
            logger.info(f"BUY Signal: RSI={rsi:.2f}, Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            return "BUY", {
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_factor": risk_factor,
                "tdi_value": rsi
            }

        # --- SELL Signal ---
        if bearish_crossover and rsi > 45 and bb_rejection_sell:
            if rsi >= Config.TDI_HARD_SELL_LEVEL:
                risk_factor = 1.5
                logger.info(f"Hard Sell Signal (RSI > {Config.TDI_HARD_SELL_LEVEL})")
            elif rsi >= Config.TDI_SOFT_SELL_LEVEL:
                risk_factor = 1.2
                logger.info(f"Soft Sell Signal (RSI > {Config.TDI_SOFT_SELL_LEVEL})")

            entry, sl, tp = self._calculate_structural_sl_tp(df, last_candle, "SELL", risk_factor)
            logger.info(f"SELL Signal: RSI={rsi:.2f}, Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
            return "SELL", {
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_factor": risk_factor,
                "tdi_value": rsi
            }

        # --- NO TRADE ---
        if consolidating_tdi:
            logger.info(f"TDI Consolidation: RSI={rsi:.2f}, No Trade Zone.")
            return "NO_TRADE", {"reason": "TDI stalling around 50."}

        logger.info("No valid trade signal found.")
        return "NO_TRADE", {}
