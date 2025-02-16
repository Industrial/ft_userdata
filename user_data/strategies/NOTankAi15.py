import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional
import talib.abstract as ta
import pandas_ta as pta
from scipy.signal import argrelextrema
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
)
from freqtrade.persistence import Trade

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class NOTankAi15(IStrategy):
    """
    Improved strategy on 15-minute timeframe.

    Major improvements:
    - Dynamic stoploss based on ATR.
    - Dynamic calculation of leverage.
    - Calculation of Murrey Math levels.
    - Extended logging and code structuring.
    """

    # General parameters of the strategy
    timeframe = "15m"
    startup_candle_count: int = 200
    stoploss = -0.99  # Basic stoploss, overridden in custom_stoploss
    trailing_stop = False
    position_adjustment_enable = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1
    process_only_new_candles = True

    # DCA Parameters (Improved for Flexibility)
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02,
        high=-0.01,
        default=-0.018,
        decimals=3,
        space="buy",
        optimize=True,
        load=True,
    )
    max_safety_orders = IntParameter(1, 6, default=2, space="buy", optimize=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05,
        high=1.5,
        default=1.25,
        decimals=2,
        space="buy",
        optimize=True,
        load=True,
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )

    # Login Options
    increment = DecimalParameter(
        low=1.0005,
        high=1.002,
        default=1.001,
        decimals=4,
        space="buy",
        optimize=True,
        load=True,
    )
    last_entry_price = None

    # Security parameters
    cooldown_lookback = IntParameter(
        2, 48, default=1, space="protection", optimize=True
    )
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(
        default=True, space="protection", optimize=True
    )

    # ROI: Time - Percentage
    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "s_extrema": {"color": "#f53580"},
                "minima_sort_threshold": {"color": "#4ae747"},
                "maxima_sort_threshold": {"color": "#5b5e4b"},
            },
            "min_max": {
                "maxima": {"color": "#a29db9"},
                "minima": {"color": "#ac7fcc"},
                "maxima_check": {"color": "#a29db9"},
                "minima_check": {"color": "#ac7fcc"},
            },
        },
    }

    @property
    def protections(self):
        """Methods of defending a position."""
        prot = [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value,
            }
        ]
        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 72,  # 3 days (24 * 3)
                    "trade_limit": 2,
                    "stop_duration_candles": self.stop_duration.value,
                    "only_per_pair": False,
                }
            )
        return prot

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Adjusts the input volume for DCA."""
        return proposed_stake / self.max_dca_multiplier

    def custom_entry_price(
        self,
        pair: str,
        trade: Optional[Trade],
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Calculates the entry price taking into account previous entries and a
        small increment to avoid identical values.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        # Average of last candle close, last candle open and offered price
        entry_price = (
            dataframe["close"].iat[-1] + dataframe["open"].iat[-1] + proposed_rate
        ) / 3
        if proposed_rate < entry_price:
            entry_price = proposed_rate

        logger.info(
            f"{pair} Entry Price: {entry_price} | Close: {dataframe['close'].iat[-1]}, "
            f"Open: {dataframe['open'].iat[-1]}, Proposed: {proposed_rate}"
        )

        # Average of last candle close, last candle open and offered price
        if (
            self.last_entry_price is not None
            and abs(entry_price - self.last_entry_price) < 0.0005
        ):
            entry_price *= self.increment.value
            logger.info(
                f"{pair} Incremented entry price to {entry_price} (prev: {self.last_entry_price})."
            )
        self.last_entry_price = entry_price
        return entry_price

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Dynamic stop loss based on ATR. Stop loss is calculated as -1.5 * ATR,
        normalized by price.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        atr = ta.ATR(dataframe, timeperiod=14).iat[-1]
        # Example: stop loss = - (1.5 * ATR / current price)
        dynamic_sl = -1.5 * atr / current_rate
        # We log the calculated stop loss
        logger.info(
            f"{pair} Dynamic Stoploss: {dynamic_sl} (ATR: {atr}, Current Rate: {current_rate})"
        )
        return dynamic_sl

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Confirmation of exit from the trade. If the reason for exit is related
        to negative profit, the trade is not closed.
        """
        if (
            exit_reason in ["partial_exit", "trailing_stop_loss"]
            and trade.calc_profit_ratio(rate) < 0
        ):
            logger.info(f"{pair} Exit signal '{exit_reason}' отклонён, прибыль ниже 0.")
            self.dp.send_msg(
                f"{pair} Exit signal '{exit_reason}' отклонён, прибыль ниже 0."
            )
            return False
        return True

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Adjustment of the position based on the current profit and the number of
        entries/exits. If the profit is high enough, part of the position is
        closed.
        """
        count_of_entries = trade.nr_of_successful_entries

        if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 4)
        if current_profit > 0.40 and trade.nr_of_successful_exits == 1:
            return -(trade.stake_amount / 3)

        # If the loss is small, no adjustment is required.
        if (
            (current_profit > -0.15 and count_of_entries == 1)
            or (current_profit > -0.3 and count_of_entries == 2)
            or (current_profit > -0.6 and count_of_entries == 3)
        ):
            return None

        try:
            stake_amount = trade.select_filled_orders(trade.entry_side)[0].cost
            return stake_amount
        except Exception as e:
            logger.error(f"Error adjusting trade position: {e}")
            return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        """
        Dynamic calculation of leverage using RSI, ATR, MACD and SMA indicators.
        """
        window_size = 50
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        close_prices = dataframe["close"].tail(window_size)
        high_prices = dataframe["high"].tail(window_size)
        low_prices = dataframe["low"].tail(window_size)
        base_leverage = 10

        rsi = ta.RSI(close_prices, timeperiod=14)
        atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        macd, macdsignal, _ = ta.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        sma = ta.SMA(close_prices, timeperiod=20)

        current_rsi = rsi[-1] if len(rsi) > 0 else 50.0
        current_atr = atr[-1] if len(atr) > 0 else 0.0
        current_macd = (
            (macd[-1] - macdsignal[-1])
            if len(macd) > 0 and len(macdsignal) > 0
            else 0.0
        )
        current_sma = sma[-1] if len(sma) > 0 else current_rate

        # Threshold values ​​for RSI
        dynamic_rsi_low = (
            np.nanmin(rsi) if len(rsi) > 0 and not np.isnan(np.nanmin(rsi)) else 30.0
        )
        dynamic_rsi_high = (
            np.nanmax(rsi) if len(rsi) > 0 and not np.isnan(np.nanmax(rsi)) else 70.0
        )

        # Factors that change leverage
        long_increase = 1.5
        long_decrease = 0.5
        volatility_decrease = 0.8

        if side == "long":
            if current_rsi < dynamic_rsi_low:
                base_leverage *= long_increase
            elif current_rsi > dynamic_rsi_high:
                base_leverage *= long_decrease

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease

            if current_macd > 0:
                base_leverage *= long_increase
            if current_rate < current_sma:
                base_leverage *= long_decrease

        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)
        return adjusted_leverage

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        Adds basic indicators and Murrey Math levels.
        """
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0

        # We determine extremes (maximums and minimums)
        maxima = np.zeros(len(dataframe))
        minima = np.zeros(len(dataframe))
        maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
        minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
        dataframe["maxima"] = maxima
        dataframe["minima"] = minima

        # Extreme signals
        dataframe["s_extrema"] = 0
        min_peaks = argrelextrema(dataframe["close"].values, np.less, order=5)[0]
        max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=5)[0]
        dataframe.loc[min_peaks, "s_extrema"] = -1
        dataframe.loc[max_peaks, "s_extrema"] = 1

        # Calculating Murrey Math Levels
        murrey_levels = calculate_murrey_math_levels(dataframe)
        for level, series in murrey_levels.items():
            dataframe[level] = series

        # Additional oscillator (MML Extreme Oscillator)
        dataframe["mmlextreme_oscillator"] = 100 * (
            (dataframe["close"] - dataframe["[4/8]P"])
            / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"])
        )
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1
        )
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=10).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=10).max()
        dataframe["minima_check"] = (
            dataframe["minima"]
            .rolling(4)
            .apply(lambda x: int((x != 1).all()), raw=True)
            .fillna(0)
        )
        dataframe["maxima_check"] = (
            dataframe["maxima"]
            .rolling(4)
            .apply(lambda x: int((x != 1).all()), raw=True)
            .fillna(0)
        )

        pair = metadata.get("pair", "PAIR")
        if (
            dataframe["maxima"].iloc[-3] == 1
            and dataframe["maxima_check"].iloc[-1] == 0
        ):
            self.dp.send_msg(f"*** {pair} *** Maxima Detected - Potential Short!!!")
        if (
            dataframe["minima"].iloc[-3] == 1
            and dataframe["minima_check"].iloc[-1] == 0
        ):
            self.dp.send_msg(f"*** {pair} *** Minima Detected - Potential Long!!!")

        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generating signals to enter a position.
        """
        # Signals for long positions
        df.loc[
            (df["DI_catch"] == 1)
            & (df["maxima_check"] == 1)
            & (df["s_extrema"] < 0)
            & (df["minima"].shift(1) == 1)
            & (df["volume"] > 0)
            & (df["rsi"] < 30),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima")

        df.loc[
            (df["minima_check"] == 0) & (df["volume"] > 0) & (df["rsi"] < 30),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima Full Send")

        df.loc[
            (df["DI_catch"] == 1)
            & (df["minima_check"] == 0)
            & (df["minima_check"].shift(5) == 1)
            & (df["volume"] > 0)
            & (df["rsi"] < 30),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima Check")

        # Signals for short positions (if realized)
        df.loc[
            (df["DI_catch"] == 1)
            & (df["minima_check"] == 1)
            & (df["s_extrema"] > 0)
            & (df["maxima"].shift(1) == 1)
            & (df["volume"] > 0)
            & (df["rsi"] > 70),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima")

        df.loc[
            (df["maxima_check"] == 0) & (df["volume"] > 0) & (df["rsi"] > 70),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima Full Send")

        df.loc[
            (df["DI_catch"] == 1)
            & (df["maxima_check"] == 0)
            & (df["maxima_check"].shift(5) == 1)
            & (df["volume"] > 0)
            & (df["rsi"] > 70),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima Check")

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generating signals to exit a position.
        """
        df.loc[
            (df["maxima_check"] == 0) & (df["volume"] > 0), ["exit_long", "exit_tag"]
        ] = (1, "Maxima Check")
        df.loc[
            (df["DI_catch"] == 1)
            & (df["s_extrema"] > 0)
            & (df["maxima"].shift(1) == 1)
            & (df["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "Maxima")
        df.loc[
            (df["maxima_check"] == 0) & (df["volume"] > 0), ["exit_long", "exit_tag"]
        ] = (1, "Maxima Full Send")

        df.loc[
            (df["minima_check"] == 0) & (df["volume"] > 0), ["exit_short", "exit_tag"]
        ] = (1, "Minima Check")
        df.loc[
            (df["DI_catch"] == 1)
            & (df["s_extrema"] < 0)
            & (df["minima"].shift(1) == 1)
            & (df["volume"] > 0),
            ["exit_short", "exit_tag"],
        ] = (1, "Minima")
        df.loc[
            (df["minima_check"] == 0) & (df["volume"] > 0), ["exit_short", "exit_tag"]
        ] = (1, "Minima Full Send")
        return df


def calculate_murrey_math_levels(df: pd.DataFrame, window_size: int = 64) -> dict:
    """
    Calculates Murrey Math levels for each index in a DataFrame. Returns a
    dictionary where each level corresponds to a Series.
    """
    rolling_max_H = df["high"].rolling(window=window_size).max()
    rolling_min_L = df["low"].rolling(window=window_size).min()

    # Initialize the dictionary for levels
    murrey_levels = {
        key: []
        for key in [
            "[-3/8]P",
            "[-2/8]P",
            "[-1/8]P",
            "[0/8]P",
            "[1/8]P",
            "[2/8]P",
            "[3/8]P",
            "[4/8]P",
            "[5/8]P",
            "[6/8]P",
            "[7/8]P",
            "[8/8]P",
            "[+1/8]P",
            "[+2/8]P",
            "[+3/8]P",
        ]
    }

    def calculate_mml(mn: float, finalH: float, mx: float, finalL: float):
        """
        Calculates the mml value and returns Murrey Math levels.
        """
        dmml = ((finalH - finalL) / 8) * 1.0699
        mml = (mx * 0.99875) + (dmml * 3)
        ml = [mml - (dmml * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0],
        }

    for i in range(len(df)):
        mn = df["low"].iloc[: i + 1].min()
        mx = df["high"].iloc[: i + 1].max()
        finalH = df["high"].iloc[: i + 1].max()
        # We divide the range into 8 parts
        dmml = (mx - mn) / 8
        x_values = [mn + i * dmml for i in range(8)]
        midpoints = [(x_values[j] + x_values[j + 1]) / 2 for j in range(7)]
        finalL = min(midpoints) if midpoints else mn

        levels = calculate_mml(mn, finalH, mx, finalL)
        for key in murrey_levels.keys():
            murrey_levels[key].append(levels.get(key, np.nan))

    # Convert Lists to Series with DataFrame Indexes
    for key in murrey_levels:
        murrey_levels[key] = pd.Series(murrey_levels[key], index=df.index)
    return murrey_levels
