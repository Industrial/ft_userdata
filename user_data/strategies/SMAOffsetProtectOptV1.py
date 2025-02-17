from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    stoploss_from_open,
    merge_informative_pair,
    DecimalParameter,
    IntParameter,
    CategoricalParameter,
)
import technical.indicators as ftt

buy_params = {
    "base_nb_candles_buy": 16,
    "ewo_high": 5.638,
    "ewo_low": -19.993,
    "low_offset": 0.978,
    "rsi_buy": 61,
}

sell_params = {
    "base_nb_candles_sell": 49,
    "high_offset": 1.006,
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif


class SMAOffsetProtectOptV1(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.02,
        "30": 0.01,
    }

    stoploss = -0.5
    # stoploss = -0.03

    base_nb_candles_buy = IntParameter(2, 1000, default=158, space="buy", optimize=True)
    base_nb_candles_sell = IntParameter(
        2,
        1000,
        default=184,
        space="sell",
        optimize=True,
    )
    low_offset = DecimalParameter(0.9, 0.99, default=0.949, space="buy", optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=1.099, space="sell", optimize=True
    )

    fast_ewo = IntParameter(2, 100, default=84, space="buy", optimize=True)
    slow_ewo = IntParameter(101, 500, default=402, space="buy", optimize=True)
    ewo_low = DecimalParameter(-20.0, -8.0, default=-9.988, space="buy", optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=6.598, space="buy", optimize=True)
    rsi_buy = IntParameter(30, 70, default=54, space="buy", optimize=True)

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    timeframe = "5m"
    informative_timeframe = "1h"

    process_only_new_candles = True
    startup_candle_count = 30

    plot_config = {
        "main_plot": {
            "ma_buy": {"color": "orange"},
            "ma_sell": {"color": "orange"},
        },
    }

    use_custom_stoploss = False

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe=self.informative_timeframe
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.base_nb_candles_buy.range:
            dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        dataframe["EWO"] = EWO(
            dataframe, int(self.fast_ewo.value), int(self.slow_ewo.value)
        )

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (
                    dataframe["close"]
                    < (
                        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                        * self.low_offset.value
                    )
                )
                & (dataframe["EWO"] > self.ewo_high.value)
                & (dataframe["rsi"] < self.rsi_buy.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                (
                    dataframe["close"]
                    < (
                        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                        * self.low_offset.value
                    )
                )
                & (dataframe["EWO"] < self.ewo_low.value)
                & (dataframe["volume"] > 0)
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (
                    dataframe["close"]
                    > (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset.value
                    )
                )
                & (dataframe["volume"] > 0)
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe
