from operator import is_
from typing import Optional
from datetime import datetime
from pandas import DataFrame
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    informative,
    Trade,
)
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_seconds


# ElliottWaveOscilator
def EWO(source, sma_length=5, sma2_length=35):
    sma1 = ta.SMA(source, timeperiod=sma_length)
    sma2 = ta.SMA(source, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / source * 100
    return smadif


def is_positive(dataframe: DataFrame, column: str):
    return dataframe[column] > 0


def is_trending_up(dataframe: DataFrame, column_first: str, column_second: str):
    return dataframe[column_first] > dataframe[column_second]


def has_crossed_above(dataframe: DataFrame, column: str, threshold: float):
    return (dataframe[column] > threshold) & (dataframe[column].shift(1) < threshold)


class SampleStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "1h"
    timeframe_mins = timeframe_to_minutes(timeframe)

    # minimal_roi = {
    #     str(timeframe_mins * 00): 0.0100,  # 10.00% after 0 minutes
    #     str(timeframe_mins * 30): 0.0050,  # 7.50% After 30 minutes
    #     str(timeframe_mins * 60): 0.0010,  # 0.01% After 60 minutes
    # }

    minimal_roi = {"0": 0.446, "303": 0.165, "983": 0.042, "1299": 0}

    # stoploss = -0.10
    stoploss = -0.03
    use_custom_stoploss = False

    trailing_stop = True
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0
    # trailing_only_offset_is_reached = False

    # ATR Stop Loss
    # risk_reward_ratio = 1.00
    # risk_reward_ratio = 0.5
    # atr_distance = 10
    # timeframe_minutes = timeframe_to_minutes(timeframe)
    # custom_info_fixed_rr = dict()
    # init_fixed_rr_dict = {
    #     "roi": 0,
    #     "sl": 0,
    # }

    # process_only_new_candles = True
    # use_exit_signal = True
    # exit_profit_only = True
    # ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    # buy_rsi = IntParameter(
    #     low=1, high=50, default=30, space="buy", optimize=True, load=True
    # )
    # sell_rsi = IntParameter(
    #     low=50, high=100, default=70, space="sell", optimize=True, load=True
    # )

    rsi_period = IntParameter(
        low=2, high=100, default=14, space="buy", optimize=False, load=True
    )
    rsi_buy_threshold = IntParameter(
        low=10, high=70, default=30, space="buy", optimize=False, load=True
    )
    rsi_ema_period = IntParameter(
        low=2, high=100, default=9, space="buy", optimize=False, load=True
    )

    ewo_low_period = IntParameter(
        low=2, high=100, default=10, space="buy", optimize=False, load=True
    )
    ewo_high_period = IntParameter(
        low=51, high=100, default=100, space="buy", optimize=False, load=True
    )
    ewo_buy_threshold = DecimalParameter(
        low=0.1, high=10, default=3, space="buy", optimize=False, load=True
    )

    tema_small_period = IntParameter(
        low=1, high=10000, default=283, space="buy", optimize=False, load=True
    )
    tema_large_period = IntParameter(
        low=284, high=100000, default=10000, space="buy", optimize=False, load=True
    )

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema_small_btc_1h": {
                "color": "cyan",
            },
            "tema_large_btc_1h": {
                "color": "blue",
            },
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
                "rsi_ema": {"color": "green"},
            },
            # "EWO": {
            #     "ewo": {"color": "blue"},
            # },
        },
    }

    def informative_pairs(self):
        return []

    # @informative("1h", "BTC/{stake}")
    # def populate_indicators_btc_1h(
    #     self, dataframe: DataFrame, metadata: dict
    # ) -> DataFrame:
    #     dataframe["tema_small"] = ta.TEMA(
    #         dataframe, timeperiod=int(self.tema_small_period.value)
    #     )
    #     dataframe["tema_large"] = ta.TEMA(
    #         dataframe, timeperiod=int(self.tema_large_period.value)
    #     )
    #     return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=int(self.rsi_period.value))
        dataframe["rsi_ema"] = ta.RSI(
            dataframe["rsi"], timeperiod=int(self.rsi_ema_period.value)
        )
        dataframe["ewo"] = EWO(
            dataframe["close"],
            int(self.ewo_low_period.value),
            int(self.ewo_high_period.value),
        )
        dataframe["tema_small"] = ta.TEMA(
            dataframe, timeperiod=int(self.tema_small_period.value)
        )
        dataframe["tema_large"] = ta.TEMA(
            dataframe, timeperiod=int(self.tema_large_period.value)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                is_trending_up(dataframe, "tema_small", "tema_large")
                # & has_crossed_above(dataframe, "rsi_ema", self.rsi_buy_threshold.value)
                & has_crossed_above(dataframe, "ewo", self.ewo_buy_threshold.value)
                & is_positive(dataframe, "volume")
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # is_trending_down_large = (
        #     dataframe["btc_usdt_tema_small_1h"] < dataframe["btc_usdt_tema_large_1h"]
        # )
        # dataframe.loc[
        #     (is_trending_down_large),
        #     "exit_long",
        # ] = 1
        return dataframe

    # def custom_exit(
    #     self,
    #     pair: str,
    #     trade: Trade,
    #     current_time: datetime,
    #     current_rate: float,
    #     current_profit: float,
    #     **kwargs
    # ) -> Optional[str]:
    #     if (current_time - trade.open_date_utc).total_seconds() >= (
    #         60
    #         * 60
    #         * 24
    #         # 50 * timeframe_to_seconds(self.timeframe)
    #     ):
    #         return "exit_timeout"
    #     return None

    # def confirm_trade_entry(
    #     self,
    #     pair: str,
    #     order_type: str,
    #     amount: float,
    #     rate: float,
    #     time_in_force: str,
    #     current_time: datetime,
    #     entry_tag: Optional[str],
    #     side: str,
    #     **kwargs
    # ) -> bool:
    #     if pair not in self.custom_info_fixed_rr:
    #         self.custom_info_fixed_rr[pair] = self.init_fixed_rr_dict.copy()
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     current_candle = dataframe.iloc[-1].squeeze()
    #     self.custom_info_fixed_rr[pair]["roi"] = current_candle["close"] + (
    #         self.atr_distance * self.risk_reward_ratio * current_candle["atr"]
    #     )
    #     self.custom_info_fixed_rr[pair]["sl"] = current_candle["close"] - (
    #         self.atr_distance * current_candle["atr"]
    #     )
    #     return True

    # def confirm_trade_exit(
    #     self,
    #     pair: str,
    #     trade: Trade,
    #     order_type: str,
    #     amount: float,
    #     rate: float,
    #     time_in_force: str,
    #     exit_reason: str,
    #     current_time: datetime,
    #     **kwargs
    # ) -> bool:
    #     if pair in self.custom_info_fixed_rr:
    #         self.custom_info_fixed_rr[pair] = self.init_fixed_rr_dict.copy()
    #     return True

    # def custom_exit(
    #     self,
    #     pair: str,
    #     trade: Trade,
    #     current_time: datetime,
    #     current_rate: float,
    #     current_profit: float,
    #     **kwargs
    # ) -> Optional[Union[str, bool]]:
    #     entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     atr_roi = trade.get_custom_data(key="atr_roi", default=0)
    #     atr_sl = trade.get_custom_data(key="atr_sl", default=0)
    #     if atr_roi == 0:
    #         # need to set the roi and sl
    #         signal_time = entry_time - timedelta(minutes=int(self.timeframe_minutes))
    #         signal_candle = dataframe.loc[dataframe["date"] == signal_time]
    #         if not signal_candle.empty:
    #             # make sure we take only a row
    #             signal_candle = signal_candle.iloc[-1].squeeze()
    #             atr_roi = signal_candle["close"] + (
    #                 self.atr_distance * self.risk_reward_ratio * signal_candle["atr"]
    #             )
    #             atr_sl = signal_candle["close"] - (
    #                 self.atr_distance * signal_candle["atr"]
    #             )
    #             trade.set_custom_data(key="atr_roi", value=atr_roi)
    #             trade.set_custom_data(key="atr_sl", value=atr_sl)
    #     if current_time - timedelta(minutes=int(self.timeframe_minutes)) >= entry_time:
    #         current_candle = dataframe.iloc[-1].squeeze()
    #         if atr_roi > 0:
    #             if current_candle["close"] >= atr_roi:
    #                 return "atr_roi"
    #             if current_candle["close"] <= atr_sl:
    #                 return "atr_sl"
    #         else:
    #             # Signal candle not found, use percentage as exits
    #             current_profit = trade.calc_profit_ratio(current_candle["close"])
    #             if current_profit > 0.01:
    #                 return "emergency_roi"
    #             elif current_profit < -0.04:
    #                 return "emergency_sl"
    #     return None
