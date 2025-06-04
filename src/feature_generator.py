# src/feature_generator.py

import pandas as pd
import numpy as np
from indicators_manager import IndicatorManager

class FeatureGenerator:
    def __init__(self, cfg: dict, db_manager=None):
        self.cfg = cfg
        self.db  = db_manager
        self.ind_manager = IndicatorManager()

    def generate(self, symbol: str, timeframe: str, selected_inds: list, params_dict: dict, include_order_book: bool=False) -> pd.DataFrame:
        # 1) Загрузить свечи
        df = self.db.fetch_candles(symbol, timeframe)
        if df.empty:
            raise ValueError("Нет данных свечей для выбранных symbol/timeframe.")

        # 2) При необходимости добавить order_book
        if include_order_book or self.cfg["default_fetch"].get("collect_order_book", False):
            ob_df = self.db.fetch_order_book(symbol)
            if not ob_df.empty:
                ob_df = ob_df.set_index("datetime").reindex(df.index, method="ffill")
                df["bid_prices"]  = ob_df["bid_prices"]
                df["bid_volumes"] = ob_df["bid_volumes"]
                df["ask_prices"]  = ob_df["ask_prices"]
                df["ask_volumes"] = ob_df["ask_volumes"]
            else:
                df["bid_prices"]  = [[] for _ in range(len(df))]
                df["bid_volumes"] = [[] for _ in range(len(df))]
                df["ask_prices"]  = [[] for _ in range(len(df))]
                df["ask_volumes"] = [[] for _ in range(len(df))]

        # 3) Базовые индикаторы (PSAR, ADX)
        import pandas_ta as ta
        # PSAR
        step = self.cfg["indicators"]["psar"]["step"]
        max_step = self.cfg["indicators"]["psar"]["max_step"]
        psar = ta.psar(high=df["High"], low=df["Low"], close=df["Close"], step=step, max_step=max_step)
        psar_col = [c for c in psar.columns if c.startswith("PSARl_")][0]
        df["PSAR_dir"] = psar[psar_col].combine(df["Close"], lambda s, c: 1 if s < c else -1)

        # ADX и DI
        period = self.cfg["indicators"]["adx"]["period"]
        adx_di = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=period)
        adx_di = adx_di.rename(columns={f"DMP_{period}":"PlusDI", f"DMN_{period}":"MinusDI", f"ADX_{period}":"ADX"})
        df = pd.concat([df, adx_di[["PlusDI","MinusDI","ADX"]]], axis=1)
        df["PlusDI_prev"] = df["PlusDI"].shift(1)
        df["DIp_dir"]     = np.where(df["PlusDI"] > df["PlusDI_prev"], 1, -1)
        df["MinusDI_prev"]= df["MinusDI"].shift(1)
        df["DIn_dir"]     = np.where(df["MinusDI"] > df["MinusDI_prev"], 1, -1)
        df["ADX_low"]     = np.where(df["ADX"] < 25, 1, 0)
        df["ADX_medium"]  = np.where((df["ADX"] >= 25) & (df["ADX"] < 50), 1, 0)
        df["ADX_high"]    = np.where(df["ADX"] >= 50, 1, 0)

        # 4) Плагины-индикаторы
        plugin_feats = self.ind_manager.calculate_all(df, selected_inds, params_dict)
        df = pd.concat([df, plugin_feats], axis=1)

        # 5) (Опционально) Можно добавить любые бинарные «выше/ниже» признаки здесь

        # 6) Удаление NaN
        df = df.dropna()

        # 7) Сохранение в БД
        if self.db:
            self.db.insert_features(df, symbol, timeframe)

        return df
