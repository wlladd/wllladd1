# src/live_trader.py

import time
import logging
import pickle
import pandas as pd
import numpy as np
from data_fetcher import DataFetcher
from feature_generator import FeatureGenerator
from telegram_notifier import TelegramNotifier
from db_manager import DBManager

class LiveTrader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.db = DBManager(cfg["data"]["db_path"])
        self.fetcher = DataFetcher(
            cfg["api"]["base_url"],
            cfg["api"].get("api_key", ""),
            cfg["api"].get("api_secret", ""),
            self.db
        )
        self.feature_gen = FeatureGenerator(cfg, self.db)
        self.notifier = TelegramNotifier(cfg["telegram"]["token"], cfg["telegram"]["chat_id"])

        self.symbol = cfg["default_fetch"]["symbol"]
        self.timeframe = cfg["default_fetch"]["timeframe"]
        self.poll_interval = cfg["live_trading"]["polling_interval_sec"]

        self.models = []
        self.model_names = cfg["plugins"]["models"]
        self.combine_logic = cfg["live_trading"]["combine_signal_logic"]

        self.risk_cfg = cfg["risk_management"]

        self._stop_flag = False
        self.is_running = False

        self.manual_order_pending = None

    def load_models(self, model_paths: list, scaler_paths: list):
        self.models = []
        for i, mp in enumerate(model_paths):
            with open(mp, "rb") as f:
                model_obj = pickle.load(f)
            self.models.append(model_obj)

    def stop(self):
        self._stop_flag = True

    def _get_combined_signal(self, last_feat: pd.DataFrame) -> int:
        preds = []
        X = last_feat.values
        for model in self.models:
            try:
                p = model.predict(X)[0]
            except:
                p = model.predict(X)
                if isinstance(p, np.ndarray):
                    p = p[0]
            p = 1 if (isinstance(p, (float, np.floating)) and p > 0.5) or (isinstance(p, (int, np.integer)) and p == 1) else 0
            preds.append(p)
        preds_arr = np.array(preds)
        if self.combine_logic == "AND":
            return int(np.all(preds_arr == 1))
        elif self.combine_logic == "OR":
            return int(np.any(preds_arr == 1))
        else:
            raise NotImplementedError("Custom combine logic не реализован.")

    def _compute_position_size(self, balance: float, price: float) -> float:
        mode = self.risk_cfg["mode"]
        if mode == "fixed_size":
            return float(self.risk_cfg.get("size_per_trade", 0))
        elif mode == "percent_of_balance":
            pip_value = 10
            risk_value = balance * (self.risk_cfg["risk_percent"] / 100)
            sl_pips = float(self.risk_cfg["stop_loss_pips"])
            if sl_pips <= 0:
                return 0.0
            lot = risk_value / (sl_pips * pip_value)
            return round(lot, 3)
        else:
            return float(self.risk_cfg.get("size_per_trade", 0))

    def start(self, model_paths: list, scaler_paths: list):
        try:
            self.load_models(model_paths, scaler_paths)
        except Exception as e:
            logging.error(f"Не удалось загрузить модели/масштабировщики: {e}")
            if self.cfg["telegram"]["send_on_error"]:
                self.notifier.send_error(str(e))
            return

        self._stop_flag = False
        self.is_running = True

        if self.cfg["telegram"]["send_on_order_open"]:
            self.notifier.send_message("LiveTrader запущен (реальные сделки).")

        while not self._stop_flag:
            try:
                bar = self.fetcher.get_realtime_data(self.symbol, self.timeframe)
                df_bar = pd.DataFrame([bar]).set_index("Datetime")

                df_hist = self.db.fetch_candles(self.symbol, self.timeframe)
                df_combined = pd.concat([df_hist, df_bar]).iloc[-100:]
                selected_inds = self.cfg["plugins"]["indicators"]
                params_dict = self.cfg["plugins"]["indicators_params"]
                df_feat = self.feature_gen.generate(self.symbol, self.timeframe, selected_inds, params_dict, include_order_book=self.cfg["default_fetch"]["collect_order_book"])
                last_feat = df_feat.iloc[[-1]]

                # Обрабатываем ручной ордер (если есть)
                if self.manual_order_pending:
                    side = self.manual_order_pending["side"]
                    volume = self.manual_order_pending["volume"]
                    try:
                        order_res = self.fetcher.place_order(self.symbol, side, volume)
                        if self.cfg["telegram"]["send_on_order_open"]:
                            self.notifier.send_order_opened(side, self.symbol, bar["Close"], volume, self.manual_order_pending["reason"])
                        self.manual_order_pending = None
                    except Exception as e:
                        logging.error(f"Ошибка вручную открытия ордера: {e}")
                        if self.cfg["telegram"]["send_on_error"]:
                            self.notifier.send_error(str(e))
                    time.sleep(self.poll_interval)
                    continue

                # Если полуавтоматический режим включён, пропускаем автоматические сигналы
                if self.cfg["live_trading"]["manual_override"]:
                    time.sleep(self.poll_interval)
                    continue

                signal = self._get_combined_signal(last_feat)

                if signal == 1:
                    price_entry = bar["Close"]
                    last_lt = self.db.fetch_live_trades(limit=1)
                    if not last_lt.empty:
                        balance = last_lt.iloc[0]["exit_price"] * last_lt.iloc[0]["volume"] + last_lt.iloc[0]["profit"]
                    else:
                        balance = float(self.cfg["simulation"]["initial_balance"])
                    lot = self._compute_position_size(balance, price_entry)
                    if lot > 0:
                        order_res = self.fetcher.place_order(self.symbol, "BUY", lot)
                        if self.cfg["telegram"]["send_on_order_open"]:
                            self.notifier.send_order_opened("BUY", self.symbol, price_entry, lot, f"Signal: {self.combine_logic}")
                        entry_dt = bar["Datetime"].strftime("%Y-%m-%d %H:%M:%S")
                        time.sleep(self.poll_interval)
                        next_bar = self.fetcher.get_realtime_data(self.symbol, self.timeframe)
                        exit_price = next_bar["Close"]
                        exit_dt = next_bar["Datetime"].strftime("%Y-%m-%d %H:%M:%S")
                        profit = (exit_price - price_entry) * lot * 100000
                        self.db.insert_live_trade(
                            entry_time=entry_dt,
                            exit_time=exit_dt,
                            symbol=self.symbol,
                            side="BUY",
                            entry_price=price_entry,
                            exit_price=exit_price,
                            volume=lot,
                            profit=profit,
                            model_name=",".join(self.model_names),
                            strategy_name=",".join(self.model_names),
                            risk_used=lot
                        )
                        if self.cfg["telegram"]["send_on_order_close"]:
                            self.notifier.send_order_closed("BUY", self.symbol, price_entry, exit_price, lot, profit, f"Signal: {self.combine_logic}")
                    else:
                        logging.info("Нулевой объём лота → сделка не открыта.")
                else:
                    logging.info("Сигнал от моделей отсутствует — ждем следующий тик.")
            except Exception as e:
                logging.error(f"Ошибка в LiveTrader: {e}", exc_info=True)
                if self.cfg["telegram"]["send_on_error"]:
                    self.notifier.send_error(str(e))
                time.sleep(10)
                continue

            time.sleep(self.poll_interval)

        self.is_running = False
        if self.cfg["telegram"]["send_on_order_close"]:
            self.notifier.send_message("LiveTrader остановлен.")

    def manual_open(self, side: str, volume: float, reason: str="Manual"):
        self.manual_order_pending = {"side": side, "volume": volume, "reason": reason}

    def manual_close_all(self):
        try:
            open_orders = self.fetcher.get_open_orders()
            for ord in open_orders:
                order_id = ord.get("orderId") or ord.get("order_id")
                self.fetcher.cancel_order(order_id, self.symbol)
            if self.cfg["telegram"]["send_on_order_close"]:
                self.notifier.send_message("Все открытые ордера отменены вручную.")
        except Exception as e:
            logging.error(f"Ошибка manual_close_all: {e}")
            if self.cfg["telegram"]["send_on_error"]:
                self.notifier.send_error(str(e))
