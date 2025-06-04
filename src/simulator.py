# src/simulator.py

import pandas as pd
import numpy as np
import logging

class BacktestSimulator:
    def __init__(self, df_features: pd.DataFrame, cfg: dict, models: list, risk_cfg: dict, strategy_logic: str="AND"):
        self.df = df_features.copy()
        self.cfg = cfg
        self.models = models
        self.model_names = cfg["plugins"]["models"]
        self.risk_mode = risk_cfg["mode"]
        self.fixed_size = float(risk_cfg.get("size_per_trade", 0))
        self.risk_percent = float(risk_cfg.get("risk_percent", 0))
        self.stop_loss_pips = int(risk_cfg.get("stop_loss_pips", 0))
        self.take_profit_pips = int(risk_cfg.get("take_profit_pips", 0))
        self.max_daily_trades = int(risk_cfg.get("max_daily_trades", 0))
        self.trade_hours_start = int(risk_cfg["trading_time"]["start_hour"])
        self.trade_hours_end = int(risk_cfg["trading_time"]["end_hour"])
        self.strategy_logic = strategy_logic.upper()
        self.results = None

    def generate_labels(self, N: int = 10) -> pd.DataFrame:
        df = self.df
        df["Open_next"] = df["Close"].shift(-1)
        df["Future_Close_N"] = df["Close"].shift(-N)
        df["Target_Long"] = np.where(df["Future_Close_N"] > df["Open_next"], 1, 0)
        df["Target_Short"] = np.where(df["Future_Close_N"] < df["Open_next"], 1, 0)
        df = df.dropna(subset=["Target_Long", "Target_Short"])
        self.df = df
        return df

    def _compute_position_size(self, balance: float, price: float) -> float:
        if self.risk_mode == "fixed_size":
            return self.fixed_size
        elif self.risk_mode == "percent_of_balance":
            pip_value = 10
            risk_value = balance * (self.risk_percent / 100)
            if self.stop_loss_pips <= 0:
                return 0.0
            lot = risk_value / (self.stop_loss_pips * pip_value)
            return round(lot, 3)
        else:
            return self.fixed_size

    def run_backtest(self) -> dict:
        df = self.df.copy()
        all_preds = []
        X = df.drop(columns=["Target_Long", "Target_Short", "Open_next", "Future_Close_N"])
        for model in self.models:
            try:
                preds = model.predict(X.values)
                try:
                    preds = (preds > 0.5).astype(int)
                except:
                    preds = preds.astype(int)
            except:
                preds = model.predict(X.values).astype(int)
            all_preds.append(preds)

        preds_matrix = np.vstack(all_preds)
        if self.strategy_logic == "AND":
            combined = np.all(preds_matrix == 1, axis=0).astype(int)
        elif self.strategy_logic == "OR":
            combined = np.any(preds_matrix == 1, axis=0).astype(int)
        else:
            raise NotImplementedError("Custom logic не реализован.")

        df["Signal"] = combined

        balance = float(self.cfg["simulation"]["initial_balance"])
        trade_log = []
        equity_curve = []
        daily_trade_count = {}
        current_date = None

        for i, (ts, row) in enumerate(df.iterrows()):
            hour = ts.hour
            if not (self.trade_hours_start <= hour < self.trade_hours_end):
                equity_curve.append({"time": ts, "balance": balance})
                continue

            date_str = ts.strftime("%Y-%m-%d")
            if current_date != date_str:
                current_date = date_str
                daily_trade_count[current_date] = 0

            if daily_trade_count[current_date] >= self.max_daily_trades:
                equity_curve.append({"time": ts, "balance": balance})
                continue

            signal = int(row["Signal"])
            price_entry = row["Open_next"]
            if signal == 1 and price_entry is not None:
                lot = self._compute_position_size(balance, price_entry)
                if lot <= 0:
                    equity_curve.append({"time": ts, "balance": balance})
                    continue

                sl_price = price_entry - (self.stop_loss_pips * 0.0001)
                tp_price = price_entry + (self.take_profit_pips * 0.0001)

                exit_price = None
                exit_ts = None
                N = int(self.cfg["simulation"].get("take_profit_pips", 10))
                for j in range(i + 1, len(df)):
                    low_j = df.iloc[j]["Low"]
                    high_j = df.iloc[j]["High"]
                    if low_j <= sl_price:
                        exit_price = sl_price
                        exit_ts = df.index[j]
                        break
                    if high_j >= tp_price:
                        exit_price = tp_price
                        exit_ts = df.index[j]
                        break
                    if j == i + N:
                        exit_price = df.iloc[j]["Future_Close_N"]
                        exit_ts = df.index[j]
                        break

                if exit_price is None:
                    exit_price = df.iloc[-1]["Close"]
                    exit_ts = df.index[-1]

                profit = (exit_price - price_entry) * lot * 100000
                balance += profit

                trade_log.append({
                    "entry_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": exit_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price": price_entry,
                    "exit_price": exit_price,
                    "volume": lot,
                    "profit": profit,
                    "model_name": ",".join(self.model_names),
                    "strategy_name": ",".join(self.model_names),
                    "risk_used": lot
                })

                daily_trade_count[current_date] += 1

            equity_curve.append({"time": ts, "balance": balance})

        trades_df = pd.DataFrame(trade_log)
        equity_df = pd.DataFrame(equity_curve)

        total_trades = len(trades_df)
        if total_trades > 0:
            wins = trades_df[trades_df["profit"] > 0]
            losses = trades_df[trades_df["profit"] <= 0]
            win_rate = len(wins) / total_trades * 100
            profit_factor = wins["profit"].sum() / abs(losses["profit"].sum()) if len(losses) > 0 else float("inf")
        else:
            win_rate = 0
            profit_factor = 0

        self.results = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "final_balance": balance,
            "trade_log": trades_df,
            "equity": equity_df
        }
        return self.results
