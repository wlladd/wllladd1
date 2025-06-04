# src/db_manager.py

import sqlite3
import threading
import json
import pandas as pd

class DBManager:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: str):
        with cls._lock:
            if db_path not in cls._instances:
                instance = super(DBManager, cls).__new__(cls)
                instance._init(db_path)
                cls._instances[db_path] = instance
            return cls._instances[db_path]

    def _init(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        # Таблица candles
        cur.execute("""
        CREATE TABLE IF NOT EXISTS candles(
            datetime TEXT PRIMARY KEY,
            symbol TEXT,
            timeframe TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """)
        # Таблица order_book
        cur.execute("""
        CREATE TABLE IF NOT EXISTS order_book(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT,
            symbol TEXT,
            bid_prices TEXT,
            bid_volumes TEXT,
            ask_prices TEXT,
            ask_volumes TEXT
        )
        """)
        # Таблица features
        cur.execute("""
        CREATE TABLE IF NOT EXISTS features(
            datetime TEXT PRIMARY KEY,
            symbol TEXT,
            timeframe TEXT,
            data TEXT
        )
        """)
        # Таблица models
        cur.execute("""
        CREATE TABLE IF NOT EXISTS models(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            model_type TEXT,
            created_at TEXT,
            hash TEXT,
            path TEXT,
            accuracy REAL,
            config_snapshot TEXT,
            is_ensemble INTEGER DEFAULT 0,
            ensemble_members TEXT
        )
        """)
        # Таблица live_trades
        cur.execute("""
        CREATE TABLE IF NOT EXISTS live_trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_time TEXT,
            exit_time TEXT,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            volume REAL,
            profit REAL,
            model_name TEXT,
            strategy_name TEXT,
            risk_used REAL
        )
        """)
        # Таблица risk_params
        cur.execute("""
        CREATE TABLE IF NOT EXISTS risk_params(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            size_per_trade REAL,
            risk_percent REAL,
            stop_loss_pips INTEGER,
            take_profit_pips INTEGER,
            max_daily_trades INTEGER,
            trading_start_hour INTEGER,
            trading_end_hour INTEGER
        )
        """)
        # Таблица strategies
        cur.execute("""
        CREATE TABLE IF NOT EXISTS strategies(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            params TEXT
        )
        """)
        self.conn.commit()

    def insert_candles(self, df: pd.DataFrame, symbol: str, timeframe: str):
        cur = self.conn.cursor()
        for idx, row in df.iterrows():
            cur.execute("""
            INSERT OR REPLACE INTO candles(datetime, symbol, timeframe, open, high, low, close, volume)
            VALUES(?,?,?,?,?,?,?,?)
            """, (idx.strftime("%Y-%m-%d %H:%M:%S"), symbol, timeframe,
                  float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])))
        self.conn.commit()

    def insert_order_book(self, datetime_str: str, symbol: str, bids: list, asks: list):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO order_book(datetime, symbol, bid_prices, bid_volumes, ask_prices, ask_volumes)
        VALUES(?,?,?,?,?,?)
        """, (datetime_str, symbol,
              json.dumps([b[0] for b in bids]), json.dumps([b[1] for b in bids]),
              json.dumps([a[0] for a in asks]), json.dumps([a[1] for a in asks])))
        self.conn.commit()

    def insert_features(self, df: pd.DataFrame, symbol: str, timeframe: str):
        cur = self.conn.cursor()
        for idx, row in df.iterrows():
            data_json = row.to_json()
            cur.execute("""
            INSERT OR REPLACE INTO features(datetime, symbol, timeframe, data)
            VALUES(?,?,?,?)
            """, (idx.strftime("%Y-%m-%d %H:%M:%S"), symbol, timeframe, data_json))
        self.conn.commit()

    def insert_model(self, name, model_type, model_hash, path, accuracy, config_snapshot, is_ensemble=0, ensemble_members=""):
        import datetime as dt
        cur = self.conn.cursor()
        created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
        INSERT INTO models(name, model_type, created_at, hash, path, accuracy, config_snapshot, is_ensemble, ensemble_members)
        VALUES(?,?,?,?,?,?,?,?,?)
        """, (name, model_type, created_at, model_hash, path, accuracy, config_snapshot, is_ensemble, ensemble_members))
        self.conn.commit()

    def fetch_models(self) -> list:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM models")
        rows = cur.fetchall()
        return [tuple(r) for r in rows]

    def insert_live_trade(self, entry_time, exit_time, symbol, side, entry_price, exit_price, volume, profit, model_name, strategy_name, risk_used):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO live_trades(entry_time, exit_time, symbol, side, entry_price, exit_price, volume, profit, model_name, strategy_name, risk_used)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, (entry_time, exit_time, symbol, side, entry_price, exit_price, volume, profit, model_name, strategy_name, risk_used))
        self.conn.commit()

    def fetch_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT datetime, open, high, low, close, volume FROM candles
        WHERE symbol=? AND timeframe=? ORDER BY datetime
        """, (symbol, timeframe))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["datetime","Open","High","Low","Close","Volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        return df

    def fetch_order_book(self, symbol: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT datetime, bid_prices, bid_volumes, ask_prices, ask_volumes FROM order_book
        WHERE symbol=? ORDER BY datetime
        """, (symbol,))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        data = []
        for r in rows:
            bids = json.loads(r["bid_prices"])
            bid_vols = json.loads(r["bid_volumes"])
            asks = json.loads(r["ask_prices"])
            ask_vols = json.loads(r["ask_volumes"])
            data.append({
                "datetime": r["datetime"],
                "bid_prices": bids,
                "bid_volumes": bid_vols,
                "ask_prices": asks,
                "ask_volumes": ask_vols
            })
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        return df

    def fetch_features(self, symbol: str, timeframe: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute("""
        SELECT datetime, data FROM features
        WHERE symbol=? AND timeframe=? ORDER BY datetime
        """, (symbol, timeframe))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        data = []
        idx = []
        for r in rows:
            idx.append(pd.to_datetime(r["datetime"]))
            data.append(json.loads(r["data"]))
        df = pd.DataFrame(data, index=idx)
        return df

    def fetch_live_trades(self, limit: int = 100) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute(f"""
        SELECT * FROM live_trades ORDER BY id DESC LIMIT {limit}
        """)
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(row) for row in rows])
        return df

    def insert_risk_params(self, risk_cfg: dict):
        import datetime as dt
        cur = self.conn.cursor()
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
        INSERT INTO risk_params(timestamp, mode, size_per_trade, risk_percent, stop_loss_pips, take_profit_pips, max_daily_trades, trading_start_hour, trading_end_hour)
        VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            timestamp,
            risk_cfg.get("mode"),
            risk_cfg.get("size_per_trade"),
            risk_cfg.get("risk_percent"),
            risk_cfg.get("stop_loss_pips"),
            risk_cfg.get("take_profit_pips"),
            risk_cfg.get("max_daily_trades"),
            risk_cfg["trading_time"].get("start_hour"),
            risk_cfg["trading_time"].get("end_hour")
        ))
        self.conn.commit()

    def insert_strategy(self, name: str, params: dict):
        cur = self.conn.cursor()
        params_json = json.dumps(params)
        cur.execute("""
        INSERT INTO strategies(name, params) VALUES(?, ?)
        """, (name, params_json))
        self.conn.commit()
