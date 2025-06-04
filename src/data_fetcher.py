# src/data_fetcher.py

import requests
import pandas as pd
import datetime as dt

class DataFetcher:
    def __init__(self, base_url: str, api_key: str, db_manager=None):
        self.base_url = base_url
        self.api_key = api_key
        self.db = db_manager

    def get_historical_data(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        url = f"{self.base_url}/market/history"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "start": start,
            "end": end,
            "apikey": self.api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("bars", [])
        records = []
        for bar in bars:
            ts = dt.datetime.fromtimestamp(bar["timestamp"])
            records.append({
                "Datetime": ts,
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
                "Volume": bar["volume"]
            })
        df = pd.DataFrame(records).set_index("Datetime").sort_index()
        if self.db:
            self.db.insert_candles(df, symbol, timeframe)
        return df

    def get_order_book(self, symbol: str, depth: int = 5) -> dict:
        url = f"{self.base_url}/market/order_book"
        params = {
            "symbol": symbol,
            "depth": depth,
            "apikey": self.api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        ts = dt.datetime.fromtimestamp(data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if self.db:
            self.db.insert_order_book(ts, symbol, bids, asks)
        return {"datetime": ts, "bids": bids, "asks": asks}

    def get_realtime_data(self, symbol: str, timeframe: str) -> dict:
        url = f"{self.base_url}/market/realtime"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "apikey": self.api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        bar = data.get("bar", {})
        ts = dt.datetime.fromtimestamp(bar["timestamp"])
        return {
            "Open": bar["open"],
            "High": bar["high"],
            "Low": bar["low"],
            "Close": bar["close"],
            "Volume": bar["volume"],
            "Datetime": ts
        }

    def place_order(self, symbol: str, side: str, volume: float) -> dict:
        url = f"{self.base_url}/trade/place_order"
        payload = {
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "apikey": self.api_key
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_open_orders(self) -> list:
        url = f"{self.base_url}/trade/open_orders"
        params = {"apikey": self.api_key}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("orders", [])

    def get_historical_trades(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        url = f"{self.base_url}/trade/historical_trades"
        params = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "apikey": self.api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        trades = data.get("trades", [])
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
        return df

    def cancel_order(self, order_id: str) -> dict:
        url = f"{self.base_url}/trade/cancel"
        payload = {"order_id": order_id, "apikey": self.api_key}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
