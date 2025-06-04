# src/data_fetcher.py

import requests
import pandas as pd
import datetime as dt
import time
import hmac
import hashlib
import urllib.parse

class DataFetcher:
    def __init__(self, base_url: str, api_key: str, api_secret: str, db_manager=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.db = db_manager

    def _signed_request(self, method: str, endpoint: str, params: dict = None):
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params, doseq=True)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}{endpoint}"
        if method == 'GET':
            return requests.get(url, params=params, headers=headers)
        elif method == 'POST':
            return requests.post(url, params=params, headers=headers)
        elif method == 'DELETE':
            return requests.delete(url, params=params, headers=headers)
        else:
            raise ValueError('Unsupported method')

    def _public_get(self, endpoint: str, params: dict = None):
        url = f"{self.base_url}{endpoint}"
        return requests.get(url, params=params)

    def get_historical_data(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        start_ts = int(pd.to_datetime(start).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end).timestamp() * 1000)
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        resp = self._public_get("/api/v2/klines", params=params)
        resp.raise_for_status()
        data = resp.json()
        records = []
        for bar in data:
            ts = dt.datetime.fromtimestamp(bar[0] / 1000)
            records.append({
                "Datetime": ts,
                "Open": float(bar[1]),
                "High": float(bar[2]),
                "Low": float(bar[3]),
                "Close": float(bar[4]),
                "Volume": float(bar[5])
            })
        df = pd.DataFrame(records).set_index("Datetime").sort_index()
        if self.db:
            self.db.insert_candles(df, symbol, timeframe)
        return df

    def get_order_book(self, symbol: str, depth: int = 5) -> dict:
        params = {"symbol": symbol, "limit": depth}
        resp = self._public_get("/api/v2/depth", params=params)
        resp.raise_for_status()
        data = resp.json()
        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if self.db:
            self.db.insert_order_book(ts, symbol, bids, asks)
        return {"datetime": ts, "bids": bids, "asks": asks}

    def get_realtime_data(self, symbol: str, timeframe: str) -> dict:
        params = {"symbol": symbol, "interval": timeframe, "limit": 1}
        resp = self._public_get("/api/v2/klines", params=params)
        resp.raise_for_status()
        data = resp.json()
        bar = data[0]
        ts = dt.datetime.fromtimestamp(bar[0] / 1000)
        return {
            "Open": float(bar[1]),
            "High": float(bar[2]),
            "Low": float(bar[3]),
            "Close": float(bar[4]),
            "Volume": float(bar[5]),
            "Datetime": ts
        }

    def place_order(self, symbol: str, side: str, volume: float) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": volume
        }
        resp = self._signed_request("POST", "/api/v2/order", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_open_orders(self) -> list:
        resp = self._signed_request("GET", "/api/v2/openOrders")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("orders", [])

    def get_historical_trades(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        start_ts = int(pd.to_datetime(start).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end).timestamp() * 1000)
        params = {
            "symbol": symbol,
            "startTime": start_ts,
            "endTime": end_ts
        }
        resp = self._signed_request("GET", "/api/v2/myTrades", params=params)
        resp.raise_for_status()
        data = resp.json()
        trades = data if isinstance(data, list) else data.get("trades", [])
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["time"], unit='ms', errors='coerce').fillna(pd.to_datetime(df.get("timestamp"), unit='s'))
        return df

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        params = {"symbol": symbol, "orderId": order_id}
        resp = self._signed_request("DELETE", "/api/v2/order", params=params)
        resp.raise_for_status()
        return resp.json()
