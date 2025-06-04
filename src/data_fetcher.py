# src/data_fetcher.py

import requests
import pandas as pd
import datetime as dt
import time
import hmac
import hashlib
import urllib.parse

class DataFetcher:
    def __init__(self, base_url: str, api_key: str, api_secret: str,
                 db_manager=None, header_name: str = "X-MBXAPIKEY"):
        """Initialize fetcher with authentication details.

        Parameters
        ----------
        base_url : str
            REST API base url.
        api_key : str
            API key value.
        api_secret : str
            API secret for signing requests.
        db_manager : optional
            Instance managing DB interactions.
        header_name : str, default ``"X-MBXAPIKEY"``
            Name of the HTTP header carrying the API key.  Older versions of
            the API used ``"X-MBX-APIKEY"`` – pass this value for backward
            compatibility.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.db = db_manager
        self.header_name = header_name

    def _signed_request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        header_name: str | None = None,
    ):
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params, doseq=True)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        hdr = header_name or self.header_name
        headers = {hdr: self.api_key}
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
        """Return candles in the given range respecting the 1000 record limit.

        The ``/api/v2/klines`` endpoint returns at most 1000 candles per
        request.  This method calculates how many milliseconds 1000 candles
        cover for the provided ``timeframe`` and splits the ``[start, end]``
        interval into chunks of that size.  Each chunk is requested
        sequentially and the results are concatenated.  If ``self.db`` is set
        the final dataframe is also inserted into the database.
        """

        start_ts = int(pd.to_datetime(start).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end).timestamp() * 1000)

        tf_map = {
            "M1": pd.Timedelta(minutes=1),
            "M5": pd.Timedelta(minutes=5),
            "M15": pd.Timedelta(minutes=15),
            "H1": pd.Timedelta(hours=1),
            "H4": pd.Timedelta(hours=4),
            "D1": pd.Timedelta(days=1),
        }

        if timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        tf_ms = int(tf_map[timeframe].total_seconds() * 1000)
        chunk_ms = tf_ms * 1000

        frames = []
        cur_start = start_ts
        while cur_start <= end_ts:
            cur_end = min(end_ts, cur_start + chunk_ms - tf_ms)
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "startTime": cur_start,
                "endTime": cur_end,
                "limit": 1000,
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
                    "Volume": float(bar[5]),
                })
            if records:
                frames.append(pd.DataFrame(records))
            if not data or len(data) < 1000:
                break
            cur_start = data[-1][0] + tf_ms

        if frames:
            df = pd.concat(frames).set_index("Datetime").sort_index()
        else:
            df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        if self.db and not df.empty:
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
