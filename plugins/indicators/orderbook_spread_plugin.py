# plugins/indicators/orderbook_spread_plugin.py

plugin_name = "OrderBookSpread"

def init_params() -> dict:
    return {
        "threshold": {"type": "float", "default": 0.0001},
        "use_relative": {"type": "select", "options": ["absolute", "percent"], "default": "absolute"}
    }

def calculate(df, params):
    import numpy as np
    use_relative = params.get("use_relative", "absolute")
    threshold = float(params.get("threshold", 0.0001))

    def compute_spread(row):
        bids = row.get("bid_prices", [])
        asks = row.get("ask_prices", [])
        if not bids or not asks:
            return -1
        best_bid = bids[0]
        best_ask = asks[0]
        if use_relative == "percent":
            spr = (best_ask - best_bid) / best_bid
        else:
            spr = best_ask - best_bid
        return 1 if spr <= threshold else -1

    return df.apply(compute_spread, axis=1)
