# plugins/strategies/simple_strategy.py

plugin_name = "SimpleStrategy"

def init_params() -> dict:
    return {
        "ind1": {"type": "select", "options": ["PSAR", "OrderBookSpread"], "default": "PSAR"},
        "ind2": {"type": "select", "options": ["ADX", "PSAR"], "default": "ADX"},
        "logic": {"type": "select", "options": ["AND", "OR"], "default": "AND"}
    }

def generate_signal(df, params):
    import pandas as pd
    ind1 = params.get("ind1")
    ind2 = params.get("ind2")
    logic = params.get("logic", "AND")
    s1 = df.get(ind1, pd.Series([0]*len(df), index=df.index))
    s2 = df.get(ind2, pd.Series([0]*len(df), index=df.index))
    if logic == "AND":
        return ((s1 > 0) & (s2 > 0)).astype(int)
    else:
        return ((s1 > 0) | (s2 > 0)).astype(int)
