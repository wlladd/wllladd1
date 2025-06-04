# plugins/indicators/psar_plugin.py

plugin_name = "PSAR"

def init_params() -> dict:
    return {
        "step": {"type": "float", "default": 0.02},
        "max_step": {"type": "float", "default": 0.2}
    }

def calculate(df, params):
    import pandas_ta as ta
    step = params.get("step", 0.02)
    max_step = params.get("max_step", 0.2)
    psar = ta.psar(high=df["High"], low=df["Low"], close=df["Close"], step=step, max_step=max_step)
    col = [c for c in psar.columns if c.startswith("PSARl_")][0]
    return psar[col].combine(df["Close"], lambda s, c: 1 if s < c else -1)
