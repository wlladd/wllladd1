# plugins/indicators/adx_plugin.py

plugin_name = "ADX"

def init_params() -> dict:
    return {
        "period": {"type": "int", "default": 14}
    }

def calculate(df, params):
    import pandas_ta as ta
    period = int(params.get("period", 14))
    adx_di = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=period)
    adx_di = adx_di.rename(columns={f"DMP_{period}": "PlusDI", f"DMN_{period}": "MinusDI", f"ADX_{period}": "ADX"})
    # Возвращаем сам ADX (числовой)
    return adx_di["ADX"]
