# plugins/indicators/custom_indicator.py

plugin_name = "CustomIndicator"

def init_params() -> dict:
    return {
        "param1": {"type": "float", "default": 1.0},
        # любые другие параметры
    }

def calculate(df, params):
    # Пример: бинарный сигнал, если Close выше предыдущего High:
    return (df["Close"] > df["High"].shift(1)).astype(int) * 2 - 1
