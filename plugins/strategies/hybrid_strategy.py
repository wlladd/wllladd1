# plugins/strategies/hybrid_strategy.py

plugin_name = "HybridStrategy"

def init_params() -> dict:
    return {
        "model": {"type": "select", "options": ["LightGBM","CatBoost","LSTM"], "default": "LightGBM"},
        "threshold": {"type": "float", "default": 0.6},
        "ind": {"type": "select", "options": ["PSAR","OrderBookSpread"], "default": "PSAR"}
    }

def generate_signal(df, params):
    # Предполагается, что df уже содержит колонку с вероятностью модели "model_prob"
    # и колонку индикатора "ind". Возвращаем 1, если model_prob >= threshold И ind > 0.
    import pandas as pd
    model = params.get("model")
    threshold = float(params.get("threshold", 0.6))
    ind  = params.get("ind")
    prob = df.get(f"{model}_prob", pd.Series([0]*len(df), index=df.index))
    sig_ind = df.get(ind, pd.Series([0]*len(df), index=df.index))
    return (((prob >= threshold) & (sig_ind > 0)).astype(int))
