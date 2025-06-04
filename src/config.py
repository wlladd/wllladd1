# src/config.py

import yaml
import json
import os

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Проверки
    mode = cfg["risk_management"]["mode"]
    if mode not in {"fixed_size", "percent_of_balance"}:
        raise ValueError("risk_management.mode должно быть fixed_size или percent_of_balance")
    if mode == "percent_of_balance":
        rp = cfg["risk_management"]["risk_percent"]
        if not (0 < rp < 100):
            raise ValueError("risk_percent должен быть в диапазоне (0, 100)")
    start_h = cfg["risk_management"]["trading_time"]["start_hour"]
    end_h = cfg["risk_management"]["trading_time"]["end_hour"]
    if not (0 <= start_h < 24 and 0 <= end_h < 24 and start_h < end_h):
        raise ValueError("trading_time.start_hour и end_hour должны быть в [0,23] и start < end")
    logic = cfg["live_trading"]["combine_signal_logic"]
    if logic not in {"AND", "OR", "custom"}:
        raise ValueError("live_trading.combine_signal_logic должно быть AND, OR или custom")
    return cfg

def save_plugins_config(plugins_cfg: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(plugins_cfg, f, ensure_ascii=False, indent=2)

def load_plugins_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
