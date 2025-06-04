# src/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import json
import os
import hashlib
import datetime
import importlib.util
import threading
import time
import logging
import pickle

from config import load_config, save_plugins_config, load_plugins_config
from db_manager import DBManager
from data_fetcher import DataFetcher
from indicators_manager import IndicatorManager
from feature_generator import FeatureGenerator
from simulator import BacktestSimulator
from model_manager import ModelManager
from live_trader import LiveTrader
from telegram_notifier import TelegramNotifier

# --- 0. Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- 1. Загрузка config ---
cfg = load_config("config/config.yaml")
plugins_cfg = load_plugins_config("config/plugins_config.json")

db = DBManager(cfg.get("data", {}).get("db_path", "data/project.db"))
data_fetcher = DataFetcher(
    cfg.get("api", {}).get("base_url", ""),
    cfg.get("api", {}).get("api_key", ""),
    cfg.get("api", {}).get("api_secret", ""),
    db
)
ind_manager = IndicatorManager()
model_manager = ModelManager(db_manager=db)
notifier = TelegramNotifier(
    cfg.get("telegram", {}).get("token", ""),
    cfg.get("telegram", {}).get("chat_id", "")
)

# --- 2. Sidebar ---
st.sidebar.title("Настройки проекта")

# --- 2.1 API & DB ---
st.sidebar.subheader("API и БД")
api_url = st.sidebar.text_input(
    "Base URL", cfg.get("api", {}).get("base_url", "")
)
api_key = st.sidebar.text_input(
    "API Key", cfg.get("api", {}).get("api_key", "")
)
api_secret = st.sidebar.text_input(
    "API Secret", cfg.get("api", {}).get("api_secret", "")
)
db_path = st.sidebar.text_input(
    "Путь к SQLite", cfg.get("data", {}).get("db_path", "data/project.db")
)

if st.sidebar.button("Сохранить настройки"):
    cfg.setdefault("api", {})["base_url"] = api_url
    cfg["api"]["api_key"] = api_key
    cfg["api"]["api_secret"] = api_secret
    cfg.setdefault("data", {})["db_path"] = db_path
    with open("config/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    st.sidebar.success("Настройки API/DB сохранены. Перезапустите приложение.")
    st.experimental_rerun()

if st.sidebar.button("Проверить API"):
    try:
        _ = data_fetcher.get_historical_data(
            symbol="EURUSD", timeframe="M1", start="2025-01-01", end="2025-01-02"
        )
        st.sidebar.success("API доступен, данные получены.")
    except Exception as e:
        st.sidebar.error(f"Ошибка: {e}")

st.sidebar.markdown("---")

# --- 2.2 Risk-менеджмент ---
st.sidebar.subheader("Риск-менеджмент")
risk_cfg = cfg.setdefault("risk_management", {})

risk_mode = st.sidebar.selectbox(
    "Режим расчёта объёма",
    options=["fixed_size", "percent_of_balance"],
    index=["fixed_size", "percent_of_balance"].index(
        risk_cfg.get("mode", "fixed_size")
    )
)

if risk_mode == "fixed_size":
    size_per_trade = st.sidebar.number_input(
        "Объём (LOT) на сделку",
        value=risk_cfg.get("size_per_trade", 0.1),
        format="%f"
    )
    risk_percent = st.sidebar.number_input(
        "—",
        value=0.0,
        format="%f",
        disabled=True
    )
else:
    size_per_trade = st.sidebar.number_input(
        "—",
        value=0.0,
        format="%f",
        disabled=True
    )
    risk_percent = st.sidebar.number_input(
        "Риск (% от баланса)",
        value=risk_cfg.get("risk_percent", 1.0),
        format="%f"
    )

stop_loss_pips = st.sidebar.number_input(
    "Stop Loss (pips)",
    value=risk_cfg.get("stop_loss_pips", 20),
    format="%d"
)
take_profit_pips = st.sidebar.number_input(
    "Take Profit (pips)",
    value=risk_cfg.get("take_profit_pips", 50),
    format="%d"
)
max_daily_trades = st.sidebar.number_input(
    "Макс. сделок в день",
    value=risk_cfg.get("max_daily_trades", 5),
    format="%d"
)

st.sidebar.markdown("**Время торговли (часа)**")
col_r1, col_r2 = st.sidebar.columns(2)
with col_r1:
    start_hour = st.sidebar.number_input(
        "С",
        min_value=0, max_value=23,
        value=risk_cfg.get("trading_time", {}).get("start_hour", 0),
        format="%d"
    )
with col_r2:
    end_hour = st.sidebar.number_input(
        "По",
        min_value=0, max_value=23,
        value=risk_cfg.get("trading_time", {}).get("end_hour", 23),
        format="%d"
    )

# Обновляем cfg["risk_management"]
risk_cfg["mode"] = risk_mode
risk_cfg["size_per_trade"] = float(size_per_trade)
risk_cfg["risk_percent"] = float(risk_percent)
risk_cfg["stop_loss_pips"] = int(stop_loss_pips)
risk_cfg["take_profit_pips"] = int(take_profit_pips)
risk_cfg["max_daily_trades"] = int(max_daily_trades)
risk_cfg.setdefault("trading_time", {})
risk_cfg["trading_time"]["start_hour"] = int(start_hour)
risk_cfg["trading_time"]["end_hour"] = int(end_hour)

# Сохраняем параметры риск-менеджмента в БД
db.insert_risk_params(risk_cfg)
st.sidebar.markdown("---")

# --- 2.3 Индикаторы ---
st.sidebar.subheader("Плагины: Индикаторы")
all_indicators = ind_manager.get_plugin_names()
selected_inds = st.sidebar.multiselect(
    "Выберите индикаторы",
    all_indicators,
    default=plugins_cfg.get("indicators", [])
)
ind_params = {}
for ind in selected_inds:
    st.sidebar.markdown(f"**{ind}**")
    params = ind_manager.get_plugin_params(ind)
    ind_params[ind] = {}
    for key, opt in params.items():
        if opt["type"] == "int":
            ind_params[ind][key] = st.sidebar.number_input(
                f"{ind} → {key}",
                value=opt["default"],
                step=1,
                format="%d"
            )
        elif opt["type"] == "float":
            ind_params[ind][key] = st.sidebar.number_input(
                f"{ind} → {key}",
                value=opt["default"],
                format="%f"
            )
        elif opt["type"] == "select":
            ind_params[ind][key] = st.sidebar.selectbox(
                f"{ind} → {key}",
                options=opt["options"],
                index=opt["options"].index(opt["default"])
            )
        else:
            ind_params[ind][key] = st.sidebar.text_input(
                f"{ind} → {key}",
                value=str(opt["default"])
            )

st.sidebar.markdown("---")

# --- 2.4 Модели ---
st.sidebar.subheader("Плагины: Модели")
all_models = model_manager.get_plugin_names()
selected_models = st.sidebar.multiselect(
    "Выберите модели (можно несколько для ансамбля)",
    all_models,
    default=plugins_cfg.get("models", [])
)
model_params = {}
for mdl in selected_models:
    st.sidebar.markdown(f"**{mdl}**")
    params = model_manager.get_plugin_params(mdl)
    model_params[mdl] = {}
    for key, opt in params.items():
        if opt["type"] == "int":
            model_params[mdl][key] = st.sidebar.number_input(
                f"{mdl} → {key}",
                value=opt["default"],
                step=1,
                format="%d"
            )
        elif opt["type"] == "float":
            model_params[mdl][key] = st.sidebar.number_input(
                f"{mdl} → {key}",
                value=opt["default"],
                format="%f"
            )
        elif opt["type"] == "select":
            model_params[mdl][key] = st.sidebar.selectbox(
                f"{mdl} → {key}",
                options=opt["options"],
                index=opt["options"].index(opt["default"])
            )
        else:
            model_params[mdl][key] = st.sidebar.text_input(
                f"{mdl} → {key}",
                value=str(opt["default"])
            )

st.sidebar.markdown("---")

# --- 2.5 Стратегии ---
st.sidebar.subheader("Плагины: Стратегии (Rule-Based)")
all_strategies = []
strategies_path = "plugins/strategies"
if os.path.isdir(strategies_path):
    for fname in os.listdir(strategies_path):
        if fname.endswith(".py") and fname != "__init__.py":
            spec = importlib.util.spec_from_file_location(
                fname[:-3], os.path.join(strategies_path, fname)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            name = getattr(module, "plugin_name", None)
            if name:
                all_strategies.append(name)

selected_strategies = st.sidebar.multiselect(
    "Выберите стратегии",
    all_strategies,
    default=plugins_cfg.get("strategies", [])
)
strategies_params = {}
for strat in selected_strategies:
    st.sidebar.markdown(f"**{strat}**")
    module_path = os.path.join(strategies_path, strat.lower() + ".py")
    spec = importlib.util.spec_from_file_location(strat, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = module.init_params()
    strategies_params[strat] = {}
    for key, opt in params.items():
        if opt["type"] == "int":
            strategies_params[strat][key] = st.sidebar.number_input(
                f"{strat} → {key}",
                value=opt["default"],
                step=1,
                format="%d"
            )
        elif opt["type"] == "float":
            strategies_params[strat][key] = st.sidebar.number_input(
                f"{strat} → {key}",
                value=opt["default"],
                format="%f"
            )
        elif opt["type"] == "select":
            strategies_params[strat][key] = st.sidebar.selectbox(
                f"{strat} → {key}",
                options=opt["options"],
                index=opt["options"].index(opt["default"])
            )
        else:
            strategies_params[strat][key] = st.sidebar.text_input(
                f"{strat} → {key}",
                value=str(opt["default"])
            )

# Сохраняем стратегии в БД
for strat in selected_strategies:
    db.insert_strategy(strat, strategies_params[strat])

st.sidebar.markdown("---")

# --- 2.6 Дефолты fetch/симуляция/бэктест ---
st.sidebar.subheader("Дефолты: Данные/Симуляция/Бэктест")

# Настройки fetch
default_fetch = cfg.setdefault("default_fetch", {})
symbol = st.sidebar.text_input(
    "Symbol", default_fetch.get("symbol", "EURUSD")
)
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["M1", "M5", "M15", "H1", "H4", "D1"],
    index=["M1", "M5", "M15", "H1", "H4", "D1"].index(
        default_fetch.get("timeframe", "M1")
    )
)
start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime(default_fetch.get("start_date", "2025-01-01")).date()
)
end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime(default_fetch.get("end_date", "2025-01-02")).date()
)
collect_ob = st.sidebar.checkbox(
    "Собирать стакан (order book)?",
    default_fetch.get("collect_order_book", False)
)
order_book_depth = st.sidebar.selectbox(
    "Depth",
    [5, 10, 20, 50],
    index=[5, 10, 20, 50].index(default_fetch.get("order_book_depth", 5))
)

# Настройки симуляции
simulation_cfg = cfg.setdefault("simulation", {})
# Даем значения по умолчанию, если раздел отсутствует

init_balance = st.sidebar.number_input(
    "Initial Balance",
    value=simulation_cfg.get("initial_balance", 1000.0),
    format="%f"
)
lot_size = st.sidebar.number_input(
    "Lot Size",
    value=simulation_cfg.get("lot_size", 0.1),
    format="%f"
)
stop_loss_pips_s = st.sidebar.number_input(
    "StopLoss (pips)",
    value=simulation_cfg.get("stop_loss_pips", 20),
    format="%d"
)
take_profit_pips_s = st.sidebar.number_input(
    "TakeProfit (pips)",
    value=simulation_cfg.get("take_profit_pips", 50),
    format="%d"
)
label_N = st.sidebar.number_input(
    "N баров для меток",
    value=simulation_cfg.get("label_N", 10),
    min_value=1,
    step=1
)
# Сохраняем обратно в cfg["simulation"]
simulation_cfg["initial_balance"] = float(init_balance)
simulation_cfg["lot_size"] = float(lot_size)
simulation_cfg["stop_loss_pips"] = int(stop_loss_pips_s)
simulation_cfg["take_profit_pips"] = int(take_profit_pips_s)
simulation_cfg["label_N"] = int(label_N)

st.sidebar.markdown("---")
st.sidebar.subheader("Параметры обучения")

training_cfg = cfg.setdefault("training", {})
test_split = st.sidebar.slider(
    "Test Split (%)",
    min_value=0.1, max_value=0.5,
    value=cfg.get("training", {}).get("test_split", 0.2),
    step=0.05
)
use_ensemble = st.sidebar.checkbox(
    "Использовать ансамбль моделей?",
    value=cfg.get("training", {}).get("use_ensemble", False)
)
if use_ensemble:
    ensemble_list = st.sidebar.multiselect(
        "Выберите модели для ансамбля",
        selected_models,
        default=plugins_cfg.get("training", {}).get("ensemble_models", [])
    )
    epochs = st.sidebar.number_input(
        "Epochs",
        value=cfg.get("training", {}).get("epochs", 10),
        step=1,
        format="%d"
    )
    batch_size = st.sidebar.number_input(
        "Batch Size",
        value=cfg.get("training", {}).get("batch_size", 32),
        step=1,
        format="%d"
    )
else:
    ensemble_list = []
    epochs = None
    batch_size = None

# Обновляем cfg["training"]
training_cfg["test_split"] = float(test_split)
training_cfg["use_ensemble"] = bool(use_ensemble)
training_cfg["ensemble_models"] = ensemble_list
if epochs is not None:
    training_cfg["epochs"] = int(epochs)
if batch_size is not None:
    training_cfg["batch_size"] = int(batch_size)

st.sidebar.markdown("---")
st.sidebar.subheader("Live Trading")

live_cfg = cfg.setdefault("live_trading", {})
live_enabled = st.sidebar.checkbox(
    "Включить реальную торговлю",
    value=live_cfg.get("enabled", False)
)
live_interval = st.sidebar.number_input(
    "Polling Interval (sec)",
    value=live_cfg.get("polling_interval_sec", 60),
    step=1,
    format="%d"
)
combine_logic = st.sidebar.selectbox(
    "Логика объединения сигналов моделей",
    ["AND", "OR", "custom"],
    index=["AND", "OR", "custom"].index(
        live_cfg.get("combine_signal_logic", "AND")
    )
)
manual_override = st.sidebar.checkbox(
    "Полуавтоматический режим (ручная торговля)",
    value=live_cfg.get("manual_override", False)
)
custom_logic = live_cfg.get("custom_logic", "")
if combine_logic == "custom":
    custom_logic = st.sidebar.text_input(
        "Custom logic (пример: (m1 AND m2) OR m3)",
        value=custom_logic
    )
    live_cfg["custom_logic"] = custom_logic

# Обновляем cfg["live_trading"]
live_cfg["enabled"] = bool(live_enabled)
live_cfg["polling_interval_sec"] = int(live_interval)
live_cfg["combine_signal_logic"] = combine_logic
live_cfg["manual_override"] = bool(manual_override)
live_cfg["custom_logic"] = custom_logic

# Сохраняем выбор плагинов в plugins_config.json
plugins_cfg["indicators"] = selected_inds
plugins_cfg["indicators_params"] = ind_params
plugins_cfg["models"] = selected_models
plugins_cfg["models_params"] = model_params
plugins_cfg["strategies"] = selected_strategies
plugins_cfg["strategies_params"] = strategies_params
plugins_cfg["default_fetch"] = {
    "symbol": symbol,
    "timeframe": timeframe,
    "start_date": str(start_date),
    "end_date": str(end_date),
    "collect_order_book": collect_ob,
    "order_book_depth": order_book_depth
}
plugins_cfg["simulation"] = {
    "initial_balance": simulation_cfg["initial_balance"],
    "lot_size": simulation_cfg["lot_size"],
    "stop_loss_pips": simulation_cfg["stop_loss_pips"],
    "take_profit_pips": simulation_cfg["take_profit_pips"],
    "label_N": simulation_cfg["label_N"]
}
plugins_cfg["training"] = {
    "test_split": training_cfg["test_split"],
    "use_ensemble": training_cfg["use_ensemble"],
    "ensemble_models": training_cfg.get("ensemble_models", []),
    "epochs": training_cfg.get("epochs", None),
    "batch_size": training_cfg.get("batch_size", None)
}
plugins_cfg["live_trading"] = {
    "enabled": live_cfg["enabled"],
    "polling_interval_sec": live_cfg["polling_interval_sec"],
    "combine_signal_logic": live_cfg["combine_signal_logic"],
    "manual_override": live_cfg["manual_override"],
    "custom_logic": live_cfg["custom_logic"]
}
save_plugins_config(plugins_cfg, "config/plugins_config.json")

# --- 3. Main Page (Tabs) ---
st.title("Веб-дашборд: Алгоритмический трейдинг с Нейросетью")
tabs = st.tabs([
    "Сбор данных",
    "Генерация признаков",
    "Бэктест & Обучение",
    "Live Трейд",
    "История & Отчёты",
    "Логи"
])

# 3.1. Сбор данных
with tabs[0]:
    st.header("Сбор исторических данных")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Symbol:** `{symbol}`")
        st.write(f"**Timeframe:** `{timeframe}`")
    with col2:
        st.write(f"**Period:** `{start_date}` → `{end_date}`")
    with col3:
        if collect_ob:
            st.write(f"**Собирать стакан:** Да, depth = {order_book_depth}")
        else:
            st.write("**Собирать стакан:** Нет")

    if st.button("Скачать и сохранить данные"):
        with st.spinner("Идёт загрузка свечей..."):
            try:
                df_hist = data_fetcher.get_historical_data(
                    symbol, timeframe, str(start_date), str(end_date)
                )
                st.success(f"Свечи загружены: {len(df_hist)} баров.")
            except Exception as e:
                st.error(f"Ошибка при загрузке свечей: {e}")
                df_hist = pd.DataFrame()

        if collect_ob and not df_hist.empty:
            st.info("Сбор стакана...")
            pb = st.progress(0)
            total = len(df_hist)
            for idx, ts in enumerate(df_hist.index):
                try:
                    data_fetcher.get_order_book(symbol, depth=order_book_depth)
                except:
                    pass
                pb.progress((idx + 1) / total)
            st.success("Стакан загружен и сохранён.")

    st.subheader("Предпросмотр последних свечей")
    df_preview = db.fetch_candles(symbol, timeframe).tail(5)
    if not df_preview.empty:
        st.dataframe(df_preview)
    else:
        st.write("Нет данных свечей для отображения.")

# 3.2. Генерация признаков
with tabs[1]:
    st.header("Генерация признаков")
    st.write(f"**Symbol:** `{symbol}`   |   **Timeframe:** `{timeframe}`")
    st.write(
        f"**Индикаторы:** {', '.join(selected_inds) if selected_inds else '—'}"
    )

    if st.button("Сгенерировать признаки"):
        with st.spinner("Генерация признаков..."):
            try:
                fg = FeatureGenerator(cfg, db)
                df_feat = fg.generate(
                    symbol, timeframe, selected_inds, ind_params,
                    include_order_book=collect_ob
                )
                st.success(f"Признаки сгенерированы: {df_feat.shape[0]} строк.")
                st.subheader("Первые 5 строк признаков")
                st.dataframe(df_feat.head())
            except Exception as e:
                st.error(f"Ошибка генерации признаков: {e}")

# 3.3. Бэктест & Обучение
with tabs[2]:
    st.header("Бэктест и обучение модели")
    st.write("**Риск-менеджмент (симуляция)**")
    st.write(f"- Режим: `{risk_mode}`")
    if risk_mode == "fixed_size":
        st.write(f"- Размер лота: {size_per_trade:.3f} LOT")
    else:
        st.write(f"- Risk: {risk_percent:.2f}% от баланса")
    st.write(f"- Stop Loss (pips): {stop_loss_pips_s}")
    st.write(f"- Take Profit (pips): {take_profit_pips_s}")
    st.write(f"- Макс. сделок в день: {max_daily_trades}")
    st.write(f"- Время торговли: {start_hour}:00 → {end_hour}:00")

    if selected_strategies:
        st.write("**Rule-Based стратегии:**")
        st.write(", ".join(selected_strategies))
    else:
        st.write("**Rule-Based стратегии:** отсутствуют")

    st.write("**Модели для обучения:**")
    if selected_models:
        st.write(", ".join(selected_models))
    else:
        st.write("—")

    if use_ensemble:
        st.write("**Ансамбль:**")
        st.write(f"- Использовать ансамбль: Да")
        st.write(f"- Состав ансамбля: {', '.join(ensemble_list)}")
    else:
        st.write("- Использовать ансамбль: Нет")

    st.write(f"**Test Split:** {test_split:.2f}")

    if st.button("Запустить бэктест и обучение"):
        with st.spinner("Идёт загрузка признаков и запуск бэктеста..."):
            try:
                df_features = db.fetch_features(symbol, timeframe)
                if df_features.empty:
                    st.error(
                        "Нет признаков для выбранного symbol/timeframe. "
                        "Сначала сгенерируйте их."
                    )
                else:
                    st.write(
                        f"Признаки загружены: "
                        f"{df_features.shape[0]} строк, {df_features.shape[1]} колонок."
                    )

                    simulator = BacktestSimulator(
                        df_features=df_features,
                        cfg=cfg,
                        models=[],
                        risk_cfg=risk_cfg,
                        strategy_logic=combine_logic
                    )
                    df_with_labels = simulator.generate_labels(
                        N=int(label_N)
                    )
                    st.write(f"Метки созданы: {df_with_labels.shape[0]} строк.")

                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler

                    X = df_with_labels.drop(
                        columns=["Target_Long", "Target_Short", "Open_next", "Future_Close_N"]
                    )
                    y = df_with_labels["Target_Long"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(test_split), shuffle=False
                    )
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    mm = ModelManager(db_manager=db)
                    config_snapshot = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "indicators": selected_inds,
                        "indicators_params": ind_params,
                        "risk_management": risk_cfg,
                        "training": {
                            "models": selected_models,
                            "model_params": model_params,
                            "test_split": test_split,
                            "use_ensemble": use_ensemble,
                            "ensemble_models": ensemble_list,
                            "epochs": epochs,
                            "batch_size": batch_size
                        },
                        "simulation": {
                            "initial_balance": simulation_cfg["initial_balance"],
                            "lot_size": simulation_cfg["lot_size"],
                            "stop_loss_pips": simulation_cfg["stop_loss_pips"],
                            "take_profit_pips": simulation_cfg["take_profit_pips"]
                        }
                    }

                    # Сохранение scaler
                    sha = hashlib.sha256(
                        json.dumps(config_snapshot, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:8]
                    scaler_filename = f"scaler_{sha}.pkl"
                    os.makedirs("models", exist_ok=True)
                    scaler_path = os.path.join("models", scaler_filename)
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scaler, f)

                    model_paths = []

                    if not use_ensemble and selected_models:
                        base_name = selected_models[0]
                        fn, acc, path = mm.train_and_save(
                            name=base_name,
                            params=model_params,
                            X_train=X_train_scaled,
                            y_train=y_train.values,
                            X_val=X_test_scaled,
                            y_val=y_test.values,
                            config_snapshot=config_snapshot,
                            ensemble=False
                        )
                        model_paths.append(path)
                        st.success(f"Модель `{fn}` сохранена (Accuracy={acc:.4f}).")
                    elif use_ensemble and ensemble_list:
                        fn, acc, path = mm.train_and_save(
                            name=None,
                            params=model_params,
                            X_train=X_train_scaled,
                            y_train=y_train.values,
                            X_val=X_test_scaled,
                            y_val=y_test.values,
                            config_snapshot=config_snapshot,
                            ensemble=True,
                            ensemble_names=ensemble_list
                        )
                        model_paths.append(path)
                        st.success(f"Ансамбль `{fn}` сохранён (Accuracy={acc:.4f}).")
                    else:
                        st.warning("Не выбраны модели для обучения.")

                    model_objs = [mm.load_model(p) for p in model_paths]
                    simulator.models = model_objs
                    simulator.model_names = (
                        ensemble_list if use_ensemble
                        else ([selected_models[0]] if selected_models else [])
                    )
                    results = simulator.run_backtest()

                    st.subheader("Результаты симуляции")
                    st.write(f"- Total Trades: {results['total_trades']}")
                    st.write(f"- Win Rate (%): {results['win_rate']:.2f}")
                    st.write(f"- Profit Factor: {results['profit_factor']:.2f}")
                    st.write(f"- Final Balance: {results['final_balance']:.2f}")

                    eq_df = results["equity"].set_index("time")
                    st.line_chart(eq_df["balance"])
            except Exception as e:
                st.error(f"Ошибка во время бэктеста/обучения: {e}")

# 3.4. Live Трейд
with tabs[3]:
    st.header("Живая торговля")
    last_lt = db.fetch_live_trades(limit=1)
    if not last_lt.empty:
        try:
            curr_balance = (
                last_lt.iloc[0]["exit_price"] * last_lt.iloc[0]["volume"]
                + last_lt.iloc[0]["profit"]
            )
            st.write(f"**Баланс:** ~{curr_balance:.2f} USD")
        except Exception:
            st.write(f"**Баланс:** {simulation_cfg.get('initial_balance', 0.0):.2f} USD")
    else:
        st.write(f"**Баланс:** {simulation_cfg.get('initial_balance', 0.0):.2f} USD (начальный)")

    st.write("**Сохранённые модели (для live)**")
    models_list = [row[1] for row in db.fetch_models()]
    live_models = st.multiselect(
        "Выберите модель(и) для live (порядок важен)",
        models_list
    )
    if live_enabled:
        st.write("> Режим: Реальная торговля")
    else:
        st.write("> Режим: Симуляция (live отключён)")

    st.write(f"- Логика объединения: `{combine_logic}`")
    if combine_logic == "custom":
        st.write(f"- Custom logic: `{custom_logic}`")

    scaler_files = [
        f for f in os.listdir("models")
        if f.startswith("scaler_") and f.endswith(".pkl")
    ]
    scaler_choice = st.selectbox("Выберите scaler (если есть)", ["Нет"] + scaler_files)
    if scaler_choice != "Нет":
        scaler_path_live = f"models/{scaler_choice}"
    else:
        scaler_path_live = None

    if manual_override:
        st.subheader("Ручная торговля")
        col_buy, col_sell = st.columns(2)
        with col_buy:
            buy_volume = st.number_input(
                "Объём для BUY",
                value=risk_cfg.get("size_per_trade", 0.1),
                format="%f"
            )
            if st.button("Открыть BUY вручную"):
                if live_trader:
                    live_trader.manual_open("BUY", buy_volume, reason="Ручной вход")
                    st.success("Ручной ордер BUY зарегистрирован.")
                else:
                    st.warning("LiveTrader ещё не запущен.")
        with col_sell:
            sell_volume = st.number_input(
                "Объём для SELL",
                value=risk_cfg.get("size_per_trade", 0.1),
                format="%f"
            )
            if st.button("Открыть SELL вручную"):
                if live_trader:
                    live_trader.manual_open("SELL", sell_volume, reason="Ручной выход")
                    st.success("Ручной ордер SELL зарегистрирован.")
                else:
                    st.warning("LiveTrader ещё не запущен.")

        if st.button("Отменить все открытые ордера (ручной выход)"):
            if live_trader:
                live_trader.manual_close_all()
                st.info("Запрос на отмену всех ордеров отправлен.")
            else:
                st.warning("LiveTrader ещё не запущен.")

    start_live = st.button("Запустить Live Trader")
    stop_live = st.button("Остановить Live Trader")
    status_text = st.empty()

    if 'live_trader' not in locals():
        live_trader = None

    # Функция для запуска LiveTrader в фоне
    def run_live_thread():
        global live_trader
        live_trader = LiveTrader(cfg)
        model_paths_list = [f"models/{m}" for m in live_models]
        scaler_path_list = (
            [scaler_path_live] * len(model_paths_list)
            if scaler_path_live else [None] * len(model_paths_list)
        )
        try:
            live_trader.start(
                model_paths_list,
                scaler_path_list
            )
        except Exception as e:
            logging.error(f"Ошибка в LiveTrader: {e}")
            if cfg.get("telegram", {}).get("send_on_error", False):
                notifier.send_error(str(e))

    if start_live:
        if not live_enabled:
            st.warning("Реальная торговля не включена (Live Trading → enabled=False).")
        elif not live_models:
            st.warning("Выберите хотя бы одну модель для live.")
        else:
            status_text.info("Запуск Live Trader в фоне...")
            live_thread = threading.Thread(
                target=run_live_thread,
                daemon=True
            )
            live_thread.start()
            time.sleep(1)
            status_text.success("Live Trader запущен.")

    if stop_live:
        if live_trader and hasattr(live_trader, "is_running") and live_trader.is_running:
            live_trader.stop()
            status_text.warning("Live Trader остановлен.")
        else:
            st.info("Live Trader не запущен или уже остановлен.")

# 3.5. История & Отчёты
with tabs[4]:
    st.header("История моделей и логов сделок")

    st.subheader("Модели (таблица `models`)")
    models = db.fetch_models()
    if models:
        df_models = pd.DataFrame(
            models,
            columns=[
                "ID", "Name", "Type", "Created At", "Hash",
                "Path", "Accuracy", "is_ensemble", "ensemble_members"
            ]
        )
        sel = st.multiselect(
            "Выберите модель(и) для подробностей или сравнения",
            df_models["ID"].tolist()
        )
        st.dataframe(df_models)

        if st.button("Показать детали выбранной модели"):
            if sel:
                for mid in sel:
                    row = next(r for r in models if r[0] == mid)
                    st.markdown(f"**Model ID: {row[0]}**")
                    st.write(f"- Name: {row[1]}")
                    st.write(f"- Type: {row[2]}")
                    st.write(f"- Created At: {row[3]}")
                    st.write(f"- Hash: {row[4]}")
                    st.write(f"- Path: {row[5]}")
                    st.write(f"- Accuracy: {row[6]:.4f}")
                    st.write(f"- is_ensemble: {'Да' if row[7] == 1 else 'Нет'}")
                    st.write(f"- ensemble_members: {row[8]}")
                    st.markdown("---")
            else:
                st.warning("Выберите хотя бы одну модель.")

        if len(sel) == 2 and st.button("Сравнить две модели"):
            st.write("Сравнение двух моделей (equity-кривые)")
            st.info("Пока placeholder — реализовать сравнение equity.")
    else:
        st.write("Таблица моделей пуста.")

    st.markdown("---")

    st.subheader("Логи живых сделок (таблица `live_trades`)")
    df_lt = db.fetch_live_trades(limit=50)
    if not df_lt.empty:
        st.dataframe(df_lt)
        if st.button("Экспорт логов live-trade в CSV"):
            path = f"logs/live_trades_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_lt.to_csv(path, index=False)
            st.success(f"Логи сохранены в {path}")
    else:
        st.write("Нет записей live-трейда.")

# 3.6. Логи
with tabs[5]:
    st.header("Логи приложения")
    if st.button("Обновить логи"):
        try:
            with open("logs/app.log", "r", encoding="utf-8") as f:
                data = f.read()
            st.code(data[-10000:], language="")
        except:
            st.write("Нет файла app.log или ошибка при чтении.")
    if st.button("Очистить логи"):
        open("logs/app.log", "w", encoding="utf-8").close()
        open("logs/error.log", "w", encoding="utf-8").close()
        st.success("Логи очищены.")
