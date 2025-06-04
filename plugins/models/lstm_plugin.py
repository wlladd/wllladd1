# plugins/models/lstm_plugin.py

plugin_name = "LSTM"

def init_params() -> dict:
    return {
        "units": {"type": "int", "default": 50},
        "dropout": {"type": "float", "default": 0.2},
        "epochs": {"type": "int", "default": 50},
        "batch_size": {"type": "int", "default": 64}
    }

def build_model(params: dict):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    units = int(params.get("units", 50))
    dropout = float(params.get("dropout", 0.2))
    model = Sequential()
    model.add(LSTM(units, input_shape=(None, None)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
