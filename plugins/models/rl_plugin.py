# plugins/models/rl_plugin.py

plugin_name = "RLAgent"

def init_params() -> dict:
    return {
        "learning_rate": {"type": "float", "default": 0.001},
        "gamma": {"type": "float", "default": 0.95},
        "n_steps": {"type": "int", "default": 10000}
    }

def build_model(params: dict):
    # Заглушка RL-архитектуры. Для реальной работы требуется tf_agents или stable-baselines
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    lr = float(params.get("learning_rate", 0.001))
    model = Sequential()
    model.add(Dense(24, activation="relu", input_shape=(None,)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse")
    return model
