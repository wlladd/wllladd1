# plugins/models/catboost_plugin.py

plugin_name = "CatBoost"

def init_params() -> dict:
    return {
        "iterations": {"type": "int", "default": 1000},
        "learning_rate": {"type": "float", "default": 0.1},
        "depth": {"type": "int", "default": 6}
    }

def build_model(params: dict):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=int(params.get("iterations", 1000)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        depth=int(params.get("depth", 6)),
        verbose=False
    )
