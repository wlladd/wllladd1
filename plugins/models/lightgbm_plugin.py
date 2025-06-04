# plugins/models/lightgbm_plugin.py

plugin_name = "LightGBM"

def init_params() -> dict:
    return {
        "num_leaves": {"type": "int", "default": 31},
        "learning_rate": {"type": "float", "default": 0.05},
        "n_estimators": {"type": "int", "default": 100}
    }

def build_model(params: dict):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        num_leaves=int(params.get("num_leaves", 31)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 100))
    )
