# src/model_manager.py

import os
import importlib.util
import hashlib
import json
import pickle
import datetime
from sklearn.ensemble import VotingClassifier

class ModelManager:
    def __init__(self, plugins_path="plugins/models", db_manager=None):
        self.plugins_path = plugins_path
        self.models_plugins = {}
        self._scan_plugins()
        self.db = db_manager

    def _scan_plugins(self):
        for filename in os.listdir(self.plugins_path):
            if filename.endswith(".py") and filename != "__init__.py":
                fullpath = os.path.join(self.plugins_path, filename)
                spec = importlib.util.spec_from_file_location(filename[:-3], fullpath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                name = getattr(module, "plugin_name", None)
                if name:
                    self.models_plugins[name] = module

    def get_plugin_names(self) -> list:
        return list(self.models_plugins.keys())

    def get_plugin_params(self, name: str) -> dict:
        module = self.models_plugins.get(name)
        if not module:
            raise ValueError(f"Модель `{name}` не найдена.")
        return module.init_params()

    def build_model(self, name: str, params: dict):
        module = self.models_plugins.get(name)
        if not module:
            raise ValueError(f"Модель `{name}` не найдена.")
        return module.build_model(params)

    def train_and_save(self, name: str, params: dict, X_train, y_train, X_val=None, y_val=None, config_snapshot: dict=None, ensemble: bool=False, ensemble_names: list=None):
        if not ensemble:
            model = self.build_model(name, params.get(name, {}))
            if X_val is not None and y_val is not None:
                try:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                except:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                preds = model.predict(X_val)
                try:
                    preds = (preds > 0.5).astype(int)
                except:
                    preds = preds.astype(int)
                accuracy = (preds == y_val).mean()
            else:
                accuracy = None

            hash_obj = hashlib.sha256()
            json_str = ""
            if config_snapshot:
                json_str = json.dumps(config_snapshot, sort_keys=True)
                hash_obj.update(json_str.encode("utf-8"))
            model_hash = hash_obj.hexdigest()[:8]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{name}_{ts}_{model_hash}.pkl"
            os.makedirs("models", exist_ok=True)
            path = os.path.join("models", filename)
            with open(path, "wb") as f:
                pickle.dump(model, f)

            if self.db:
                self.db.insert_model(
                    name=filename,
                    model_type=name,
                    model_hash=model_hash,
                    path=path,
                    accuracy=accuracy,
                    config_snapshot=json_str,
                    is_ensemble=0,
                    ensemble_members=""
                )
            return filename, accuracy, path

        else:
            estimators = []
            for plugin_name in ensemble_names:
                module = self.models_plugins.get(plugin_name)
                if not module:
                    raise ValueError(f"Модель `{plugin_name}` не найдена для ансамбля.")
                sub_params = params.get(plugin_name, {})
                base_model = module.build_model(sub_params)
                base_model.fit(X_train, y_train)
                estimators.append((plugin_name, base_model))

            ensemble_model = VotingClassifier(estimators=estimators, voting="soft")
            if X_val is not None and y_val is not None:
                preds = ensemble_model.predict(X_val)
                try:
                    preds = (preds > 0.5).astype(int)
                except:
                    preds = preds.astype(int)
                accuracy = (preds == y_val).mean()
            else:
                accuracy = None

            hash_obj = hashlib.sha256()
            json_str = ""
            if config_snapshot:
                json_str = json.dumps(config_snapshot, sort_keys=True)
                hash_obj.update(json_str.encode("utf-8"))
            model_hash = hash_obj.hexdigest()[:8]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_{'_'.join(ensemble_names)}_{ts}_{model_hash}.pkl"
            os.makedirs("models", exist_ok=True)
            path = os.path.join("models", filename)
            with open(path, "wb") as f:
                pickle.dump(ensemble_model, f)

            if self.db:
                self.db.insert_model(
                    name=filename,
                    model_type="Ensemble",
                    model_hash=model_hash,
                    path=path,
                    accuracy=accuracy,
                    config_snapshot=json_str,
                    is_ensemble=1,
                    ensemble_members=json.dumps(ensemble_names)
                )
            return filename, accuracy, path

    def load_model(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
