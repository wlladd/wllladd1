# src/indicators_manager.py

import os
import importlib.util

class IndicatorManager:
    def __init__(self, plugins_path="plugins/indicators"):
        self.plugins_path = plugins_path
        self.indicators = {}
        self._scan_plugins()

    def _scan_plugins(self):
        for filename in os.listdir(self.plugins_path):
            if filename.endswith(".py") and filename != "__init__.py":
                fullpath = os.path.join(self.plugins_path, filename)
                spec = importlib.util.spec_from_file_location(filename[:-3], fullpath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                name = getattr(module, "plugin_name", None)
                if name:
                    self.indicators[name] = module

    def get_plugin_names(self) -> list:
        return list(self.indicators.keys())

    def get_plugin_params(self, name: str) -> dict:
        module = self.indicators.get(name)
        if not module:
            raise ValueError(f"Индикатор `{name}` не найден.")
        return module.init_params()

    def calculate(self, name: str, df, params: dict):
        module = self.indicators.get(name)
        if not module:
            raise ValueError(f"Индикатор `{name}` не найден.")
        return module.calculate(df, params)

    def calculate_all(self, df, selected: list, params_dict: dict):
        import pandas as pd
        result = pd.DataFrame(index=df.index)
        for name in selected:
            params = params_dict.get(name, {})
            series = self.calculate(name, df, params)
            result[name] = series
        return result
