# src/utils.py

import hashlib
import json

def hash_config(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
