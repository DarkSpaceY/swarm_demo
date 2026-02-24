import os
import yaml

_CONFIG = None

def load_config(path=None):
    config_path = path
    if not config_path:
        env_path = os.environ.get("SWARM_CONFIG_PATH")
        if env_path:
            config_path = env_path
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def get_config(path=None):
    global _CONFIG
    if _CONFIG is None or path:
        _CONFIG = load_config(path)
    return _CONFIG
