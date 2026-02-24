import os
import logging
import yaml

_CONFIG = None
_LOGGER = logging.getLogger(__name__)

def load_config(path=None):
    config_path = path
    if not config_path:
        env_path = os.environ.get("SWARM_CONFIG_PATH")
        if env_path:
            config_path = env_path
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except FileNotFoundError:
        _LOGGER.error("Config file not found: %s", config_path)
        return {}
    except yaml.YAMLError as exc:
        _LOGGER.error("Failed to parse config file: %s", config_path)
        _LOGGER.error("YAML error: %s", exc)
        return {}
    except Exception as exc:
        _LOGGER.error("Failed to load config file: %s", config_path)
        _LOGGER.error("Error: %s", exc)
        return {}

def get_config(path=None):
    global _CONFIG
    if _CONFIG is None or path:
        _CONFIG = load_config(path)
    return _CONFIG
