import os
import yaml
import logging

_CONFIG = None

def load_config(path=None):
    """
    加载 YAML 配置文件，添加错误处理以提高鲁棒性。
    """
    config_path = path
    if not config_path:
        env_path = os.environ.get("SWARM_CONFIG_PATH")
        if env_path:
            config_path = env_path
    if not config_path:
        # 默认尝试当前目录下的 config.yaml
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML config at {config_path}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error loading config from {config_path}: {e}")
        return {}

def get_config(path=None):
    global _CONFIG
    if _CONFIG is None or path:
        _CONFIG = load_config(path)
    return _CONFIG
