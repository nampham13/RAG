import os
import yaml
from typing import Any, Dict, Optional

class YAMLConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Ưu tiên rag/config/preprocessing.yaml
            base = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base, 'config', 'preprocessing.yaml')
        self.config_path = os.path.abspath(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            if not isinstance(config, dict):
                raise ValueError(f"Config file {self.config_path} must contain a dictionary, got {type(config)}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        value = self._config.get(section, {})
        if not isinstance(value, dict):
            return {}
        return value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()

class ConfigManager:
    """Singleton Config Manager - thread-safe và strict error handling."""
    _instance = None
    _config_loader = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_config_loader(self) -> YAMLConfigLoader:
        if self._config_loader is None:
            try:
                self._config_loader = YAMLConfigLoader()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize config loader: {e}")
        return self._config_loader

def get_config_loader() -> YAMLConfigLoader:
    """Factory function để lấy config loader."""
    manager = ConfigManager()
    return manager.get_config_loader()

def load_preprocessing_config() -> Dict[str, Any]:
    return get_config_loader().get_all()

def get_pdf_processing_config() -> Dict[str, Any]:
    return get_config_loader().get_section('pdf_processing')

def get_config_value(key: str, default: Any = None) -> Any:
    return get_config_loader().get(key, default)

def get_nested_config_value(*keys: str, default: Any = None) -> Any:
    return get_config_loader().get_nested(*keys, default=default)
