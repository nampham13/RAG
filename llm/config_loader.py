"""
Config Loader - Đọc config từ app.yaml
KHÔNG có API key handling cho local services (Ollama, LM Studio)
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def get_config() -> Dict[str, Any]:
    """Load và cache config từ app.yaml"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    config_path = Path(__file__).parent.parent / "config" / "app.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE

def _require(cfg: Dict[str, Any], key: str) -> Any:
    """Helper để lấy required config value"""
    if key not in cfg:
        raise ValueError(f"Missing required config: {key}")
    return cfg[key]

def ui_default_backend() -> str:
    """Lấy default backend (gemini/ollama/lmstudio)"""
    try:
        config = get_config()
        return config.get("ui", {}).get("default_backend", "gemini")
    except Exception:
        return "gemini"

def ui_default_local_backend() -> str:
    """Lấy default local backend (ollama/lmstudio)"""
    try:
        config = get_config()
        return config.get("ui", {}).get("default_local_backend", "ollama")
    except Exception:
        return "ollama"

def paths_data_dir() -> str:
    """Lấy data directory path"""
    try:
        config = get_config()
        return config.get("paths", {}).get("data_dir", "data")
    except Exception:
        return "data"

def paths_prompt_path() -> str:
    """Lấy prompt directory path"""
    try:
        config = get_config()
        return config.get("paths", {}).get("prompt_path", "prompts")
    except Exception:
        return "prompts"

def resolve_ollama_settings(
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_top_p: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve Ollama settings từ config.
    Ollama local - chỉ cần base_url và model params.
    """
    config = get_config()
    cfg = _require(config, "llm")
    ollama_cfg = _require(cfg, "ollama")

    base_url = _require(ollama_cfg, "base_url")
    base_url = os.getenv("OLLAMA_BASE_URL", base_url)

    model = override_model or os.getenv("OLLAMA_MODEL", _require(ollama_cfg, "model"))
    temperature = override_temperature if override_temperature is not None else ollama_cfg.get("temperature", 0.7)
    top_p = override_top_p if override_top_p is not None else ollama_cfg.get("top_p", 0.95)
    max_tokens = override_max_tokens if override_max_tokens is not None else ollama_cfg.get("max_tokens", 2048)

    return {
        "base_url": base_url,
        "model": model,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

def resolve_lmstudio_settings(
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_top_p: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve LM Studio settings từ config.
    LM Studio local - chỉ cần base_url và model params.
    """
    config = get_config()
    cfg = _require(config, "llm")
    lms_cfg = _require(cfg, "lmstudio")

    base_url = _require(lms_cfg, "base_url")
    base_url = os.getenv("LMSTUDIO_BASE_URL", base_url)

    model = override_model or os.getenv("LMSTUDIO_MODEL", _require(lms_cfg, "model"))
    temperature = override_temperature if override_temperature is not None else lms_cfg.get("temperature", 0.7)
    top_p = override_top_p if override_top_p is not None else lms_cfg.get("top_p", 0.95)
    max_tokens = override_max_tokens if override_max_tokens is not None else lms_cfg.get("max_tokens", 2048)

    return {
        "base_url": base_url,
        "model": model,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

def resolve_gemini_settings(
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve Gemini settings từ config.
    Gemini API - cần API key từ secrets.toml hoặc environment.
    
    Returns:
        Dict chứa model, temperature, max_tokens
        (API key được load riêng từ Streamlit secrets)
    """
    config = get_config()
    cfg = _require(config, "llm")
    gemini_cfg = _require(cfg, "gemini")

    model = override_model or os.getenv("GEMINI_MODEL", _require(gemini_cfg, "model"))
    temperature = override_temperature if override_temperature is not None else gemini_cfg.get("temperature", 0.7)
    max_tokens = override_max_tokens if override_max_tokens is not None else gemini_cfg.get("max_tokens", 2048)

    return {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }