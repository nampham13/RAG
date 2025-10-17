from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    # Find the RAG project root by looking for key directories/files
    current = Path(__file__).resolve().parent

    # Go up until we find a directory containing 'pipeline' and 'llm' directories
    for parent in [current] + list(current.parents):
        if (parent / 'pipeline').exists() and (parent / 'llm').exists():
            return parent

    # Fallback to going up 2 levels (original logic)
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def _require(cfg: Dict[str, Any], dotted_key: str):
    node: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"Missing required key `{dotted_key}` in config/app.yaml")
        node = node[k]
    return node


_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    app_yaml = _repo_root() / "config" / "app.yaml"
    cfg = _load_yaml(app_yaml)
    if not cfg:
        raise RuntimeError(f"Missing or empty config file: {app_yaml}")

    _CONFIG_CACHE = cfg
    return cfg


def repo_path(*parts: str) -> Path:
    return _repo_root().joinpath(*parts)


def paths_data_dir() -> Path:
    rel = _require(get_config(), "paths.data_dir")
    return repo_path(str(rel))


def paths_prompt_path() -> Path:
    rel = _require(get_config(), "paths.prompt_path")
    return repo_path(str(rel))


def ui_default_backend() -> str:
    return str(_require(get_config(), "ui.default_backend"))


def resolve_gemini_settings(override_model=None, override_temperature=None, override_max_tokens=None):
    """Resolve Gemini settings from config and secrets."""
    try:
        import streamlit as st
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets'):
            # Check for [gemini] section in secrets.toml
            if 'gemini' in st.secrets and 'api_key' in st.secrets['gemini']:
                api_key = st.secrets['gemini']['api_key']
            # Fallback to direct gemini_api_key
            elif 'gemini_api_key' in st.secrets:
                api_key = st.secrets['gemini_api_key']
            else:
                # Fallback to config file
                cfg = _require(get_config(), "llm.gemini")
                api_key = cfg.get("api_key", "")
        else:
            # Fallback to config file
            cfg = _require(get_config(), "llm.gemini")
            api_key = cfg.get("api_key", "")
    except ImportError:
        # Streamlit not available, use config file
        cfg = _require(get_config(), "llm.gemini")
        api_key = cfg.get("api_key", "")

    if not api_key:
        raise ValueError("Gemini API key not found in secrets.toml or config/app.yaml")

    cfg = _require(get_config(), "llm.gemini")
    return {
        "api_key": api_key,
        "model": override_model or cfg.get("model", "gemini-2.0-flash-exp"),
        "temperature": override_temperature if override_temperature is not None else cfg.get("temperature", 0.7),
        "max_tokens": override_max_tokens if override_max_tokens is not None else cfg.get("max_tokens", 2048),
    }


def resolve_lmstudio_settings(
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_top_p: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = _require(get_config(), "llm.lmstudio")

    base_url = _require(cfg, "base_url")
    # Allow env override for base_url if provided
    base_url = os.getenv("LMSTUDIO_BASE_URL", base_url)

    api_key_env = _require(cfg, "api_key.env")
    api_key_default = _require(cfg, "api_key.default")
    api_key = os.getenv(str(api_key_env), api_key_default)

    model_cfg = _require(cfg, "model")
    model = override_model if override_model is not None else os.getenv("LMSTUDIO_MODEL", model_cfg)

    sampling = _require(cfg, "sampling")
    temperature = override_temperature if override_temperature is not None else sampling.get("temperature", None)
    top_p = override_top_p if override_top_p is not None else sampling.get("top_p", None)
    max_tokens = override_max_tokens if override_max_tokens is not None else sampling.get("max_tokens", None)

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
