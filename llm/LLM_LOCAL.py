"""
LM Studio Client - Chỉ lo gọi LM Studio API
Nhận messages dạng OpenAI format (List[Dict])
LM Studio hỗ trợ OpenAI format nên KHÔNG CẦN convert
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI

# Handle both direct execution and module import
try:
    # When run as module
    from .config_loader import resolve_lmstudio_settings
except ImportError:
    # When run directly as script
    from config_loader import resolve_lmstudio_settings


def get_client() -> OpenAI:
    """
    Tạo OpenAI client kết nối tới LM Studio
    
    Returns:
        OpenAI client instance
    """
    settings = resolve_lmstudio_settings()
    return OpenAI(base_url=settings["base_url"], api_key=settings["api_key"])


def call_lmstudio(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = 512,
) -> str:
    """
    Gọi LM Studio API
    
    Args:
        messages: Messages theo OpenAI format (DÙNG TRỰC TIẾP)
        model: Tên model (mặc định từ env LMSTUDIO_MODEL)
        temperature: Temperature (0.0-1.0)
        top_p: Top-p sampling
        max_tokens: Max output tokens
    
    Returns:
        Response text từ LM Studio
    """
    try:
        # Resolve settings (params override config)
        resolved = resolve_lmstudio_settings(
            override_model=model,
            override_temperature=temperature,
            override_top_p=top_p,
            override_max_tokens=max_tokens,
        )
        # Get client
        client = get_client()
        
        # Get model name from resolved config
        
        # Call API - LM Studio hỗ trợ OpenAI format sẵn!
        response = client.chat.completions.create(
            model=resolved["model"],
            messages=messages,  # ← DÙNG TRỰC TIẾP, KHÔNG CẦN CONVERT
            temperature=resolved["temperature"],
            top_p=resolved["top_p"],
            max_tokens=resolved["max_tokens"],
        )
        
        return response.choices[0].message.content or ""
    
    except Exception as e:
        return f"[LM Studio Error] {e}"
