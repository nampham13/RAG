"""
LM Studio Client - Chỉ lo gọi LM Studio API
Nhận messages dạng OpenAI format (List[Dict])
LM Studio hỗ trợ OpenAI format nên KHÔNG CẦN convert
"""
import os
from typing import Any, List, Dict, Optional
from openai import OpenAI
from config_loader import resolve_lmstudio_settings
import time

def get_client() -> OpenAI:
    """
    Tạo OpenAI client kết nối tới LM Studio
    
    Returns:
        OpenAI client instance
    """
    settings = resolve_lmstudio_settings()
    return OpenAI(base_url=settings["base_url"], api_key=settings["api_key"])

def call_lmstudio_with_timing(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Call LM Studio API with timing information
    
    Returns:
        Dict with 'response', 'time_taken', and token usage keys
    """
    start_time = time.time()
    
    try:
        resolved = resolve_lmstudio_settings(
            override_model=model,
            override_temperature=temperature,
            override_top_p=top_p,
            override_max_tokens=max_tokens,
        )
        client = get_client()
        
        response = client.chat.completions.create(
            model=resolved["model"],
            messages=messages,
            temperature=resolved["temperature"],
            top_p=resolved["top_p"],
            max_tokens=resolved["max_tokens"],
        )
        
        elapsed = time.time() - start_time
        
        # Extract token usage
        usage = getattr(response, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', prompt_tokens + completion_tokens) if usage else 0
        
        return {
            "response": response.choices[0].message.content or "",
            "time_taken": elapsed,
            "prompt_tokens": prompt_tokens,
            "response_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "response": f"[LM Studio Error] {e}",
            "time_taken": elapsed,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }