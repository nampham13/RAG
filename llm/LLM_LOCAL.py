"""
LM Studio Client - Gọi LM Studio API
Endpoint: http://localhost:1234/v1/chat/completions (OpenAI compatible)
Không cần API key - chỉ cần base_url
"""
from typing import Any, List, Dict, Optional
from openai import OpenAI
from config_loader import resolve_lmstudio_settings
import time

def get_client() -> OpenAI:
    """
    Tạo OpenAI client kết nối tới LM Studio.
    LM Studio không cần API key, chỉ cần base_url.
    """
    settings = resolve_lmstudio_settings()
    return OpenAI(
        base_url=f"{settings['base_url']}/v1",
        api_key="not-needed"  # LM Studio không check API key
    )

def call_lmstudio_with_timing(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Call LM Studio API với timing information.
    
    Args:
        messages: OpenAI format messages
        model: Override model name
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
    
    Returns:
        Dict với 'response', 'time_taken', và token usage
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