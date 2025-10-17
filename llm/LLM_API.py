"""
Gemini Client - Chỉ lo gọi Gemini API
Nhận messages dạng OpenAI format (List[Dict])
"""
import google.generativeai as genai
from config_loader import resolve_gemini_settings
from typing import Any, List, Dict, Optional
import time

# Cấu hình API key

def convert_to_gemini_format(messages: List[Dict[str, str]]) -> str:
    """
    Convert OpenAI Chat Format → Gemini String Format
    
    Args:
        messages: [{"role": "system/user/assistant", "content": "..."}, ...]
    
    Returns:
        String format cho Gemini:
        "System: ...\nUser: ...\nAssistant: ...\nUser: ..."
    """
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if not content:
            continue
        
        # Map roles
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    return "\n".join(prompt_parts)


def clean_response(response_text: str) -> str:
    """
    Loại bỏ prefix "Assistant:" nếu Gemini tự thêm vào
    
    Args:
        response_text: Raw response từ Gemini
    
    Returns:
        Cleaned response text
    """
    # Strip leading/trailing whitespace
    text = response_text.strip()
    
    # Remove "Assistant:" prefix (case-insensitive) và các biến thể
    prefixes = ["Assistant:", "Assistant :", "assistant:", "ASSISTANT:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    return text

def call_gemini_with_timing(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call Gemini API with timing information
    
    Returns:
        Dict with 'response', 'time_taken', and token usage keys
    """
    start_time = time.time()
    
    # Get settings and configure
    _settings = resolve_gemini_settings(
        override_model=model_name,
        override_temperature=temperature,
        override_max_tokens=max_tokens,
    )
    _api_key = _settings.get("api_key")
    if not _api_key:
        return {
            "response": "[Gemini Error] Missing API key (configure secrets or env).",
            "time_taken": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }
    
    genai.configure(api_key=_api_key)
    
    if model_name is None:
        model_name = _settings.get("model")
    if temperature is None:
        temperature = _settings.get("temperature")
    if max_tokens is None:
        max_tokens = _settings.get("max_tokens")
    
    try:
        prompt_text = convert_to_gemini_format(messages)
        model = genai.GenerativeModel(model_name)
        
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        if generation_config:
            response = model.generate_content(
                prompt_text,
                generation_config=generation_config
            )
        else:
            response = model.generate_content(prompt_text)
        
        elapsed = time.time() - start_time
        
        # Extract token usage from response
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            response_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
        else:
            prompt_tokens = 0
            response_tokens = 0
        
        total_tokens = prompt_tokens + response_tokens
        
        return {
            "response": clean_response(response.text),
            "time_taken": elapsed,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "response": f"[Gemini Error] {e}",
            "time_taken": elapsed,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }