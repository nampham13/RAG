"""
Gemini Client - Chỉ lo gọi Gemini API
Nhận messages dạng OpenAI format (List[Dict])
"""
"""
Gemini Client - Chỉ lo gọi Gemini API
Nhận messages dạng OpenAI format (List[Dict])
"""
# Configure logging and suppress warnings BEFORE importing Google libraries
import os
import sys
import logging

# Set environment variables to suppress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging to suppress Google Cloud warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.api_core').setLevel(logging.ERROR)

# Note: ALTS credential warnings are harmless and can be ignored
# They appear when Google Cloud libraries detect they're not running on GCP

import google.generativeai as genai

# Handle both direct execution and module import
try:
    # When run as module
    from .config_loader import resolve_gemini_settings
except ImportError:
    # When run directly as script
    from config_loader import resolve_gemini_settings

from typing import List, Dict, Optional


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


def call_gemini(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Gọi Gemini API
    
    Args:
        messages: Messages theo OpenAI format
        model_name: Tên model Gemini
        temperature: Temperature (0.0-1.0)
        max_tokens: Max output tokens
    
    Returns:
        Response text từ Gemini (đã clean)
    """
    try:
        # Resolve config and API key
        _settings = resolve_gemini_settings(
            override_model=model_name,
            override_temperature=temperature,
            override_max_tokens=max_tokens,
        )
        _api_key = _settings.get("api_key")
        if not _api_key:
            return "[Gemini Error] Missing API key (configure secrets or env)."
        
        # Configure API key globally
        genai.configure(api_key=_api_key)
        
        # Fill defaults from config when args are None
        if model_name is None:
            model_name = _settings.get("model", "gemini-2.0-flash-exp")  # Default fallback
        if temperature is None:
            temperature = _settings.get("temperature", 0.7)
        if max_tokens is None:
            max_tokens = _settings.get("max_tokens", 1000)
        
        # Ensure model_name is not None
        if not model_name:
            model_name = "gemini-2.0-flash-exp"
        # Convert format
        prompt_text = convert_to_gemini_format(messages)
        
        # Initialize model (without api_key parameter)
        model = genai.GenerativeModel(model_name) 
        
        # Generate
        generation_config = genai.GenerationConfig()
        if temperature is not None:
            generation_config.temperature = temperature
        if max_tokens is not None:
            generation_config.max_output_tokens = max_tokens
        
        response = model.generate_content(
            prompt_text,
            generation_config=generation_config
        )
        
        # Clean response trước khi return
        return clean_response(response.text)
    
    except Exception as e:
        return f"[Gemini Error] {e}"
