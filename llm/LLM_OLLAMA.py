"""
Ollama Client - Gọi Ollama HTTP API
Endpoint: http://localhost:11434/api/chat
Không cần API key - chỉ cần base_url
"""
import requests
from typing import Any, List, Dict, Optional
from config_loader import resolve_ollama_settings
import time
import logging

logger = logging.getLogger(__name__)

def call_ollama_with_timing(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = 512,
) -> Dict[str, Any]:
    """
    Gọi Ollama /api/chat với timing info.
    
    Args:
        messages: OpenAI format [{"role": "user", "content": "..."}]
        model: Override model name (default từ config)
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
    
    Returns:
        Dict với 'response', 'time_taken', 'prompt_tokens', 'response_tokens', 'total_tokens'
    """
    start_time = time.time()

    try:
        resolved = resolve_ollama_settings(
            override_model=model,
            override_temperature=temperature,
            override_top_p=top_p,
            override_max_tokens=max_tokens,
        )

        base_url = resolved["base_url"]
        url = f"{base_url.rstrip('/')}/api/chat"

        payload = {
            "model": resolved["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": resolved["temperature"],
                "top_p": resolved["top_p"],
                "num_predict": resolved["max_tokens"],
            }
        }

        logger.info(f"Calling Ollama at {url} with model {resolved['model']}")
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start_time

        if resp.status_code != 200:
            logger.error(f"Ollama API error: {resp.status_code} - {resp.text}")
            return {
                "response": f"[Ollama Error {resp.status_code}] {resp.text[:200]}",
                "time_taken": elapsed,
                "prompt_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0
            }

        data = resp.json()
        
        # Parse Ollama response
        response_text = ""
        if "message" in data and isinstance(data["message"], dict):
            response_text = data["message"].get("content", "")

        # Token counts
        prompt_tokens = data.get("prompt_eval_count", 0)
        response_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + response_tokens

        return {
            "response": response_text or "",
            "time_taken": elapsed,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens
        }

    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start_time
        logger.error("Cannot connect to Ollama server")
        return {
            "response": f"[Ollama Error] Cannot connect to {base_url}. Make sure Ollama is running.",
            "time_taken": elapsed,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Ollama call failed: {e}")
        return {
            "response": f"[Ollama Error] {e}",
            "time_taken": elapsed,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }