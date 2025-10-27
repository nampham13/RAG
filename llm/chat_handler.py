"""
Chat Handler - Xử lý logic chat
Không phụ thuộc vào LLM cụ thể nào
"""
from typing import List, Dict, Optional

# Handle both direct execution and module import
try:
    # When run as module
    from .config_loader import paths_prompt_path
except ImportError:
    # When run directly as script
    from config_loader import paths_prompt_path

# Đường dẫn cố định tới system prompt



def load_system_prompt() -> str:
    """
    Load system prompt template từ file
    
    Returns:
        System prompt template (có {context})
    """
    try:
        prompt_file = paths_prompt_path() / "rag_system_prompt.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback nếu không tìm thấy file
        return """Bạn là trợ lý AI. Trả lời ngắn gọn, rõ ràng và chính xác.

Context từ tài liệu:
{context}"""
    except Exception as e:
        raise RuntimeError(f"Không thể đọc system prompt: {e}")


def format_system_prompt(context: str) -> str:
    """
    Format system prompt với context
    
    Args:
        context: Context từ retrieval system (tạm thời có thể rỗng)
    
    Returns:
        System prompt đã format
    """
    template = load_system_prompt()
    
    # Nếu không có context, thông báo rõ ràng
    if not context or context.strip() == "":
        context = "(Chưa có tài liệu nào được tải lên)"
    
    return template.format(context=context)


def normalize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Chuẩn hóa history về OpenAI Chat Format
    
    Args:
        history: History từ frontend [{"role": "user"/"bot", "content": "..."}]
    
    Returns:
        History đã chuẩn hóa [{"role": "user"/"assistant", "content": "..."}]
    """
    normalized = []
    
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Skip invalid messages
        if not role or not content:
            continue
        
        # Normalize role
        if role == "bot":
            role = "assistant"
        
        # Chỉ giữ user và assistant
        if role in ["user", "assistant"]:
            normalized.append({
                "role": role,
                "content": content
            })
    
    return normalized


def build_messages(query: str, context: str, history: list) -> list:
    """Build messages for LLM with adaptive system prompt based on context"""
    
    # Check if this is a general conversation (no context)
    if not context or context.strip() == "":
        # Use conversational system prompt
        system_prompt = """You are a helpful and friendly AI assistant. 
Engage in natural conversation with the user. Be warm, concise, and helpful.
If the user asks factual questions that you don't have information about, politely let them know."""
    else:
        # Use RAG-focused system prompt
        system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.
If the context doesn't contain relevant information, say so politely.

Context:
{context}
"""
        system_prompt = system_prompt.format(context=context)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in history[-10:]:  # Keep last 10 messages
        messages.append(msg)
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    return messages
