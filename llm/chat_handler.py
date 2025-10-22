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


def build_messages(query: str, context: str = "", history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """
    Build messages list theo OpenAI Chat Format
    
    Args:
        query: Câu hỏi hiện tại của user
        context: Context từ retrieval (mặc định rỗng)
        history: Lịch sử chat trước đó (mặc định None)
    
    Returns:
        Messages theo format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "..."}  # current query
        ]
    """
    messages = []

    # 1. System prompt (với context)
    system_content = format_system_prompt(context)
    messages.append({
        "role": "system",
        "content": system_content
    })
    
    # 2. History (nếu có)
    if history:
        normalized = normalize_history(history)
        messages.extend(normalized)
    
    # 3. Current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    return messages
