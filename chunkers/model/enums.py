"""
Chunker Enums
=============
Định nghĩa enums cho chunking system.
"""

from enum import Enum


class ChunkType(Enum):
    """Loại chunk dựa trên nguồn gốc và cấu trúc"""
    SEMANTIC = "semantic"      # Chunk dựa trên ngữ nghĩa/coherence
    STRUCTURAL = "structural"  # Chunk dựa trên cấu trúc (heading, list, table)
    FIXED_SIZE = "fixed_size"  # Chunk dựa trên token count
    HYBRID = "hybrid"          # Chunk kết hợp nhiều chiến lược


class ChunkStrategy(Enum):
    """Chiến lược chunking được sử dụng"""
    SEMANTIC_COHERENCE = "semantic_coherence"
    HEADING_BASED = "heading_based"
    LIST_BASED = "list_based"
    TABLE_BASED = "table_based"
    TOKEN_LENGTH = "token_length"
    PARAGRAPH_BASED = "paragraph_based"