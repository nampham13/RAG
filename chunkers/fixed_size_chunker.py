"""
Fixed Size Chunker (Refactored Compact Version)
==============================================
Chunker chia text theo độ dài token cố định, sử dụng tiktoken.
Được dùng làm fallback khi các phương pháp semantic/structural không khả thi.
"""

import hashlib
import re
import time
from typing import List, Optional, Tuple
from .base_chunker import BaseChunker
from .model import (
    Chunk, ChunkSet, ChunkType, ChunkStrategy,
    ProvenanceAgg, BlockSpan, Score
)
from loaders.model.document import PDFDocument
from loaders.model.block import Block


# ============================================================
# Utility: TokenizerAdapter (encapsulate tiktoken)
# ============================================================
class TokenizerAdapter:
    """Wrapper cho tiktoken encoder, có fallback character-based."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding(encoding_name)
            self.name = encoding_name
        except Exception:
            self.encoder = None
            self.name = "character-based"

    def encode(self, text: str) -> List[int]:
        """Encode text → token ids (hoặc pseudo-char tokens)."""
        if self.encoder:
            return self.encoder.encode(text)
        return list(range(max(1, len(text) // 4)))

    def count(self, text: str) -> int:
        """Đếm token nhanh."""
        return len(self.encode(text))

    def decode_tokens(self, tokens: List[int]) -> str:
        if not self.encoder:
            return ""  # fallback
        try:
            return self.encoder.decode(tokens)
        except Exception:
            return ""
# ============================================================
# Utility: Score computation
# ============================================================
def compute_score(chunk_text: str, token_count: int, max_tokens: int) -> Score:
    """Tính điểm cơ bản cho FixedSize chunk."""
    completeness = 0.7
    if chunk_text.strip().endswith(('.', '!', '?')):
        completeness += 0.1
    structural = 0.6 + (0.1 if "\n\n" in chunk_text else 0)
    ratio = min(1.0, token_count / max_tokens)
    return Score(
        coherence_score=0.5,
        completeness_score=min(1.0, completeness),
        token_ratio=ratio,
        structural_integrity=min(1.0, structural),
    )


# ============================================================
# FixedSizeChunker main class
# ============================================================
class FixedSizeChunker(BaseChunker):
    """
    Chunk text theo độ dài token cố định.
    Sử dụng tiktoken nếu có, fallback character-based nếu không.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
        respect_sentence_boundary: bool = True,
    ):
        super().__init__(max_tokens, overlap_tokens)
        self.tokenizer = TokenizerAdapter(encoding_name)
        self.respect_sentence_boundary = respect_sentence_boundary

    # ------------------------------
    # Public: Chunk entire document
    # ------------------------------
    def chunk(self, document: PDFDocument) -> ChunkSet:
        # Lưu file_path để sử dụng trong chunk_blocks
        self._current_file_path = document.file_path
        
        chunk_set = ChunkSet(
            doc_id=document.meta.get("doc_id", "unknown"),
            file_path=document.file_path,
            chunk_strategy="fixed_size",
        )
        all_blocks = [
            b for p in document.pages for b in p.blocks if b.text.strip()
        ]
        chunks = self.chunk_blocks(all_blocks, chunk_set.doc_id)
        for c in chunks:
            chunk_set.add_chunk(c)
        chunk_set.link_chunks()
        return chunk_set

    # ------------------------------
    # Internal: Chunk a list of blocks
    # ------------------------------
    def chunk_blocks(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        text, positions = self._merge_blocks(blocks)
        return self._chunk_text(text, positions, doc_id)

    # ------------------------------
    # Core chunking logic
    # ------------------------------
    def _chunk_text(
        self, text: str, block_positions: List[Tuple[int, int, Block]], doc_id: str
    ) -> List[Chunk]:
        tokens = self.tokenizer.encode(text)
        chunks, idx = [], 0
        total = len(tokens)

        while idx < total:
            end_idx = min(idx + self.max_tokens, total)

            # optional boundary adjustment
            if self.respect_sentence_boundary and end_idx < total:
                end_idx = self._find_boundary(tokens, end_idx)

            sub_tokens = tokens[idx:end_idx]
            chunk_text = (
                text[self._token_to_char(idx, text): self._token_to_char(end_idx, text)]
                .strip()
            )
            if not chunk_text:
                break

            chunk = self._create_chunk(chunk_text, idx, end_idx, block_positions, doc_id, len(chunks))
            chunks.append(chunk)

            # Update idx with proper bounds checking
            if end_idx >= total:
                # Reached the end, stop
                break
            else:
                # Move forward with overlap
                idx = max(idx + 1, end_idx - self.overlap_tokens)

        return chunks

    # ------------------------------
    # Helper methods
    # ------------------------------
    def _merge_blocks(self, blocks: List[Block]) -> Tuple[str, List[Tuple[int, int, Block]]]:
        text, positions = "", []
        for b in blocks:
            start = len(text)
            text += b.text + "\n\n"
            positions.append((start, len(text), b))
        return text, positions

    def _token_to_char(self, token_idx: int, text: str) -> int:
        """Approximate mapping token index → character position."""
        if self.tokenizer.encoder:
            # estimate 4 chars/token on average if not mapped
            return min(len(text), token_idx * 4)
        return min(len(text), token_idx * 4)

    def _find_boundary(self, tokens: List[int], end_idx: int) -> int:
        """Tìm vị trí boundary hợp lý gần end_idx."""
        look_back = int(self.max_tokens * 0.15)
        return max(end_idx - look_back, 0)

    def _create_chunk(
        self,
        text: str,
        start_token: int,
        end_token: int,
        block_positions: List[Tuple[int, int, Block]],
        doc_id: str,
        index: int,
    ) -> Chunk:
        import logging
        logger = logging.getLogger(__name__)

        logger.debug(f"Creating chunk {index}: text_len={len(text)}, start_token={start_token}, end_token={end_token}")

        start_char = start_token * 4
        end_char = end_token * 4
        provenance = ProvenanceAgg(doc_id=doc_id, file_path=getattr(self, '_current_file_path', None))

        # Populate provenance by mapping the chunk range back to contributing blocks
        # Note: token->char mapping is approximate (4 chars/token), but sufficient for page tracking
        try:
            for b_start, b_end, b in block_positions:
                # Check overlap between block span in merged text and chunk char window
                if b_end > start_char and b_start < end_char:
                    page_num = None
                    try:
                        meta = getattr(b, 'metadata', None)
                        if isinstance(meta, dict):
                            page_num = meta.get('page_number')
                    except Exception:
                        page_num = None

                    provenance.add_span(BlockSpan(
                        block_id=getattr(b, 'stable_id', None) or f"block_{id(b)}",
                        start_char=0,
                        end_char=len(getattr(b, 'text', '') or ''),
                        page_number=page_num,
                    ))
        except Exception:
            # If anything goes wrong, keep provenance minimal but not empty
            pass

        # Fast token counting - avoid potential hangs
        try:
            token_count = self.tokenizer.count(text)
        except Exception as e:
            logger.warning(f"Token counting failed for chunk {index}, using estimate: {e}")
            token_count = len(text) // 4  # Fallback estimate

        score = compute_score(text, token_count, self.max_tokens)
        chunk_id = self._generate_chunk_id(doc_id, index, text)

        return Chunk(
            chunk_id=chunk_id,
            text=text,
            token_count=token_count,
            char_count=len(text),
            chunk_type=ChunkType.FIXED_SIZE,
            strategy=ChunkStrategy.TOKEN_LENGTH,
            provenance=provenance,
            score=score,
            metadata={
                "chunk_index": index,
                "tokenizer": self.tokenizer.name,
                "start_token": start_token,
                "end_token": end_token,
            },
        )

    def _generate_chunk_id(self, doc_id: str, idx: int, text: str) -> str:
        raw = f"{doc_id}|{idx}|{text[:50]}"
        return f"chunk_{doc_id}_{idx}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
