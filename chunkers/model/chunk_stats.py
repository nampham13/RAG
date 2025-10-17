"""
Chunk Statistics Model
======================
Thống kê tổng hợp về chunking.
"""

from dataclasses import dataclass, field
from typing import Dict
from .chunk_set import ChunkSet
from loaders.model.base import LoaderBaseModel


@dataclass
class ChunkStats(LoaderBaseModel):
    """
    Thống kê tổng hợp về chunking.
    Single Responsibility: Tính toán và lưu trữ thống kê chunking.
    """
    total_chunks: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    avg_tokens_per_chunk: float = 0.0
    avg_chars_per_chunk: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    min_chars: int = 0
    max_chars: int = 0

    # Thống kê theo loại
    chunks_by_type: Dict[str, int] = field(default_factory=dict)
    chunks_by_strategy: Dict[str, int] = field(default_factory=dict)

    # Điểm chất lượng trung bình
    avg_coherence_score: float = 0.0
    avg_completeness_score: float = 0.0
    avg_overall_score: float = 0.0

    # Phân bố pages
    chunks_per_page: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_chunk_set(cls, chunk_set: ChunkSet):
        """Tạo ChunkStats từ ChunkSet"""
        if not chunk_set.chunks:
            return cls()

        stats = cls()
        stats.total_chunks = len(chunk_set.chunks)
        stats.total_tokens = chunk_set.total_tokens
        stats.total_chars = chunk_set.total_chars

        # Token và char stats
        token_counts = [c.token_count for c in chunk_set.chunks]
        char_counts = [c.char_count for c in chunk_set.chunks]

        stats.avg_tokens_per_chunk = sum(token_counts) / len(token_counts)
        stats.avg_chars_per_chunk = sum(char_counts) / len(char_counts)
        stats.min_tokens = min(token_counts)
        stats.max_tokens = max(token_counts)
        stats.min_chars = min(char_counts)
        stats.max_chars = max(char_counts)

        # Đếm theo type và strategy
        for chunk in chunk_set.chunks:
            type_key = chunk.chunk_type.value
            stats.chunks_by_type[type_key] = stats.chunks_by_type.get(type_key, 0) + 1

            if chunk.strategy:
                strategy_key = chunk.strategy.value
                stats.chunks_by_strategy[strategy_key] = stats.chunks_by_strategy.get(strategy_key, 0) + 1

        # Score metrics
        coherence_scores = []
        completeness_scores = []
        overall_scores = []

        for chunk in chunk_set.chunks:
            if chunk.score:
                coherence_scores.append(chunk.score.coherence_score)
                completeness_scores.append(chunk.score.completeness_score)
                overall_scores.append(chunk.score.overall_score())

        if coherence_scores:
            stats.avg_coherence_score = sum(coherence_scores) / len(coherence_scores)
        if completeness_scores:
            stats.avg_completeness_score = sum(completeness_scores) / len(completeness_scores)
        if overall_scores:
            stats.avg_overall_score = sum(overall_scores) / len(overall_scores)

        # Chunks per page
        for chunk in chunk_set.chunks:
            if chunk.provenance:
                for page_num in chunk.provenance.page_numbers:
                    stats.chunks_per_page[page_num] = stats.chunks_per_page.get(page_num, 0) + 1

        return stats

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary"""
        return {
            'total_chunks': self.total_chunks,
            'total_tokens': self.total_tokens,
            'total_chars': self.total_chars,
            'avg_tokens_per_chunk': round(self.avg_tokens_per_chunk, 2),
            'avg_chars_per_chunk': round(self.avg_chars_per_chunk, 2),
            'min_tokens': self.min_tokens,
            'max_tokens': self.max_tokens,
            'min_chars': self.min_chars,
            'max_chars': self.max_chars,
            'chunks_by_type': self.chunks_by_type,
            'chunks_by_strategy': self.chunks_by_strategy,
            'avg_coherence_score': round(self.avg_coherence_score, 3),
            'avg_completeness_score': round(self.avg_completeness_score, 3),
            'avg_overall_score': round(self.avg_overall_score, 3),
            'chunks_per_page': self.chunks_per_page
        }

    def summary(self) -> str:
        """Tạo summary text của statistics"""
        return f"""
Chunking Statistics:
-------------------
Total Chunks: {self.total_chunks}
Total Tokens: {self.total_tokens}
Total Chars: {self.total_chars}
Avg Tokens/Chunk: {self.avg_tokens_per_chunk:.2f}
Token Range: {self.min_tokens} - {self.max_tokens}

Chunks by Type: {self.chunks_by_type}
Chunks by Strategy: {self.chunks_by_strategy}

Quality Scores:
  Coherence: {self.avg_coherence_score:.3f}
  Completeness: {self.avg_completeness_score:.3f}
  Overall: {self.avg_overall_score:.3f}
        """.strip()