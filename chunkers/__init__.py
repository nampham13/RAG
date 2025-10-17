"""
Chunkers package - Text chunking with source tracking and block merging integration.
"""

# Import data models
from .model import (
    BlockSpan,
    ProvenanceAgg,
    Score,
    Chunk,
    ChunkSet,
    ChunkStats,
    ChunkType,
    ChunkStrategy
)

# Import base chunker
from .base_chunker import BaseChunker

# Import chunker implementations
from .hybrid_chunker import HybridChunker, ChunkerMode
from .semantic_chunker import SemanticChunker
from .rule_based_chunker import RuleBasedChunker
from .fixed_size_chunker import FixedSizeChunker

__all__ = [
    # Data Models
    'BlockSpan',
    'ProvenanceAgg',
    'Score',
    'Chunk',
    'ChunkSet',
    'ChunkStats',
    'ChunkType',
    'ChunkStrategy',
    
    # Base Chunker
    'BaseChunker',
    
    # Chunker Implementations
    'HybridChunker',
    'SemanticChunker',
    'RuleBasedChunker',
    'FixedSizeChunker',
    
    # Enums
    'ChunkerMode',
]