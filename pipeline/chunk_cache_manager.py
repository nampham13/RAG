"""
Chunk Cache Manager for RAG Pipeline
====================================
Handles caching and tracking of processed chunks to avoid re-processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ChunkCacheManager:
    """
    Manages cache of processed chunks to avoid duplicate processing.
    Single Responsibility: Track and cache chunk processing state.
    """

    def __init__(self, cache_file: Path):
        """
        Initialize Chunk Cache Manager.

        Args:
            cache_file: Path to cache file for storing processed chunks
        """
        self.cache_file = cache_file
        self.processed_chunks: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache of processed chunks from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.processed_chunks = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed chunks cache: {e}")
                self.processed_chunks = {}
        else:
            self.processed_chunks = {}

    def _save_cache(self):
        """Save cache of processed chunks to file."""
        try:
            # Ensure parent directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save processed chunks cache: {e}")

    def is_chunk_processed(self, chunk_id: str, content_hash: str) -> bool:
        """
        Check if chunk has already been processed.

        Args:
            chunk_id: Unique chunk identifier
            content_hash: Hash of chunk content for verification

        Returns:
            True if chunk was already processed, False otherwise
        """
        if chunk_id in self.processed_chunks:
            stored_hash = self.processed_chunks[chunk_id].get('content_hash')
            return stored_hash == content_hash
        return False

    def mark_chunk_processed(self, chunk_id: str, content_hash: str, metadata: Dict[str, Any]):
        """
        Mark chunk as processed and save to cache.

        Args:
            chunk_id: Unique chunk identifier
            content_hash: Hash of chunk content
            metadata: Additional metadata to store
        """
        self.processed_chunks[chunk_id] = {
            'content_hash': content_hash,
            'processed_at': datetime.now().isoformat(),
            'metadata': metadata
        }
        self._save_cache()

    def remove_chunk(self, chunk_id: str):
        """
        Remove a chunk from the cache.

        Args:
            chunk_id: Unique chunk identifier to remove
        """
        if chunk_id in self.processed_chunks:
            del self.processed_chunks[chunk_id]
            self._save_cache()
            logger.info(f"Removed chunk {chunk_id} from cache")
        else:
            logger.warning(f"Chunk {chunk_id} not found in cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        return {
            "total_cached_chunks": len(self.processed_chunks),
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            "oldest_entry": min((entry['processed_at'] for entry in self.processed_chunks.values()), default=None),
            "newest_entry": max((entry['processed_at'] for entry in self.processed_chunks.values()), default=None)
        }

    def clear_cache(self):
        """Clear all cached chunks."""
        self.processed_chunks = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Chunk cache cleared")