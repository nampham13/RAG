"""
Vector Store - FAISS Index Management
=====================================
Handles FAISS vector index creation, persistence, and metadata management.
Single Responsibility: Vector storage and retrieval operations.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages FAISS vector indexes and associated metadata.
    Single Responsibility: FAISS index operations and metadata persistence.
    """

    def __init__(self, vectors_dir: Path):
        """
        Initialize VectorStore.

        Args:
            vectors_dir: Directory to store FAISS indexes and metadata
        """
        self.vectors_dir = vectors_dir
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    def create_index(self, embeddings_data: List[Dict[str, Any]],
                    file_name: str, timestamp: str) -> tuple[Path, Path]:
        """
        Create FAISS index from embeddings data.

        Args:
            embeddings_data: List of embedding dictionaries
            file_name: Base name for output files
            timestamp: Timestamp string for file versioning

        Returns:
            Tuple of (faiss_file_path, metadata_file_path)
        """
        if not embeddings_data:
            raise ValueError("No embeddings data provided")

        # Extract vectors and create metadata map
        dimension = embeddings_data[0]["embedding_dimension"]
        vectors = np.array([item["embedding"] for item in embeddings_data], dtype='float32')

        metadata_map = self._create_metadata_map(embeddings_data)

        # Normalize vectors for cosine similarity (inner product of normalized vectors = cosine similarity)
        vectors_normalized = self._normalize_vectors(vectors)

        # Create FAISS index for cosine similarity using Inner Product
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors_normalized)  # type: ignore[arg-type]

        logger.info(f"Created FAISS index with cosine similarity: {index.ntotal} vectors, {dimension} dimensions")

        # Save files
        faiss_file = self.vectors_dir / f"{file_name}_vectors_{timestamp}.faiss"
        metadata_file = self.vectors_dir / f"{file_name}_metadata_map_{timestamp}.pkl"

        faiss.write_index(index, str(faiss_file))

        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata_map, f)

        logger.info(f"Saved FAISS index: {faiss_file.name}")
        logger.info(f"Saved metadata map: {metadata_file.name}")
        logger.info(f"Index type: IndexFlatIP (cosine similarity via inner product)")

        return faiss_file, metadata_file

    def load_index(self, faiss_file: Path, metadata_file: Path) -> tuple[faiss.Index, Dict[int, Dict[str, Any]]]:
        """
        Load FAISS index and metadata from disk.

        Args:
            faiss_file: Path to FAISS index file
            metadata_file: Path to metadata pickle file

        Returns:
            Tuple of (faiss_index, metadata_map)
        """
        index = faiss.read_index(str(faiss_file))

        with open(metadata_file, 'rb') as f:
            metadata_map = pickle.load(f)

        return index, metadata_map

    def _create_metadata_map(self, embeddings_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Create metadata mapping for FAISS index.

        Args:
            embeddings_data: List of embedding dictionaries

        Returns:
            Dictionary mapping index positions to metadata
        """
        metadata_map = {}
        for idx, item in enumerate(embeddings_data):
            metadata_map[idx] = {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "text_length": item["text_length"],
                "file_name": item["file_name"],
                "file_path": item["file_path"],
                "page_number": item["page_number"],
                "page_numbers": item["page_numbers"],
                "chunk_index": item["chunk_index"],
                "block_type": item["block_type"],
                "block_ids": item["block_ids"],
                "is_table": item["is_table"],
                "token_count": item["token_count"]
            }
        return metadata_map

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.

        Args:
            vectors: Input vectors array

        Returns:
            Normalized vectors array
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms