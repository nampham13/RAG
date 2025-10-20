"""
Retriever - Vector Similarity Search
====================================
Handles similarity search against FAISS vector indexes using cosine similarity.
Single Responsibility: Vector similarity search and result formatting.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import faiss

from embedders.i_embedder import IEmbedder

logger = logging.getLogger(__name__)


class Retriever:
    """
    Performs similarity search against FAISS vector indexes.
    Single Responsibility: Vector search operations and result formatting.
    """

    def __init__(self, embedder: IEmbedder, temperature: float = 1.0):
        """
        Initialize Retriever.

        Args:
            embedder: Embedder instance for query encoding
            temperature: Temperature parameter for softmax scoring (default: 1.0)
        """
        self.embedder = embedder
        self.temperature = temperature

    def search_similar(self, faiss_file: Path, metadata_map_file: Path,
                    query_text: str, top_k: int = 5, use_softmax: bool = False,
                    temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity with FAISS.

        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search
            top_k: Number of results to return

        Returns:
            List of similar chunks with metadata and cosine similarity scores
        """
        # Load FAISS index and metadata
        index, metadata_map = self._load_index_and_metadata(faiss_file, metadata_map_file)

        # Generate query embedding and normalize
        query_embedding = self.embedder.embed(query_text)
        query_vector = np.array([query_embedding], dtype='float32')
        query_normalized = self._normalize_vectors(query_vector)

        # Perform search using inner product (cosine similarity for normalized vectors)
        similarities, indices = index.search(query_normalized, top_k)  # type: ignore[call-arg]
        # Apply softmax scoring if requested
        if use_softmax:
            softmax_scores = self._apply_softmax_scoring(similarities[0], temperature)
        else:
            softmax_scores = None

# Format results (similarities are already cosine similarities for normalized vectors)
        import logging
        logger = logging.getLogger(__name__)

        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            if idx >= 0 and idx < len(metadata_map):
                # Get metadata entry - DON'T use .copy() yet
                metadata_entry = metadata_map[idx]
                
                # DEBUG: Check what's in metadata_map BEFORE copying
                text_in_metadata = metadata_entry.get("text", "")
                logger.info(f"DEBUG retriever BEFORE copy: idx={idx}, text length = {len(text_in_metadata)}, first 100 chars = {text_in_metadata[:100]}")
                
                # Create result dict manually to ensure text is preserved
                result = {
                    "text": text_in_metadata,  # EXPLICITLY set text first
                    "cosine_similarity": float(similarity),
                    "distance": 1.0 - float(similarity),
                    "similarity_score": float(similarity)
                }
                
                # Add other metadata fields
                for key, value in metadata_entry.items():
                    if key not in result:  # Don't overwrite what we already set
                        result[key] = value
                
                if use_softmax and softmax_scores is not None:
                    result["softmax_score"] = float(softmax_scores[i])
                
                results.append(result)
            else:
                logger.warning(f"Invalid index {idx} returned by FAISS search")

        logger.info(f"Cosine similarity search completed: found {len(results)} results for query")
        return results

    def _load_index_and_metadata(self, faiss_file: Path, metadata_map_file: Path) -> tuple[faiss.Index, Dict[int, Dict[str, Any]]]:
        """
        Load FAISS index and metadata from disk.

        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file

        Returns:
            Tuple of (faiss_index, metadata_map)
        """
        index = faiss.read_index(str(faiss_file))

        import pickle
        with open(metadata_map_file, 'rb') as f:
            metadata_map = pickle.load(f)

        return index, metadata_map

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a single vector for cosine similarity.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize multiple vectors for cosine similarity.

        Args:
            vectors: Input vectors array

        Returns:
            Normalized vectors array
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def _apply_softmax_scoring(self, similarities: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        """
        Apply exponential scaling, temperature, and normalization (softmax) to scores.

        Args:
            similarities: Raw similarity scores
            temperature: Temperature parameter for softmax (higher = more uniform, lower = more peaked)
                        If None, uses instance temperature

        Returns:
            Softmax-normalized scores
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Apply exponential and temperature
        exp_scores = np.exp(similarities / temp)
        
        # Normalize to sum to 1
        normalized_scores = exp_scores / np.sum(exp_scores)
        
        return normalized_scores