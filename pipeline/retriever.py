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
from rerankers.i_reranker import IReranker

logger = logging.getLogger(__name__)


class Retriever:
    """
    Performs similarity search against FAISS vector indexes.
    Single Responsibility: Vector search operations and result formatting.
    """

    def __init__(self, embedder: IEmbedder, temperature: float = 1.0, reranker: Optional[IReranker] = None):
        """
        Initialize Retriever.

        Args:
            embedder: Embedder instance for query encoding
            temperature: Temperature parameter for softmax scoring (default: 1.0)
            reranker: Optional reranker for re-scoring retrieved results
        """
        self.embedder = embedder
        self.temperature = temperature
        self.reranker = reranker

    def search_similar(self, faiss_file: Path, metadata_map_file: Path,
                    query_text: str, top_k: int = 5, use_softmax: bool = True,
                    temperature: Optional[float] = None, use_reranking: bool = False,
                    reranking_top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity with FAISS.

        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search
            top_k: Number of results to return
            use_softmax: Whether to apply softmax scoring
            temperature: Temperature for softmax scoring
            use_reranking: Whether to apply reranking if reranker is available
            reranking_top_k: Number of candidates to retrieve before reranking (default: top_k * 3)

        Returns:
            List of similar chunks with metadata and cosine similarity scores
        """
        # Load FAISS index and metadata
        index, metadata_map = self._load_index_and_metadata(faiss_file, metadata_map_file)

        # Generate query embedding and normalize
        query_embedding = self.embedder.embed(query_text)
        query_vector = np.array([query_embedding], dtype='float32')
        query_normalized = self._normalize_vectors(query_vector)

        # Determine initial retrieval count for reranking
        if use_reranking and self.reranker is not None:
            initial_top_k = reranking_top_k if reranking_top_k else top_k * 3
            logger.info(f"Retrieving {initial_top_k} candidates for reranking")
        else:
            initial_top_k = top_k

        # Perform search using inner product (cosine similarity for normalized vectors)
        similarities, indices = index.search(query_normalized, initial_top_k)  # type: ignore[call-arg]
        
        # Apply softmax scoring if requested
        if use_softmax:
            softmax_scores = self._apply_softmax_scoring(similarities[0], temperature)
        else:
            softmax_scores = None

        # Format results (similarities are already cosine similarities for normalized vectors)
        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            if idx >= 0 and idx < len(metadata_map):
                result = metadata_map[idx].copy()
                result["cosine_similarity"] = float(similarity)
                result["distance"] = 1.0 - float(similarity)
                result["similarity_score"] = float(similarity)
                
                if use_softmax and softmax_scores is not None:
                    result["softmax_score"] = float(softmax_scores[i])
                
                results.append(result)
            else:
                logger.warning(f"Invalid index {idx} returned by FAISS search")

        # Apply reranking if enabled and reranker is available
        if use_reranking and self.reranker is not None and results:
            logger.info(f"Reranking {len(results)} candidates with {type(self.reranker).__name__}")
            
            # Prepare candidates for reranking
            candidates = []
            for result in results:
                candidates.append({
                    "id": result.get("chunk_id", result.get("id", "")),
                    "text": result.get("text", ""),
                    "metadata": result
                })
            
            # Perform reranking
            try:
                reranked = self.reranker.rerank(query_text, candidates, top_k=top_k)
                
                # Update results with reranking scores
                reranked_results = []
                for item in reranked:
                    # Get original metadata
                    original_metadata = item.get("metadata", {})
                    # Add reranking score
                    original_metadata["rerank_score"] = item.get("score", 0.0)
                    # Update similarity_score to rerank_score for consistency
                    original_metadata["similarity_score"] = item.get("score", 0.0)
                    reranked_results.append(original_metadata)
                
                results = reranked_results
                logger.info(f"Reranking completed: returned {len(results)} results")
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}, returning original results")

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