"""
Jina Reranker Provider
=====================
Implementation cho Jina AI reranking models từ HuggingFace.
"""

import logging
from typing import List, Dict, Optional

from .base_reranker import BaseReranker
from ..model.reranker_profile import RerankerProfile

logger = logging.getLogger(__name__)


class JinaReranker(BaseReranker):
    """
    Jina AI reranker provider using HuggingFace models.
    Single Responsibility: Reranking sử dụng Jina reranker models.
    
    Config:
        - Model: jinaai/jina-reranker-v2-base-multilingual (default)
        - Batch size: 16
        - Max length: 1024
        - Provider: jina
        - Device: cpu (forced)
    
    Note: Uses sentence-transformers for cross-encoder functionality.
    """
    
    # Class-level constants
    MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual"
    PROVIDER = "jina"
    BATCH_SIZE = 16
    MAX_LENGTH = 1024
    DEVICE = "cpu"  # Force CPU only

    def __init__(self,
                 profile: Optional[RerankerProfile] = None):
        """
        Initialize Jina reranker.

        Args:
            profile: Reranker profile, nếu None sẽ tạo từ class constants
        """
        if profile is None:
            profile = self._create_profile()

        super().__init__(profile)
        self.device = self.DEVICE  # Always use CPU
        self._model = None
        self._initialize_model()

    @classmethod
    def _create_profile(cls) -> RerankerProfile:
        """
        Tạo RerankerProfile từ class constants.
        """
        return RerankerProfile(
            model_id=cls.MODEL_ID,
            provider=cls.PROVIDER,
            batch_size=cls.BATCH_SIZE,
            max_length=cls.MAX_LENGTH,
            normalize=True,
            use_fp16=False  # Disable FP16 for CPU
        )

    def _initialize_model(self):
        """Initialize Jina reranker model từ HuggingFace."""
        try:
            from sentence_transformers import CrossEncoder
            
            self._model = CrossEncoder(
                self.profile.model_id,
                device=self.DEVICE,
                max_length=self.profile.max_length
            )
            logger.info(f"Loaded Jina reranker model on CPU: {self.profile.model_id}")
            
        except ImportError as e:
            logger.error(
                "sentence-transformers library not installed. "
                "Please install with `pip install sentence-transformers`."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Jina reranker: {e}")
            raise

    @classmethod
    def create_default(cls) -> 'JinaReranker':
        """
        Factory method để tạo Jina reranker với cấu hình mặc định.

        Returns:
            JinaReranker: Configured Jina reranker (CPU only)
        """
        return cls(profile=None)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        text_key: str = "text"
    ) -> List[Dict]:
        """
        Re-rank candidates theo relevance với query.

        Args:
            query: Query string
            candidates: List của candidate documents
            top_k: Number of top results to return
            text_key: Key trong dict chứa text content

        Returns:
            List[Dict]: Reranked candidates với rerank_score
        """
        if not candidates:
            return []

        # Extract texts from candidates
        texts = [self._extract_text(c, text_key) for c in candidates]

        try:
            # Create query-document pairs
            pairs = [[query, text] for text in texts]
            
            # Compute relevance scores using CrossEncoder
            scores = self._model.predict(
                pairs,
                batch_size=self.profile.batch_size,
                show_progress_bar=False
            )
            
            # Convert to list if needed
            if not isinstance(scores, list):
                scores = scores.tolist()
            
            # Normalize scores if requested
            if self.profile.normalize:
                # Apply sigmoid to normalize to [0, 1] range
                import numpy as np
                scores = 1 / (1 + np.exp(-np.array(scores)))
                scores = scores.tolist()
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original ordering.")
            # If reranking fails, keep original order with score = 0
            out = [dict(c, rerank_score=0.0) for c in candidates]
            if top_k is not None:
                return out[:top_k]
            return out

        # Build output list with scores
        reranked = []
        for c, score in zip(candidates, scores):
            new_item = dict(c)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        # Sort descending by rerank_score
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def test_connection(self) -> bool:
        """
        Test if model is loaded and working.

        Returns:
            bool: True if model is ready
        """
        try:
            if self._model is None:
                return False
            
            # Test with dummy input
            test_pairs = [["test query", "test document"]]
            _ = self._model.predict(test_pairs, show_progress_bar=False)
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_config(self) -> dict:
        """
        Lấy config của Jina reranker.
        
        Returns:
            dict: Configuration dictionary
        """
        return {
            "model_id": self.MODEL_ID,
            "provider": self.PROVIDER,
            "batch_size": self.profile.batch_size,
            "max_length": self.profile.max_length,
            "normalize": self.profile.normalize,
            "device": self.DEVICE
        }