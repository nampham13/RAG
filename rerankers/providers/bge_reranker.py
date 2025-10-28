"""
BGE Reranker Provider
====================
Implementation cho BGE-v2 reranking models.
"""

import logging
from typing import List, Dict, Optional

from .base_reranker import BaseReranker
from ..model.reranker_profile import RerankerProfile

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """
    BGE-v2 reranker provider using FlagEmbedding.
    Single Responsibility: Reranking sử dụng BGE cross-encoder models.
    
    Config:
        - Model: BAAI/bge-reranker-v2-m3 (default)
        - Batch size: 8
        - Max length: 512
        - Provider: bge
        - Device: cpu (forced)
    """
    
    # Class-level constants
    MODEL_ID = "BAAI/bge-reranker-v2-m3"
    PROVIDER = "bge"
    BATCH_SIZE = 8
    MAX_LENGTH = 512
    DEVICE = "cpu"  # Force CPU only

    def __init__(self,
                 profile: Optional[RerankerProfile] = None):
        """
        Initialize BGE reranker.

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
        """Initialize FlagReranker model."""
        try:
            from FlagEmbedding import FlagReranker
            
            self._model = FlagReranker(
                self.profile.model_id,
                use_fp16=False,  # FP16 not supported on CPU
                device=self.DEVICE
            )
            logger.info(f"Loaded BGE reranker model on CPU: {self.profile.model_id}")
            
        except ImportError as e:
            logger.error(
                "FlagEmbedding library not installed. "
                "Please install with `pip install FlagEmbedding`."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BGE reranker: {e}")
            raise

    @classmethod
    def create_default(cls) -> 'BGEReranker':
        """
        Factory method để tạo BGE reranker với cấu hình mặc định.

        Returns:
            BGEReranker: Configured BGE reranker (CPU only)
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
            pairs = [(query, text) for text in texts]
            
            # Compute relevance scores
            scores = self._model.compute_score(
                pairs,
                batch_size=self.profile.batch_size,
                normalize=self.profile.normalize
            )
            
            # Handle both single score and list of scores
            if not isinstance(scores, list):
                scores = [scores]
            
            # Convert to float list
            score_list = [float(s) if s is not None else 0.0 for s in scores]
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original ordering.")
            # If reranking fails, keep original order with score = 0
            out = [dict(c, rerank_score=0.0) for c in candidates]
            if top_k is not None:
                return out[:top_k]
            return out

        # Build output list with scores
        reranked = []
        for c, score in zip(candidates, score_list):
            new_item = dict(c)
            new_item["rerank_score"] = score
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
            test_pairs = [("test query", "test document")]
            _ = self._model.compute_score(test_pairs, normalize=False)
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_config(self) -> dict:
        """
        Lấy config của BGE reranker.
        
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