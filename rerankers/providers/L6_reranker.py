"""
MS MARCO MiniLM-L6-v2 Local Reranker
====================================
Lightweight cross-encoder reranker that runs fully offline using a local
copy of the `cross-encoder/ms-marco-MiniLM-L-6-v2` model.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_reranker import BaseReranker
from ..model.reranker_profile import RerankerProfile
from rerankers.model.reranker_profile import RerankerProfile
from rerankers.providers.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class L6Reranker(BaseReranker):
    """
    MS MARCO MiniLM-L6-v2 local reranker provider.
    Single Responsibility: Reranking sử dụng local MiniLM model.
    
    Config:
        - Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
        - Model path: C:\\Users\\ENGUYEMOX\\OneDrive - NTT DATA EMEAL\\Desktop\\RAG_sprint2\\RAG\\rerank_models
        - Batch size: 16
        - Max length: 512
        - Provider: local
        - Device: cpu (default, can use cuda)
    """
    
    # Class-level constants
    MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODEL_PATH = Path(r"C:\Users\ENGUYEMOX\OneDrive - NTT DATA EMEAL\Desktop\RAG_sprint2\RAG\rerank_models\model\model")
    PROVIDER = "local"
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    DEFAULT_DEVICE = "cpu"

    def __init__(self,
                 profile: Optional[RerankerProfile] = None,
                 device: Optional[str] = None):
        """
        Initialize L6 reranker.

        Args:
            profile: Reranker profile, nếu None sẽ tạo từ class constants
            device: Device to run on ('cpu' or 'cuda'), defaults to 'cpu'
        """
        if profile is None:
            profile = self._create_profile()

        super().__init__(profile)
        
        # Set device
        self.device = device if device is not None else self.DEFAULT_DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        self._model = None
        self._tokenizer = None
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
            normalize=False,  # MiniLM outputs raw logits
            use_fp16=False
        )

    def _initialize_model(self):
        """Initialize MiniLM model and tokenizer from local storage."""
        try:
            # Verify model path exists
            if not self.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {self.MODEL_PATH}"
                )
            
            # Load tokenizer and model from local path
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.MODEL_PATH))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.MODEL_PATH)
            )
            
            # Move to device and set to eval mode
            self._model.to(self.device)
            self._model.eval()
            
            logger.info(
                f"Loaded L6 reranker model on {self.device}: {self.MODEL_ID} "
                f"from {self.MODEL_PATH}"
            )
            
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Failed to initialize L6 reranker: {e}")
            raise

    @classmethod
    def create_default(cls, device: Optional[str] = None) -> 'L6Reranker':
        """
        Factory method để tạo L6 reranker với cấu hình mặc định.

        Args:
            device: Device to run on ('cpu' or 'cuda')

        Returns:
            L6Reranker: Configured L6 reranker
        """
        return cls(profile=None, device=device)

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
            
            # Compute scores in batches
            scores = []
            batch_size = self.profile.batch_size
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                with torch.no_grad():
                    # Tokenize batch
                    inputs = self._tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_LENGTH,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    # Get logits
                    logits = self._model(**inputs, return_dict=True).logits.view(-1)
                    batch_scores = logits.cpu().float().tolist()
                    scores.extend(batch_scores)
            
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
            if self._model is None or self._tokenizer is None:
                return False
            
            # Test with dummy input
            with torch.no_grad():
                inputs = self._tokenizer(
                    [("test query", "test document")],
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                _ = self._model(**inputs, return_dict=True).logits
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_config(self) -> dict:
        """
        Lấy config của L6 reranker.
        
        Returns:
            dict: Configuration dictionary
        """
        return {
            "model_id": self.MODEL_ID,
            "model_path": str(self.MODEL_PATH),
            "provider": self.PROVIDER,
            "batch_size": self.profile.batch_size,
            "max_length": self.profile.max_length,
            "normalize": self.profile.normalize,
            "device": self.device
        }