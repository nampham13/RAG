"""
HF BGE-v2 Re-ranker using FlagEmbedding
----------------------------------------
Loads a Hugging Face BGE-v2 reranker model and uses it to re-rank
candidate texts based on their relevance to a query.

Notes:
- BGE-v2 rerankers (e.g., BAAI/bge-reranker-v2-m3) are cross-encoder models
  that directly compute relevance scores for query-document pairs
- Default model is bge-reranker-v2-m3, but you can specify others
"""
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class BGEV2Reranker:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 8,
        use_fp16: bool = False,
    ):
        """
        Args:
            model_name: BGE reranker model name or path. If None, will
                        try to read from env var HF_BGE_V2_MODEL; if still None,
                        defaults to 'BAAI/bge-reranker-v2-m3'.
            device: "cpu" or "cuda"
            batch_size: batch size for computing scores
            use_fp16: whether to use FP16 (only for CUDA)
        """
        self.batch_size = batch_size
        self.device = device
        self._model = None

        # Determine model path
        model_path = model_name or os.getenv("HF_BGE_V2_MODEL") or "BAAI/bge-reranker-v2-m3"
        
        try:
            from FlagEmbedding import FlagReranker
            
            self._model = FlagReranker(
                model_path,
                use_fp16=use_fp16 and device == "cuda",
                device=device
            )
            logger.info(f"Loaded BGE-v2 reranker model: {model_path}")
            
        except ImportError as e:
            logger.error(
                "FlagEmbedding library not installed. "
                "Please install with `pip install FlagEmbedding`."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BGE-v2 reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        text_key: str = "text",
    ) -> List[Dict]:
        """
        Re-rank candidates by relevance to the query using BGE-v2 reranker.

        Args:
            query: the user query string
            candidates: list of dicts; each dict is expected to contain candidate[text_key]
            top_k: if provided, return only top_k items after reranking
            text_key: key in candidate dict that contains the text to rank

        Returns:
            list of dicts (copied) sorted with highest relevance first; 
            each dict gets "rerank_score" float
        """
        if not candidates:
            return []

        # Prepare candidate texts
        texts = []
        for c in candidates:
            t = c.get(text_key) or c.get("snippet") or c.get("text", "") or ""
            texts.append(t)

        try:
            # Create query-document pairs for the reranker (must be tuples)
            pairs = [(query, text) for text in texts]
            
            # Compute relevance scores
            scores = self._model.compute_score(
                pairs,
                batch_size=self.batch_size,
                normalize=True  # Normalize scores to [0, 1] range
            )
            
            # Handle both single score and list of scores
            if not isinstance(scores, list):
                scores = [scores]
            
            # Ensure scores is a list and convert to float
            score_list = []
            for s in scores:
                if s is None:
                    score_list.append(0.0)
                else:
                    score_list.append(float(s))
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original ordering.")
            # If reranking fails, keep original order but add rerank_score = 0
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