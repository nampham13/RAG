"""
Reranker Factory
================
Factory pattern cho việc tạo reranker instances.
"""

import logging
from typing import Optional, Callable, List, Any
from .i_reranker import IReranker
from .bm25_reranker import BM25Reranker
from .bge_reranker import BGEV2Reranker


logger = logging.getLogger(__name__)


class RerankerFactory:
    """
    Factory để tạo reranker instances.
    Single Responsibility: Centralize reranker creation logic.
    """

    @staticmethod
    def create_bm25(
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        bm25_kwargs: Optional[dict] = None
    ) -> BM25Reranker:
        """
        Tạo BM25 reranker với default config.

        Args:
            tokenizer: Custom tokenizer function (callable)
            bm25_kwargs: Additional BM25 parameters (k1, b, epsilon, etc.)

        Returns:
            BM25Reranker: Configured reranker
        """
        logger.info("Creating BM25 reranker")
        return BM25Reranker(tokenizer=tokenizer, bm25_kwargs=bm25_kwargs)

    @staticmethod
    def create_bge(
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        embed_cache: bool = True
    ) -> BGEV2Reranker:
        """
        Tạo BGE V2 reranker với default config (CPU-only).

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5)
            device: Compute device (forced to 'cpu' for compatibility)
            batch_size: Batch size for embedding computation
            embed_cache: Enable embedding caching

        Returns:
            BGEV2Reranker: Configured reranker
        """
        logger.info(f"Creating BGE V2 reranker: {model_name or 'default'}")
        return BGEV2Reranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            embed_cache=embed_cache
        )

    @staticmethod
    def create_from_config(reranker_type: str, **kwargs) -> Optional[IReranker]:
        """
        Tạo reranker từ config string.

        Args:
            reranker_type: Type of reranker ('bm25' or 'bge')
            **kwargs: Additional parameters

        Returns:
            Optional[IReranker]: Configured reranker or None
        """
        reranker_type = reranker_type.lower()
        
        if reranker_type == "bm25":
            return RerankerFactory.create_bm25(**kwargs)
        elif reranker_type == "bge":
            return RerankerFactory.create_bge(**kwargs)
        else:
            logger.error(f"Unknown reranker type: {reranker_type}")
            return None