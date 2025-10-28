"""
Reranker Factory
===============
Creates reranker instances from configuration.
"""

from typing import Dict, Optional, Any

from .reranker_type import RerankerType
from .model.reranker_profile import RerankerProfile
from .i_reranker import IReranker
from .providers.bge_reranker import BGEReranker
from .providers.jina_reranker import JinaReranker


class RerankerFactory:
    """
    Factory to build rerankers based on profile and type.
    Single Responsibility: centralize reranker instantiation logic.
    All rerankers run on CPU only.
    """

    def __init__(self, registry: Dict[RerankerType, type] | None = None):
        self._registry = registry or {}

    def create(
        self,
        reranker_type: RerankerType,
        profile: RerankerProfile,
        **kwargs: Any
    ) -> IReranker:
        """
        Create the requested reranker.

        Args:
            reranker_type: Type of reranker to create
            profile: Reranker configuration profile
            **kwargs: Additional arguments for specific rerankers

        Returns:
            IReranker: Configured reranker instance (CPU only)
        """
        if reranker_type == RerankerType.BGE:
            return BGEReranker(profile=profile)
        elif reranker_type == RerankerType.JINA:
            return JinaReranker(profile=profile)
        
        raise ValueError(f"Unsupported reranker type: {reranker_type!r}")

    def create_bge(self, **kwargs: Any) -> BGEReranker:
        """
        Factory method cho BGE reranker.

        Args:
            **kwargs: Additional arguments (ignored for CPU-only mode)

        Returns:
            BGEReranker: BGE reranker instance (CPU only)
        """
        return BGEReranker.create_default()

    def create_jina(self, **kwargs: Any) -> JinaReranker:
        """
        Factory method cho Jina reranker.

        Args:
            **kwargs: Additional arguments (ignored for CPU-only mode)

        Returns:
            JinaReranker: Jina reranker instance (CPU only)
        """
        return JinaReranker.create_default()

    def create_bge_custom(
        self,
        model_id: str,
        batch_size: int = 8,
        **kwargs: Any
    ) -> BGEReranker:
        """
        Factory method cho custom BGE reranker model.

        Args:
            model_id: BGE model ID (e.g., "BAAI/bge-reranker-base")
            batch_size: Batch size for scoring
            **kwargs: Additional arguments

        Returns:
            BGEReranker: Custom BGE reranker instance (CPU only)
        """
        profile = RerankerProfile(
            model_id=model_id,
            provider="bge",
            batch_size=batch_size,
            max_length=kwargs.get('max_length', 512),
            normalize=kwargs.get('normalize', True),
            use_fp16=False  # Always False for CPU
        )
        return BGEReranker(profile=profile)

    def create_jina_custom(
        self,
        model_id: str,
        **kwargs: Any
    ) -> JinaReranker:
        """
        Factory method cho custom Jina reranker model.

        Args:
            model_id: Jina model ID from HuggingFace
            **kwargs: Additional arguments

        Returns:
            JinaReranker: Custom Jina reranker instance (CPU only)
        """
        profile = RerankerProfile(
            model_id=model_id,
            provider="jina",
            batch_size=kwargs.get('batch_size', 16),
            max_length=kwargs.get('max_length', 1024),
            normalize=kwargs.get('normalize', True),
            use_fp16=False  # Always False for CPU
        )
        return JinaReranker(profile=profile)