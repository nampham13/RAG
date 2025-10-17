from typing import List, Dict, Any, Callable, Optional
from .i_reranker import IReranker
import logging
import numpy as np

# Try optional dependency: rank_bm25
try:
    from rank_bm25 import BM25Okapi 
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

# Fallback: sklearn Tfidf if BM25 not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    _HAS_TFIDF = True
except Exception:
    _HAS_TFIDF = False

logger = logging.getLogger(__name__)

def _default_tokenizer(text: str) -> List[str]:
    # Rất đơn giản: lowercase + split whitespace; project may replace bằng tokenizer chuyên sâu
    return text.lower().split()

class BM25Reranker(IReranker):
    """
    BM25-based reranker.
    - Nếu `rank_bm25` có sẵn sẽ dùng BM25Okapi.
    - Nếu không có, dùng TF-IDF cosine như fallback.
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        bm25_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tokenizer = tokenizer or _default_tokenizer
        self.bm25_kwargs = bm25_kwargs or {}
        self._fitted = False
        self._docs = []
        self._ids = []
        self._meta = []
        self._bm25 = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

    def fit(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Build BM25 / TF-IDF index từ candidates.
        candidates: list of dicts with keys 'id' and 'text' (optional 'metadata')
        """
        self._docs = [c.get("text", "") for c in candidates]
        self._ids = [c.get("id", idx) for idx, c in enumerate(candidates)]
        self._meta = [c.get("metadata", None) for c in candidates]

        if _HAS_BM25:
            tokenized = [self.tokenizer(d) for d in self._docs]
            self._bm25 = BM25Okapi(tokenized, **self.bm25_kwargs) # type: ignore
            logger.debug("BM25 index built with rank_bm25")
        else:
            logger.warning("No rank_bm25 or sklearn available: BM25Reranker will do simple substring scoring")
        self._fitted = True

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank candidates bằng BM25 hoặc TF-IDF fallback.
        Nếu fit() chưa được gọi, tự động fit dựa trên `candidates`.
        """
        if not self._fitted or self._docs != [c.get("text", "") for c in candidates]:
            # rebuild index for provided candidates
            self.fit(candidates)

        scores = []
        if _HAS_BM25 and self._bm25 is not None:
            q_tok = self.tokenizer(query)
            raw_scores = self._bm25.get_scores(q_tok)
            scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        elif _HAS_TFIDF and self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            q_vec = self._tfidf_vectorizer.transform([query])
            sims = (self._tfidf_matrix @ q_vec.T).toarray().ravel()
            # normalize by L2 to get cosine-like scores
            norms_docs = np.linalg.norm(self._tfidf_matrix.toarray(), axis=1) + 1e-12
            norm_q = np.linalg.norm(q_vec.toarray()) + 1e-12
            scores = (sims / (norms_docs * norm_q)).tolist()
        else:
            # Simple fallback: count occurrences
            q_terms = set(query.lower().split())
            for d in self._docs:
                score = sum(1 for t in q_terms if t in d.lower())
                scores.append(float(score))

        # aggregate results
        results = []
        for idx, s in enumerate(scores):
            results.append({
                "id": self._ids[idx],
                "text": self._docs[idx],
                "metadata": self._meta[idx] if idx < len(self._meta) else None,
                "score": float(s),
            })

        # sort and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]