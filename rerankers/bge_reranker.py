from typing import List, Dict, Any, Optional, Iterable
from .i_reranker import IReranker
import logging
import os

logger = logging.getLogger(__name__)

try:
    # Dynamically import to avoid static analysis errors when packages are not installed
    import importlib
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModel = getattr(transformers, "AutoModel")
    F = torch.nn.functional
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch, seq_len, dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

class BGEV2Reranker(IReranker):
    """
    Reranker dùng embedding từ model BGE v2 (HuggingFace).
    Tính cosine similarity giữa query embedding và candidate embeddings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        embed_cache: bool = True,
    ) -> None:
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers/torch không có sẵn. Cần cài đặt để dùng BGEV2Reranker.")
        self.model_name = model_name or os.getenv("BGE_V2_MODEL", "BAAI/bge-small-en-v1.5")
        # Force CPU-only mode
        self.device = device if device is not None else "cpu"
        if self.device != "cpu":
            logger.warning(f"Device '{self.device}' requested but forcing CPU mode for compatibility")
            self.device = "cpu"
        self.batch_size = batch_size
        self.embed_cache = embed_cache

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self._ids = []
        self._docs = []
        self._meta = []
        self._embeddings = None  # torch.Tensor [N, D]

    def _embed_texts(self, texts: Iterable[str]) -> Any:
        all_embeds = []
        logger.info(f"BGEV2Reranker initialized on device={self.device} model={self.model_name}")
        with torch.no_grad():
            for i in range(0, len(list(texts)), self.batch_size):
                batch_texts = list(texts)[i:i+self.batch_size]
                enc = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                embeds = _mean_pooling(out, enc["attention_mask"])
                embeds = F.normalize(embeds, p=2, dim=1)
                all_embeds.append(embeds.cpu())
        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        return torch.empty((0, self.model.config.hidden_size))
    def fit(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Tạo embedding cho tất cả candidates và cache nếu được bật.
        """
        self._docs = [c.get("text", "") for c in candidates]
        self._ids = [c.get("id", idx) for idx, c in enumerate(candidates)]
        self._meta = [c.get("metadata", None) for c in candidates]
        self._id_to_idx = {cid: i for i, cid in enumerate(self._ids)}

        if len(self._docs) == 0:
            self._embeddings = torch.empty((0, self.model.config.hidden_size))
            return

        # compute embeddings
        self._embeddings = self._embed_texts(self._docs)  # cpu tensor
        logger.debug(f"Built embeddings for {self._embeddings.shape[0]} candidates (dim={self._embeddings.shape[1]})")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Compute query embedding, cosine similarity against candidate embeddings, return top_k.
        Nếu fit() chưa được gọi, sẽ tự động tính embeddings cho `candidates`.
        """
        if not hasattr(self, "_embeddings") or self._embeddings is None or len(self._docs) != len(candidates):
            self.fit(candidates)

        q_emb = self._embed_texts([query])  # cpu tensor [1, D]
        # Guard against None or empty tensors
        if q_emb is None or self._embeddings is None or q_emb.numel() == 0 or self._embeddings.numel() == 0:
            return []

        # embeddings are normalized -> cosine = dot
        sims = (self._embeddings @ q_emb.T).squeeze(1).numpy().tolist()  # list of floats

        results = []
        for idx, s in enumerate(sims):
            results.append({
                "id": self._ids[idx],
                "text": self._docs[idx],
                "metadata": self._meta[idx] if idx < len(self._meta) else None,
                "score": float(s),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]