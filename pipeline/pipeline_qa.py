"""
RAG Retrieval Service
=====================
Module chỉ phụ trách phần Retrieval (FAISS search) để UI có thể hiển thị nguồn
và/hoặc tự ghép context vào prompt. Không gọi LLM tại đây.

Sử dụng nhanh:
    from RAG_system.pipeline.rag_pipeline import RAGPipeline
    from RAG_system.pipeline.rag_qa_engine import RAGRetrievalService

    pipeline = RAGPipeline(output_dir="data")
    retriever = RAGRetrievalService(pipeline)
    results = retriever.retrieve("Tìm điểm chính?", top_k=5)
    context = retriever.build_context(results)
    ui_items = retriever.to_ui_items(results)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from pipeline.rag_pipeline import RAGPipeline
from pipeline.query_expander import QueryExpander


logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """
    Dịch vụ Retrieval thuần: tìm kiếm Top-K đoạn liên quan từ FAISS index và
    cung cấp tiện ích build context + payload hiển thị cho UI.
    Không gọi LLM tại đây (UI hoặc lớp khác sẽ làm việc đó).
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.query_expander = QueryExpander()

    # ---------- Retrieval utilities ----------
    def _match_metadata_for_vectors(self, vectors_file: Path) -> Optional[Path]:
        """
        Tìm file metadata_map tương ứng với vectors bằng pattern tên file.
        Ví dụ: mydoc_vectors_20250101_120000.faiss => mydoc_metadata_map_20250101_120000.pkl
        """
        name = vectors_file.name
        if "_vectors_" not in name:
            return None
        candidate = self.pipeline.vectors_dir / name.replace("_vectors_", "_metadata_map_").replace(".faiss", ".pkl")
        return candidate if candidate.exists() else None

    def get_latest_index_pair(self) -> Optional[Tuple[Path, Path]]:
        """
        Lấy cặp (faiss_index, metadata_map) mới nhất trong thư mục vectors.
        Bỏ qua các file FAISS bị hỏng và thử file tiếp theo.
        Trả về None nếu không tìm thấy file hợp lệ.
        """
        faiss_files = sorted(
            self.pipeline.vectors_dir.glob("*_vectors_*.faiss"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for vf in faiss_files:
            mf = self._match_metadata_for_vectors(vf)
            if mf is not None:
                # Test if FAISS file can be loaded
                if self._test_faiss_file(vf):
                    return vf, mf
                else:
                    logger.warning(f"Skipping corrupted FAISS file: {vf}")
        return None

    def _test_faiss_file(self, faiss_file: Path) -> bool:
        """
        Test if a FAISS file can be loaded without errors.
        """
        try:
            import faiss
            faiss.read_index(str(faiss_file))
            return True
        except Exception as e:
            logger.warning(f"FAISS file test failed for {faiss_file}: {e}")
            # Don't auto-cleanup for now - let user decide
            # self._cleanup_corrupted_file(faiss_file)
            return False

    def _cleanup_corrupted_file(self, faiss_file: Path) -> None:
        """
        Remove corrupted FAISS file and its corresponding metadata file.
        """
        try:
            # Remove FAISS file
            faiss_file.unlink()
            logger.info(f"Removed corrupted FAISS file: {faiss_file}")

            # Remove corresponding metadata file
            mf = self._match_metadata_for_vectors(faiss_file)
            if mf and mf.exists():
                mf.unlink()
                logger.info(f"Removed corresponding metadata file: {mf}")
        except Exception as e:
            logger.error(f"Failed to cleanup corrupted files: {e}")

    def build_context(self, results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
        """
        Tạo chuỗi context gọn từ danh sách kết quả retrieval (top-k).
        Sử dụng provenance information để tạo source attribution chi tiết hơn.
        Cắt ngắn mỗi chunk để đảm bảo có chỗ cho nhiều sources.
        """
        parts: List[str] = []
        total = 0
        max_per_chunk = max(400, max_chars // 8)  # Mỗi chunk tối đa 400 ký tự để đảm bảo capture keywords
        
        for i, r in enumerate(results, 1):
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = r.get("similarity_score", 0.0)
            text = r.get("text", "")
            
            # Cắt ngắn text
            if len(text) > max_per_chunk:
                text = text[:max_per_chunk] + "..."

            # Enhanced source attribution using provenance if available
            provenance = r.get("provenance")
            if provenance and isinstance(provenance, dict):
                # Use provenance for more detailed source info
                page_nums = provenance.get("page_numbers", [])
                if page_nums:
                    page_range = f"pages {min(page_nums)}-{max(page_nums)}" if len(page_nums) > 1 else f"page {page_nums[0]}"
                else:
                    page_range = f"page {page}"

                source_blocks = provenance.get("source_blocks", [])
                if source_blocks:
                    block_info = f", blocks {len(source_blocks)}"
                else:
                    block_info = ""

                source_info = f"{file_name} ({page_range}{block_info})"
            else:
                # Fallback to basic info
                source_info = f"{file_name} (page {page})"

            # Check for table data
            table_data = r.get("table_data")
            if table_data:
                table_note = " [TABLE DATA]"
            else:
                table_note = ""

            piece = f"[{i}] Source: {source_info}, score {score:.3f}{table_note}\n{text}"
            parts.append(piece)
            total += len(piece)
            if total > max_chars:
                break
        return "\n\n".join(parts)

    def retrieve(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Trả về danh sách kết quả giống với retriever (metadata + similarity_score).
        Nếu không có index hoặc embedder không sẵn sàng, trả về list rỗng.
        """
        try:
            pair = self.get_latest_index_pair()
            if pair is None:
                logger.info("Không tìm thấy index trong thư mục vectors.")
                return []
            if not self.pipeline.embedder.test_connection():
                logger.warning("Embedder (Ollama) chưa sẵn sàng; bỏ qua retrieval.")
                return []
            
            # Expand query to improve matching
            expanded_query = self.query_expander.expand(query_text)
            
            faiss_file, metadata_map_file = pair
            return self.pipeline.search_similar(
                faiss_file=faiss_file,
                metadata_map_file=metadata_map_file,
                query_text=expanded_query,  # Use expanded query
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def to_ui_items(self, results: List[Dict[str, Any]], max_text_len: int = 500) -> List[Dict[str, Any]]:
        """
        Chuyển danh sách kết quả sang dạng dễ hiển thị ở UI.
        Mỗi item gồm: title, snippet, file_name, page_number, similarity_score, distance.
        """
        ui_items: List[Dict[str, Any]] = []
        for r in results:
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = float(r.get("similarity_score", 0.0))
            dist = float(r.get("distance", 0.0))
            text = r.get("text", "") or ""
            snippet = (text[: max_text_len - 3] + "...") if len(text) > max_text_len else text
            ui_items.append(
                {
                    "title": f"{file_name} - trang {page}",
                    "snippet": snippet,
                    "file_name": file_name,
                    "page_number": page,
                    "similarity_score": round(score, 4),
                    "distance": round(dist, 4),
                }
            )
        return ui_items


def fetch_retrieval(
    query_text: str,
    pipeline: Optional[RAGPipeline] = None,
    top_k: int = 10,
    max_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Tiện ích một hàm: thực hiện retrieval và trả về {context, sources} cho UI.
    """
    if pipeline is None:
        pipeline = RAGPipeline(output_dir="data")
    service = RAGRetrievalService(pipeline)
    results = service.retrieve(query_text=query_text, top_k=top_k)
    context = service.build_context(results, max_chars=max_chars) if results else ""
    return {"context": context, "sources": results}