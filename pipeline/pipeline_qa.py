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


logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """
    Dịch vụ Retrieval thuần: tìm kiếm Top-K đoạn liên quan từ FAISS index và
    cung cấp tiện ích build context + payload hiển thị cho UI.
    Không gọi LLM tại đây (UI hoặc lớp khác sẽ làm việc đó).
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

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

    def get_all_index_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Lấy tất cả cặp (faiss_index, metadata_map) hợp lệ trong thư mục vectors.
        Trả về list các cặp, không chỉ latest.
        """
        index_pairs = []
        faiss_files = list(self.pipeline.vectors_dir.glob("*_vectors_*.faiss"))

        for vf in faiss_files:
            mf = self._match_metadata_for_vectors(vf)
            if mf is not None and mf.exists():
                # Test if FAISS file can be loaded
                if self._test_faiss_file(vf):
                    index_pairs.append((vf, mf))
                else:
                    logger.warning(f"Skipping corrupted FAISS file: {vf}")

        # Sort by modification time (newest first)
        index_pairs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
        return index_pairs

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


def fetch_retrieval(query_text: str, top_k: int = 5, max_chars: int = 8000) -> Dict[str, Any]:
    """
    Hàm tiện ích để retrieval từ FAISS indexes.
    Tự động tìm FAISS index mới nhất và thực hiện search.

    Args:
        query_text: Câu hỏi cần tìm
        top_k: Số lượng kết quả trả về
        max_chars: Độ dài tối đa của context

    Returns:
        Dict với keys: "context" (str), "sources" (list)
    """
    try:
        # Khởi tạo pipeline và retriever
        from pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        retriever = RAGRetrievalService(pipeline)

        # Lấy tất cả cặp FAISS indexes hợp lệ
        index_pairs = retriever.get_all_index_pairs()
        if not index_pairs:
            logger.warning("Không tìm thấy FAISS index nào")
            return {"context": "", "sources": []}

        # Search across tất cả indexes và combine results
        all_results = []
        for faiss_file, metadata_file in index_pairs:
            try:
                results = pipeline.search_similar(
                    faiss_file=faiss_file,
                    metadata_map_file=metadata_file,
                    query_text=query_text,
                    top_k=top_k * 2  # Lấy nhiều hơn để có thể chọn top-k tốt nhất
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Lỗi khi search trong {faiss_file}: {e}")
                continue

        # Sort tất cả results by similarity score và lấy top-k
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        top_results = all_results[:top_k]

        # Build context
        context = retriever.build_context(top_results, max_chars=max_chars)

        # Convert to UI format
        sources = retriever.to_ui_items(top_results)

        return {
            "context": context,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Lỗi trong fetch_retrieval: {e}")
        return {"context": "", "sources": []}