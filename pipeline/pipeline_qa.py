"""
RAG Retrieval Service with Reranking
=====================================
Enhanced retrieval service with BGE-v2 reranker integration for better result quality.

Usage:
    from RAG_system.pipeline.rag_pipeline import RAGPipeline
    from RAG_system.pipeline.rag_qa_engine import RAGRetrievalService

    pipeline = RAGPipeline(output_dir="data")
    retriever = RAGRetrievalService(
        pipeline, 
        enable_rerank=True,
        rerank_model="BAAI/bge-reranker-v2-m3"
    )
    results = retriever.retrieve("Tìm điểm chính?", top_k=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """
    Dịch vụ Retrieval với reranking: tìm kiếm Top-K đoạn liên quan từ FAISS index,
    tùy chọn rerank bằng BGE-v2, và cung cấp tiện ích build context + payload hiển thị cho UI.
    """

    def __init__(
        self, 
        pipeline: RAGPipeline,
        enable_rerank: bool = True,
        rerank_model: Optional[str] = None,
        rerank_top_r: int = 20,  # Changed from rerank_top_k_multiplier
        rerank_device: str = "cpu",
        rerank_batch_size: int = 8,
    ):
        """
        Args:
            pipeline: RAGPipeline instance
            enable_rerank: Whether to enable reranking
            rerank_model: BGE-v2 reranker model name (e.g., "BAAI/bge-reranker-v2-m3")
            rerank_top_r: Number of results to retrieve before reranking (default: 20)
            rerank_device: "cpu" or "cuda"
            rerank_batch_size: Batch size for reranking
        """
        self.pipeline = pipeline
        self.enable_rerank = enable_rerank
        self.rerank_top_r = rerank_top_r  # Changed from rerank_top_k_multiplier
        self._reranker = None
        
        # Initialize reranker if enabled
        if self.enable_rerank:
            try:
                from rerankers.bge_v2_reranker import BGEV2Reranker
                self._reranker = BGEV2Reranker(
                    model_name=rerank_model,
                    device=rerank_device,
                    batch_size=rerank_batch_size,
                    use_fp16=(rerank_device == "cuda")
                )
                logger.info(f"Reranker initialized: {rerank_model or 'default model'} | top_r={rerank_top_r}")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}. Continuing without reranking.")
                self.enable_rerank = False
                self._reranker = None

    def _rerank_results(self, query_text: str, results: List[Dict[str, Any]], 
                       target_top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank results using BGE-v2 reranker.
        
        Args:
            query_text: Original query
            results: List of retrieval results
            target_top_k: Number of results to return after reranking
            
        Returns:
            Reranked and truncated results with rerank_score added
        """
        if not self._reranker or not results:
            return results[:target_top_k]
        
        try:
            # Log BEFORE reranking
            logger.info(f"\n{'='*80}")
            logger.info(f"BEFORE RERANKING - Top {min(10, len(results))} results by FAISS cosine similarity:")
            logger.info(f"{'='*80}")
            for i, r in enumerate(results[:10], 1):
                file_name = r.get("file_name", "?")
                page = r.get("page_number", "?")
                similarity = r.get("similarity_score", 0.0)
                text_preview = r.get("text", "")[:100].replace("\n", " ")
                logger.info(
                    f"  [{i:2d}] {file_name} (page {page}) | "
                    f"similarity: {similarity:.4f} | "
                    f"text: {text_preview}..."
                )
            
            # Rerank using the text field
            reranked = self._reranker.rerank(
                query=query_text,
                candidates=results,
                top_k=target_top_k,
                text_key="text"
            )
            
            # Log AFTER reranking
            logger.info(f"\n{'='*80}")
            logger.info(f"AFTER RERANKING - Top {len(reranked)} results by BGE-v2 reranker:")
            logger.info(f"{'='*80}")
            for i, r in enumerate(reranked, 1):
                file_name = r.get("file_name", "?")
                page = r.get("page_number", "?")
                similarity = r.get("similarity_score", 0.0)
                rerank_score = r.get("rerank_score", 0.0)
                text_preview = r.get("text", "")[:100].replace("\n", " ")
                logger.info(
                    f"  [{i:2d}] {file_name} (page {page}) | "
                    f"similarity: {similarity:.4f} | rerank: {rerank_score:.4f} | "
                    f"text: {text_preview}..."
                )
            
            logger.info(f"{'='*80}")
            logger.info(
                f"Reranking summary: {len(results)} candidates → {len(reranked)} results | "
                f"Rerank score range: [{reranked[-1]['rerank_score']:.4f}, {reranked[0]['rerank_score']:.4f}]"
            )
            logger.info(f"{'='*80}\n")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original results.")
            return results[:target_top_k]

    # ---------- Retrieval utilities ----------
    def _match_metadata_for_vectors(self, vectors_file: Path) -> Optional[Path]:
        """
        Tìm file metadata_map tương ứng với vectors bằng pattern tên file.
        """
        name = vectors_file.name
        if "_vectors_" not in name:
            return None
        candidate = self.pipeline.vectors_dir / name.replace("_vectors_", "_metadata_map_").replace(".faiss", ".pkl")
        return candidate if candidate.exists() else None

    def get_all_index_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Lấy tất cả cặp (faiss_index, metadata_map) hợp lệ trong thư mục vectors.
        """
        index_pairs = []
        faiss_files = list(self.pipeline.vectors_dir.glob("*_vectors_*.faiss"))

        for vf in faiss_files:
            mf = self._match_metadata_for_vectors(vf)
            if mf is not None and mf.exists():
                if self._test_faiss_file(vf):
                    index_pairs.append((vf, mf))
                else:
                    logger.warning(f"Skipping corrupted FAISS file: {vf}")

        index_pairs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
        return index_pairs

    def _test_faiss_file(self, faiss_file: Path) -> bool:
        """Test if a FAISS file can be loaded without errors."""
        try:
            import faiss
            faiss.read_index(str(faiss_file))
            return True
        except Exception as e:
            logger.warning(f"FAISS file test failed for {faiss_file}: {e}")
            return False

    def build_context(self, results: List[Dict[str, Any]], max_chars: int = 8000) -> str:
        """
        Tạo chuỗi context từ danh sách kết quả retrieval.
        Enhanced với rerank_score nếu có.
        """
        parts: List[str] = []
        total = 0
        
        for i, r in enumerate(results, 1):
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = r.get("similarity_score", 0.0)
            rerank_score = r.get("rerank_score")
            text = r.get("text", "")

            # Enhanced source attribution
            provenance = r.get("provenance")
            if provenance and isinstance(provenance, dict):
                page_nums = provenance.get("page_numbers", [])
                if page_nums:
                    page_range = f"pages {min(page_nums)}-{max(page_nums)}" if len(page_nums) > 1 else f"page {page_nums[0]}"
                else:
                    page_range = f"page {page}"
                source_blocks = provenance.get("source_blocks", [])
                block_info = f", blocks {len(source_blocks)}" if source_blocks else ""
                source_info = f"{file_name} ({page_range}{block_info})"
            else:
                source_info = f"{file_name} (page {page})"

            # Table data indicator
            table_note = " [TABLE DATA]" if r.get("table_data") else ""
            
            # Score display
            if rerank_score is not None:
                score_str = f"similarity {score:.3f}, rerank {rerank_score:.3f}"
            else:
                score_str = f"score {score:.3f}"

            piece = f"[{i}] Source: {source_info}, {score_str}{table_note}\n{text}"
            parts.append(piece)
            total += len(piece)
            if total > max_chars:
                break
                
        return "\n\n".join(parts)

    def to_ui_items(self, results: List[Dict[str, Any]], max_text_len: int = 3000) -> List[Dict[str, Any]]:
        """
        Chuyển danh sách kết quả sang dạng dễ hiển thị ở UI.
        Includes rerank_score if available.
        """
        ui_items: List[Dict[str, Any]] = []
        for r in results:
            item = {
                "title": f"{r.get('file_name', '?')} - trang {r.get('page_number', '?')}",
                "snippet": r.get("text", ""),
                "text": r.get("text", ""),
                "file_name": r.get("file_name", "?"),
                "page_number": r.get("page_number", "?"),
                "similarity_score": round(float(r.get("similarity_score", 0.0)), 4),
                "distance": round(float(r.get("distance", 0.0)), 4),
            }
            
            # Add rerank score if available
            if "rerank_score" in r:
                item["rerank_score"] = round(float(r["rerank_score"]), 4)
            
            ui_items.append(item)
        return ui_items
    
    def adaptive_retrieve(self, query_text: str, llm_callable=None, 
                        max_chars: int = 8000, top_k: int = 5) -> Dict[str, Any]:
        """
        Adaptive retrieval with reranking support.
        Retrieves top_r results, then reranks to get top_k.
        
        Args:
            query_text: Query text to search
            llm_callable: Optional LLM for query analysis
            max_chars: Maximum context length
            top_k: Number of final results to return (default: 5)
        """
        from pipeline.query_router import QueryRouter
        
        # Analyze query
        router = QueryRouter(llm_callable=llm_callable)
        analysis = router.analyze_query(query_text)
        
        logger.info(f"Query routing: {analysis.query_type} (confidence: {analysis.confidence:.2f})")
        
        if not analysis.requires_retrieval:
            return {
                "context": "",
                "sources": [],
                "routing_info": {
                    "query_type": analysis.query_type,
                    "reasoning": analysis.reasoning,
                    "retrieval_used": False,
                    "rerank_used": False,
                }
            }
        
        index_pairs = self.get_all_index_pairs()
        if not index_pairs:
            logger.warning("No FAISS indexes found")
            return {
                "context": "",
                "sources": [],
                "routing_info": {
                    "query_type": analysis.query_type,
                    "reasoning": "No indexes available",
                    "retrieval_used": False,
                    "rerank_used": False,
                }
            }
        
        # Use suggested top_k or fall back to router's suggestion
        final_top_k = analysis.suggested_top_k if analysis.suggested_top_k else top_k
        
        # Execute retrieval based on query type
        if analysis.query_type == "simple_factual":
            results = self._simple_retrieval(query_text, index_pairs, final_top_k)
        else:
            results = self._complex_retrieval(query_text, index_pairs, final_top_k)
        
        # Build context
        context = self.build_context(results, max_chars=max_chars)
        
        return {
            "context": context,
            "sources": results,
            "routing_info": {
                "query_type": analysis.query_type,
                "reasoning": analysis.reasoning,
                "retrieval_used": True,
                "rerank_used": self.enable_rerank,
                "num_sources": len(results),
                "top_k": final_top_k,
                "initial_retrieval_k": self.rerank_top_r if self.enable_rerank else final_top_k
            }
        }

    def _simple_retrieval(self, query_text: str, index_pairs: list, 
                        top_k: int) -> list:
        """
        Single-step retrieval with reranking support.
        Retrieves top_r results, then reranks to get top_k.
        """
        # Determine how many to retrieve initially
        retrieval_k = self.rerank_top_r if self.enable_rerank else top_k
        
        logger.info(f"Simple retrieval: retrieving {retrieval_k} results, target top_k={top_k}")
        
        all_results = []
        for faiss_file, metadata_file in index_pairs:
            try:
                results = self.pipeline.search_similar(
                    faiss_file=faiss_file,
                    metadata_map_file=metadata_file,
                    query_text=query_text,
                    top_k=retrieval_k
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error searching in {faiss_file}: {e}")
                continue
        
        # Sort by similarity
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        initial_results = all_results[:retrieval_k]
        
        # Rerank if enabled
        if self.enable_rerank and initial_results:
            logger.info(f"Reranking {len(initial_results)} results to get top {top_k}")
            return self._rerank_results(query_text, initial_results, top_k)
        
        return initial_results[:top_k]

    def _complex_retrieval(self, query_text: str, index_pairs: list, 
                        top_k: int) -> list:
        """
        Multi-step retrieval with reranking support.
        Retrieves broader set, diversifies, then reranks to top_k.
        """
        # Step 1: Broader initial retrieval
        if self.enable_rerank:
            # For complex queries, retrieve 2x top_r for better diversity
            initial_k = min(self.rerank_top_r * 2, 50)
        else:
            initial_k = min(top_k * 2, 50)
        
        logger.info(f"Complex retrieval: retrieving {initial_k} results, target top_k={top_k}")
        
        all_results = []
        for faiss_file, metadata_file in index_pairs:
            try:
                results = self.pipeline.search_similar(
                    faiss_file=faiss_file,
                    metadata_map_file=metadata_file,
                    query_text=query_text,
                    top_k=initial_k
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error searching in {faiss_file}: {e}")
                continue
        
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Step 2: Diversity filtering to top_r for reranking
        diverse_k = self.rerank_top_r if self.enable_rerank else top_k
        diverse_results = self._diversify_results(all_results, diverse_k)
        
        # Step 3: Rerank if enabled
        if self.enable_rerank and diverse_results:
            logger.info(f"Reranking {len(diverse_results)} diverse results to get top {top_k}")
            final_results = self._rerank_results(query_text, diverse_results, top_k)
            logger.info(
                f"Complex retrieval pipeline: "
                f"{len(all_results)} initial → {len(diverse_results)} diverse → {len(final_results)} reranked"
            )
            return final_results
        
        logger.info(f"Complex retrieval: {len(all_results)} initial → {len(diverse_results)} diverse results")
        return diverse_results[:top_k]

    def _diversify_results(self, results: list, target_k: int) -> list:
        """Ensure diversity in results by including chunks from different sources/pages."""
        if not results:
            return []
        
        diverse = []
        seen_sources = set()
        
        # First pass: one result per unique (file, page)
        for result in results:
            source_key = f"{result.get('file_name', '')}:{result.get('page_number', 0)}"
            if source_key not in seen_sources:
                diverse.append(result)
                seen_sources.add(source_key)
                if len(diverse) >= target_k:
                    break
        
        # Second pass: fill remaining slots
        if len(diverse) < target_k:
            for result in results:
                if result not in diverse:
                    diverse.append(result)
                    if len(diverse) >= target_k:
                        break
        
        return diverse


def fetch_retrieval(
    query_text: str, 
    top_k: int = 5, 
    max_chars: int = 8000,
    enable_rerank: bool = False,
    rerank_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Hàm tiện ích để retrieval từ FAISS indexes với reranking option.

    Args:
        query_text: Câu hỏi cần tìm
        top_k: Số lượng kết quả trả về
        max_chars: Độ dài tối đa của context
        enable_rerank: Có sử dụng reranking không
        rerank_model: Model reranker (mặc định: BAAI/bge-reranker-v2-m3)

    Returns:
        Dict với keys: "context" (str), "sources" (list), "rerank_used" (bool)
    """
    try:
        from pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        retriever = RAGRetrievalService(
            pipeline,
            enable_rerank=enable_rerank,
            rerank_model=rerank_model
        )

        index_pairs = retriever.get_all_index_pairs()
        if not index_pairs:
            logger.warning("Không tìm thấy FAISS index nào")
            return {"context": "", "sources": [], "rerank_used": False}

        # Use _simple_retrieval which handles reranking internally
        top_results = retriever._simple_retrieval(query_text, index_pairs, top_k)

        # Build context
        context = retriever.build_context(top_results, max_chars=max_chars)

        return {
            "context": context,
            "sources": top_results,
            "rerank_used": enable_rerank and retriever._reranker is not None
        }

    except Exception as e:
        logger.error(f"Lỗi trong fetch_retrieval: {e}")
        return {"context": "", "sources": [], "rerank_used": False}