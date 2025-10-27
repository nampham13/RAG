"""
RAG Retrieval Service with Reranking
=====================================
Enhanced retrieval service with reranker integration for better result quality.

Usage:
    from RAG_system.pipeline.rag_pipeline import RAGPipeline
    from RAG_system.pipeline.rag_qa_engine import RAGRetrievalService

    pipeline = RAGPipeline(output_dir="data")
    retriever = RAGRetrievalService(
        pipeline, 
        enable_rerank=True,
        reranker_type="bge"  # or "jina"
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
    tùy chọn rerank bằng BGE hoặc Jina, và cung cấp tiện ích build context + payload hiển thị cho UI.
    """

    def __init__(
        self, 
        pipeline: RAGPipeline,
        enable_rerank: bool = True,
        reranker_type: str = "bge",  # "bge" or "jina"
        rerank_top_r: int = 50,
        rerank_batch_size: Optional[int] = None,
    ):
        """
        Args:
            pipeline: RAGPipeline instance
            enable_rerank: Whether to enable reranking
            reranker_type: Type of reranker ("bge" or "jina")
            rerank_top_r: Number of results to retrieve before reranking (default: 50)
            rerank_batch_size: Batch size for reranking (optional, uses model defaults)
        """
        self.pipeline = pipeline
        self.enable_rerank = enable_rerank
        self.rerank_top_r = rerank_top_r
        self.reranker_type = reranker_type
        self._reranker = None
        
        # Initialize reranker if enabled
        if self.enable_rerank:
            try:
                from rerankers.reranker_factory import RerankerFactory
                
                factory = RerankerFactory()
                
                if reranker_type.lower() == "bge":
                    self._reranker = factory.create_bge()
                    logger.info(f"BGE reranker initialized (CPU) | top_r={rerank_top_r}")
                elif reranker_type.lower() == "jina":
                    self._reranker = factory.create_jina()
                    logger.info(f"Jina reranker initialized (CPU) | top_r={rerank_top_r}")
                else:
                    raise ValueError(f"Unsupported reranker type: {reranker_type}")
                
                # Override batch size if specified
                if rerank_batch_size is not None:
                    self._reranker.profile.batch_size = rerank_batch_size
                    
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}. Continuing without reranking.")
                self.enable_rerank = False
                self._reranker = None

    def _rerank_results(self, query_text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using configured reranker.
        ONLY reorders, does NOT filter - returns ALL input results in new order.
        
        Args:
            query_text: Original query
            results: List of retrieval results
            
        Returns:
            Reranked results (same count as input) with rerank_score added
        """
        if not self._reranker or not results:
            return results
        
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
            
            # Rerank ALL results without filtering
            reranked = self._reranker.rerank(
                query=query_text,
                candidates=results,
                top_k=len(results),  # Return ALL results, just reordered
                text_key="text"
            )
            
            # Log AFTER reranking
            logger.info(f"\n{'='*80}")
            logger.info(f"AFTER RERANKING - Top {min(10, len(reranked))} results by {self.reranker_type.upper()} reranker:")
            logger.info(f"{'='*80}")
            for i, r in enumerate(reranked[:10], 1):
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
                f"Reranking summary: {len(results)} candidates → {len(reranked)} results (reordered) | "
                f"Rerank score range: [{reranked[-1]['rerank_score']:.4f}, {reranked[0]['rerank_score']:.4f}]"
            )
            logger.info(f"{'='*80}\n")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original results.")
            return results

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
                        max_chars: int = 8000, top_k: int = None) -> Dict[str, Any]:
        """
        Adaptive retrieval with reranking support.
        
        Flow:
        1. Analyze query to get suggested_top_k
        2. Retrieve rerank_top_r results (e.g., 50)
        3. Rerank ALL retrieved results (just reorder, no filtering)
        4. Take top suggested_top_k from reranked results for LLM
        
        Args:
            query_text: Query text to search
            llm_callable: Optional LLM for query analysis
            max_chars: Maximum context length
            top_k: Override for final result count (if None, uses router's suggested_top_k)
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
        
        # Determine final output size - ONLY from analysis.suggested_top_k or override
        # final_top_k = top_k if top_k is not None else analysis.suggested_top_k
        final_top_k = analysis.suggested_top_k if analysis.suggested_top_k is not None else top_k
        
        logger.info(f"Final LLM input will be top {final_top_k} results (from router suggestion)")
        
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
                "reranker_type": self.reranker_type if self.enable_rerank else None,
                "num_sources": len(results),
                "final_top_k": final_top_k,
                "initial_retrieval_k": self.rerank_top_r if self.enable_rerank else final_top_k
            }
        }

    def _simple_retrieval(self, query_text: str, index_pairs: list, 
                        final_top_k: int) -> list:
        """
        Single-step retrieval with reranking support.
        
        Flow:
        1. Retrieve rerank_top_r results (e.g., 50)
        2. Rerank ALL retrieved results (reorder only)
        3. Return top final_top_k from reranked results
        
        Args:
            query_text: Search query
            index_pairs: List of (faiss_file, metadata_file) tuples
            final_top_k: Number of results to return to LLM (from analysis.suggested_top_k)
        """
        # Step 1: Retrieve candidates for reranking
        retrieval_k = self.rerank_top_r if self.enable_rerank else final_top_k
        
        logger.info(f"Simple retrieval: fetching {retrieval_k} candidates")
        
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
        
        # Step 2: Rerank if enabled (reorders ALL, no filtering)
        if self.enable_rerank and initial_results:
            logger.info(f"Reranking {len(initial_results)} results (will reorder all)")
            reranked_results = self._rerank_results(query_text, initial_results)
            
            # Step 3: Take top final_top_k AFTER reranking
            final_results = reranked_results[:final_top_k]
            logger.info(f"Returning top {len(final_results)} results for LLM")
            return final_results
        
        # No reranking: just return top final_top_k
        return initial_results[:final_top_k]

    def _complex_retrieval(self, query_text: str, index_pairs: list, 
                        final_top_k: int) -> list:
        """
        Multi-step retrieval with reranking support.
        
        Flow:
        1. Retrieve broader set (2x rerank_top_r for diversity)
        2. Diversify to rerank_top_r results
        3. Rerank ALL diversified results (reorder only)
        4. Return top final_top_k from reranked results
        
        Args:
            query_text: Search query
            index_pairs: List of (faiss_file, metadata_file) tuples
            final_top_k: Number of results to return to LLM (from analysis.suggested_top_k)
        """
        # Step 1: Broader initial retrieval for diversity
        if self.enable_rerank:
            initial_k = min(self.rerank_top_r * 2, 100)
        else:
            initial_k = min(final_top_k * 2, 50)
        
        logger.info(f"Complex retrieval: fetching {initial_k} candidates for diversity")
        
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
        
        # Step 2: Diversify to rerank_top_r for reranking input
        diverse_k = self.rerank_top_r if self.enable_rerank else final_top_k
        diverse_results = self._diversify_results(all_results, diverse_k)
        logger.info(f"Diversified to {len(diverse_results)} results")
        
        # Step 3: Rerank if enabled (reorders ALL, no filtering)
        if self.enable_rerank and diverse_results:
            logger.info(f"Reranking {len(diverse_results)} diverse results (will reorder all)")
            reranked_results = self._rerank_results(query_text, diverse_results)
            
            # Step 4: Take top final_top_k AFTER reranking
            final_results = reranked_results[:final_top_k]
            logger.info(
                f"Complex retrieval pipeline: "
                f"{len(all_results)} initial → {len(diverse_results)} diverse → "
                f"{len(reranked_results)} reranked → {len(final_results)} final for LLM"
            )
            return final_results
        
        # No reranking: just return top final_top_k
        logger.info(f"Complex retrieval: {len(all_results)} initial → {len(diverse_results)} diverse results")
        return diverse_results[:final_top_k]

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
    max_chars: int = 8000
) -> Dict[str, Any]:
    """
    Hàm tiện ích để retrieval từ FAISS indexes.

    Args:
        query_text: Câu hỏi cần tìm
        top_k: Số lượng kết quả trả về
        max_chars: Độ dài tối đa của context

    Returns:
        Dict với keys: "context" (str), "sources" (list)
    """
    try:
        from pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        retriever = RAGRetrievalService(
            pipeline,
            enable_rerank=False
        )

        index_pairs = retriever.get_all_index_pairs()
        if not index_pairs:
            logger.warning("Không tìm thấy FAISS index nào")
            return {"context": "", "sources": []}

        # Use _simple_retrieval which handles reranking internally
        top_results = retriever._simple_retrieval(query_text, index_pairs, top_k)

        # Build context
        context = retriever.build_context(top_results, max_chars=max_chars)

        return {
            "context": context,
            "sources": top_results
        }

    except Exception as e:
        logger.error(f"Lỗi trong fetch_retrieval: {e}")
        return {"context": "", "sources": []}