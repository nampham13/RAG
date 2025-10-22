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
        max_per_chunk = max_chars  # Don't limit per chunk, use full max_chars
        
        for i, r in enumerate(results, 1):
            file_name = r.get("file_name", "?")
            page = r.get("page_number", "?")
            score = r.get("similarity_score", 0.0)
            text = r.get("text", "")

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

    def to_ui_items(self, results: List[Dict[str, Any]], max_text_len: int = 3000) -> List[Dict[str, Any]]:
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
            logger.info(f"DEBUG to_ui_items: text length from results = {len(text)}, first 100 chars = {text[:100]}")
            # no truncate
            snippet = text
            ui_items.append(
                {
                    "title": f"{file_name} - trang {page}",
                    "snippet": snippet,
                    "text": text,
                    "file_name": file_name,
                    "page_number": page,
                    "similarity_score": round(score, 4),
                    "distance": round(dist, 4),
                }
            )
        return ui_items
    
    def adaptive_retrieve(self, query_text: str, llm_callable=None, 
                        max_chars: int = 8000) -> Dict[str, Any]:
        """
        Adaptive retrieval based on query complexity.
        
        Args:
            query_text: User query
            llm_callable: Optional LLM callable for query classification
            max_chars: Max characters for context
            
        Returns:
            Dict with context, sources, and routing info
        """
        from pipeline.query_router import QueryRouter
        
        # Analyze query
        router = QueryRouter(llm_callable=llm_callable)
        analysis = router.analyze_query(query_text)
        
        logger.info(f"Query routing: {analysis.query_type} (confidence: {analysis.confidence:.2f})")
        logger.info(f"Reasoning: {analysis.reasoning}")
        
        # Route based on query type
        if not analysis.requires_retrieval:
            return {
                "context": "",
                "sources": [],
                "routing_info": {
                    "query_type": analysis.query_type,
                    "reasoning": analysis.reasoning,
                    "retrieval_used": False
                }
            }
        
        # Get all index pairs
        index_pairs = self.get_all_index_pairs()
        if not index_pairs:
            logger.warning("No FAISS indexes found")
            return {
                "context": "",
                "sources": [],
                "routing_info": {
                    "query_type": analysis.query_type,
                    "reasoning": "No indexes available",
                    "retrieval_used": False
                }
            }
        
        # Execute retrieval based on query type
        if analysis.query_type == "simple_factual":
            results = self._simple_retrieval(query_text, index_pairs, analysis.suggested_top_k)
        else:  # complex_analytical
            results = self._complex_retrieval(query_text, index_pairs, analysis.suggested_top_k)
        
        # Build context
        context = self.build_context(results, max_chars=max_chars)
        
        return {
            "context": context,
            "sources": results,
            "routing_info": {
                "query_type": analysis.query_type,
                "reasoning": analysis.reasoning,
                "retrieval_used": True,
                "num_sources": len(results),
                "top_k": analysis.suggested_top_k
            }
        }

    def _simple_retrieval(self, query_text: str, index_pairs: list, 
                        top_k: int) -> list:
        """Single-step retrieval for simple factual queries"""
        all_results = []
        
        for faiss_file, metadata_file in index_pairs:
            try:
                results = self.pipeline.search_similar(
                    faiss_file=faiss_file,
                    metadata_map_file=metadata_file,
                    query_text=query_text,
                    top_k=top_k
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error searching in {faiss_file}: {e}")
                continue
        
        # Sort by similarity and return top-k
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return all_results[:top_k]

    def _complex_retrieval(self, query_text: str, index_pairs: list, 
                        top_k: int) -> list:
        """Multi-step iterative retrieval for complex analytical queries"""
        
        # Step 1: Initial broad retrieval
        initial_k = min(top_k * 2, 30)
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
        
        # Sort by similarity
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Step 2: Diversity filtering - ensure multiple sources/pages
        diverse_results = self._diversify_results(all_results, top_k)
        
        logger.info(f"Complex retrieval: {len(all_results)} initial → {len(diverse_results)} diverse results")
        
        return diverse_results

    def _diversify_results(self, results: list, target_k: int) -> list:
        """
        Ensure diversity in results by including chunks from different sources/pages.
        Helps with complex queries that may need broader context.
        """
        if not results:
            return []
        
        diverse = []
        seen_sources = set()
        
        # First pass: one result per unique (file, page) combination
        for result in results:
            file_name = result.get("file_name", "")
            page = result.get("page_number", 0)
            source_key = f"{file_name}:{page}"
            
            if source_key not in seen_sources:
                diverse.append(result)
                seen_sources.add(source_key)
                
                if len(diverse) >= target_k:
                    break
        
        # Second pass: fill remaining slots with high-scoring duplicates
        if len(diverse) < target_k:
            for result in results:
                if result not in diverse:
                    diverse.append(result)
                    if len(diverse) >= target_k:
                        break
        
        return diverse

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
                    top_k=top_k * 2
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

        sources = top_results

        return {
            "context": context,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Lỗi trong fetch_retrieval: {e}")
        return {"context": "", "sources": []}

def fetch_retrieval_with_rewrite(query_text: str, llm_callable=None, top_k: int = 10, max_chars: int = 8000) -> Dict[str, Any]:
    """
    Query rewriting version of retrieval.
    Uses LLM to rewrite the query for better retrieval before searching.
    
    Args:
        query_text: Original user query
        llm_callable: LLM callable for query rewriting
        top_k: Number of results to return
        max_chars: Max context length
        
    Returns:
        Dict with keys: "context" (str), "sources" (list), "rewrite_info" (dict)
    """
    try:
        from pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        retriever = RAGRetrievalService(pipeline)
        
        # Get FAISS indexes
        index_pairs = retriever.get_all_index_pairs()
        if not index_pairs:
            logger.warning("No FAISS indexes found")
            return {"context": "", "sources": [], "rewrite_info": {}}
        
        # Rewrite query using LLM if available
        rewritten_query = query_text
        rewrite_info = {"original_query": query_text, "rewritten_query": query_text, "rewrite_used": False}
        
        if llm_callable:
            try:
                rewrite_prompt = f"""Rewrite the following query to be more specific and optimized for document retrieval. Focus on:
- Making implicit information explicit
- Adding relevant keywords
- Breaking down complex questions
- Maintaining the original intent

Original query: {query_text}

Respond with ONLY the rewritten query, no explanation."""

                messages = [{"role": "user", "content": rewrite_prompt}]
                result = llm_callable(messages)
                rewritten_query = result.get("response", query_text).strip()
                
                # Clean up the response (remove quotes, extra whitespace)
                rewritten_query = rewritten_query.strip('"').strip("'").strip()
                
                rewrite_info["rewritten_query"] = rewritten_query
                rewrite_info["rewrite_used"] = True
                
                logger.info(f"Query rewritten: '{query_text}' -> '{rewritten_query}'")
                
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}, using original query")
                rewritten_query = query_text
        
        # Search using rewritten query
        all_results = []
        for faiss_file, metadata_file in index_pairs:
            try:
                results = pipeline.search_similar(
                    faiss_file=faiss_file,
                    metadata_map_file=metadata_file,
                    query_text=rewritten_query,
                    top_k=top_k * 2
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error searching in {faiss_file}: {e}")
                continue
        
        # Sort and get top-k
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        top_results = all_results[:top_k]
        
        # Build context
        context = retriever.build_context(top_results, max_chars=max_chars)
        
        return {
            "context": context,
            "sources": top_results,
            "rewrite_info": rewrite_info
        }
        
    except Exception as e:
        logger.error(f"Error in fetch_retrieval_with_rewrite: {e}")
        return {"context": "", "sources": [], "rewrite_info": {}}