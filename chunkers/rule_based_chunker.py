"""
Rule-Based Chunker (Compact Version)
====================================
Phân chia tài liệu theo cấu trúc: headings, lists, tables, code blocks.
Sử dụng pattern matching và metadata từ loader.
"""

import re, hashlib
from typing import List, Optional, Tuple
from .base_chunker import BaseChunker
from .model import (
    Chunk, ChunkSet, ChunkType, ChunkStrategy,
    ProvenanceAgg, BlockSpan, Score
)
from loaders.model.document import PDFDocument
from loaders.model.block import Block


class RuleBasedChunker(BaseChunker):
    """Chunker dựa trên cấu trúc tài liệu (structure-first)."""

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm"
    ):
        super().__init__(max_tokens, overlap_tokens)
        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load(spacy_model)
            except Exception:
                self._nlp = None

        # regex patterns (tối giản)
        self.heading_patterns = [
            re.compile(r"^(?:#{1,6}\s+|(?:Chapter|Section|Part)\s+\d+|[A-Z\s]{5,}|(?:\d+\.)+\s+[A-Z0-9])"),
        ]
        self.list_patterns = [
            re.compile(r"^[\s]*(?:[-•*◦▪▫➢➤■□●○]|\d+[.)])\s+\w+", re.MULTILINE),
        ]
        self.table_patterns = [
            re.compile(r"\|.+\|"), re.compile(r"(\t[^\n]+\t)+"),
        ]
        self.code_patterns = [
            re.compile(r"```[\s\S]+?```"), re.compile(r"^(?: {4,}|\t).+", re.MULTILINE),
        ]

    # -------------------------------
    # Public API
    # -------------------------------
    def chunk(self, document: PDFDocument) -> ChunkSet:
        # Lưu file_path để sử dụng trong chunk_blocks  
        self._current_file_path = document.file_path
        
        cs = ChunkSet(
            doc_id=document.meta.get("doc_id", "unknown"),
            file_path=document.file_path,
            chunk_strategy="rule_based",
        )
        blocks = [b for p in document.pages for b in p.blocks if b.text.strip()]
        for ch in self.chunk_blocks(blocks, cs.doc_id):
            cs.add_chunk(ch)
        cs.link_chunks()
        return cs

    def chunk_blocks(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        """Group blocks theo cấu trúc và tạo ChunkSet."""
        groups = self._group_blocks(blocks)
        chunks, idx = [], 0
        for gtype, gblocks, title in groups:
            chs = self._make_chunks(gtype, gblocks, doc_id, idx, title)
            chunks.extend(chs)
            idx += len(chs)
        return chunks

    # -------------------------------
    # Core grouping logic
    # -------------------------------
    def _group_blocks(self, blocks: List[Block]) -> List[Tuple[str, List[Block], Optional[str]]]:
        """Nhóm các block theo pattern nhận dạng (heading/list/table/code/paragraph)."""
        groups, cur_blocks, cur_type, cur_title = [], [], None, None

        for b in blocks:
            btype = self._detect_block_type(b)
            if btype == "heading":
                if cur_blocks:
                    groups.append((cur_type, cur_blocks, cur_title))
                cur_title, cur_type, cur_blocks = b.text.strip(), "heading", [b]
                continue
            if btype in {"table", "code"}:
                if cur_blocks:
                    groups.append((cur_type, cur_blocks, cur_title))
                groups.append((btype, [b], cur_title))
                cur_blocks, cur_type = [], None
                continue
            if btype == "list":
                if cur_type == "list":
                    cur_blocks.append(b)
                else:
                    if cur_blocks:
                        groups.append((cur_type, cur_blocks, cur_title))
                    cur_blocks, cur_type = [b], "list"
                continue
            # paragraph default
            if cur_type not in {"paragraph", "list"}:
                if cur_blocks:
                    groups.append((cur_type, cur_blocks, cur_title))
                cur_blocks, cur_type = [b], "paragraph"
            else:
                text = "\n\n".join(bl.text for bl in cur_blocks)
                if self.estimate_tokens(text) + self.estimate_tokens(b.text) > self.max_tokens:
                    groups.append((cur_type, cur_blocks, cur_title))
                    cur_blocks = [b]
                else:
                    cur_blocks.append(b)

        if cur_blocks:
            groups.append((cur_type, cur_blocks, cur_title))
        return groups

    # -------------------------------
    # Block classification
    # -------------------------------
    def _detect_block_type(self, block: Block) -> str:
        text = block.text.strip()
        if not text:
            return "empty"

        # metadata hint - check both 'type' and 'block_type'
        if block.metadata:
            if block.metadata.get("type") in ("heading", "table", "list", "code"):
                return block.metadata["type"]
            if block.metadata.get("block_type") == "table":
                return "table"

        # Check if block is TableBlock
        from loaders.model.block import TableBlock
        if isinstance(block, TableBlock):
            return "table"

        # regex detection
        for p in self.code_patterns:
            if p.search(text):
                return "code"
        for p in self.table_patterns:
            if p.search(text):
                return "table"
        for p in self.heading_patterns:
            if p.match(text):
                return "heading"
        for p in self.list_patterns:
            if p.match(text):
                return "list"

        # NLP-based heading fallback
        if self._nlp and len(text.split()) < 15:
            doc = self._nlp(text)
            if all(t.pos_ in ("NOUN","PROPN","ADJ","DET","PUNCT") for t in doc):
                return "heading"

        return "paragraph"

    # -------------------------------
    # Chunk creation
    # -------------------------------
    def _make_chunks(
        self,
        gtype: str,
        blocks: List[Block],
        doc_id: str,
        start_idx: int,
        title: Optional[str],
    ) -> List[Chunk]:
        text = "\n\n".join(b.text.strip() for b in blocks)
        toks = self.estimate_tokens(text)
        if toks <= self.max_tokens:
            return [self._build_chunk(text, blocks, gtype, doc_id, start_idx, title)]

        # split by block nếu vượt token limit
        out, cur_blocks, cur_toks, idx = [], [], 0, start_idx
        for b in blocks:
            btok = self.estimate_tokens(b.text)
            if cur_toks + btok > self.max_tokens and cur_blocks:
                subtext = "\n\n".join(x.text for x in cur_blocks)
                out.append(self._build_chunk(subtext, cur_blocks, gtype, doc_id, idx, title))
                cur_blocks, cur_toks, idx = [b], btok, idx + 1
            else:
                cur_blocks.append(b)
                cur_toks += btok
        if cur_blocks:
            subtext = "\n\n".join(x.text for x in cur_blocks)
            out.append(self._build_chunk(subtext, cur_blocks, gtype, doc_id, idx, title))
        return out

    def _build_chunk(
        self,
        text: str,
        blocks: List[Block],
        gtype: str,
        doc_id: str,
        idx: int,
        title: Optional[str],
    ) -> Chunk:

        prov = ProvenanceAgg(doc_id=doc_id, file_path=getattr(self, '_current_file_path', None))
        for b in blocks:
            prov.add_span(BlockSpan(
                block_id=b.stable_id or f"block_{id(b)}",
                start_char=0, end_char=len(b.text),
                page_number=b.metadata.get("page_number") if b.metadata else None,
            ))

        score = Score(
            coherence_score=0.8,
            completeness_score=0.9,
            token_ratio=min(1.0, self.estimate_tokens(text) / self.max_tokens),
            structural_integrity=0.9,
        )

        strat = {
            "heading": ChunkStrategy.HEADING_BASED,
            "list": ChunkStrategy.LIST_BASED,
            "table": ChunkStrategy.TABLE_BASED,
            "code": ChunkStrategy.TOKEN_LENGTH,
        }.get(gtype, ChunkStrategy.PARAGRAPH_BASED)

        metadata = {"group_type": gtype, "block_count": len(blocks), "chunk_index": idx}

        # Nếu là bảng, lưu payload và embedding_text
        if gtype == "table" and blocks:
            table_schema = None
            # Ưu tiên lấy TableSchema từ block.table (TableBlock)
            from loaders.model.block import TableBlock
            if isinstance(blocks[0], TableBlock) and hasattr(blocks[0], "table"):
                table_schema = blocks[0].table
            # Nếu không phải TableBlock, thử lấy từ metadata
            elif hasattr(blocks[0], "metadata") and blocks[0].metadata:
                table_schema = blocks[0].metadata.get("table_schema")
            
            if table_schema:
                metadata["table_payload"] = table_schema
                
                # Generate embedding text and check token budget
                rows = getattr(table_schema, "rows", [])
                header = getattr(table_schema, "header", [])
                
                # Build embedding text row by row
                embedding_lines = []
                header_text = " | ".join(str(h) for h in header)
                embedding_lines.append(header_text)
                
                # Track tokens for budget enforcement
                current_tokens = self.estimate_tokens(header_text)
                included_rows = []
                truncated = False
                
                for idx, r in enumerate(rows):
                    if hasattr(r, "cells"):
                        line = " | ".join([str(c.value) for c in r.cells])
                        row_tokens = self.estimate_tokens(line)
                        
                        # Check if adding this row would exceed budget
                        if current_tokens + row_tokens > self.max_tokens:
                            truncated = True
                            break
                        
                        embedding_lines.append(line)
                        included_rows.append(idx)
                        current_tokens += row_tokens
                
                metadata["embedding_text"] = "\n".join(embedding_lines)
                
                # Add metadata about truncation
                metadata["table_text_mode"] = "pipe_separated"
                metadata["row_priority"] = "sequential"
                metadata["header_included"] = True
                metadata["total_rows"] = len(rows)
                metadata["included_rows"] = len(included_rows)
                
                if truncated:
                    metadata["truncated"] = True
                    metadata["truncation_reason"] = "token_budget_exceeded"
                    metadata["row_range"] = [0, len(included_rows)]
                else:
                    metadata["truncated"] = False
                    metadata["row_range"] = [0, len(rows)]
                
                # Add cell-level provenance for precise citation
                cell_provenance = []
                page_num = blocks[0].metadata.get("page_number") if blocks[0].metadata else None
                
                for row_idx in included_rows:
                    if row_idx < len(rows):
                        row = rows[row_idx]
                        if hasattr(row, "cells"):
                            for cell in row.cells:
                                cell_provenance.append({
                                    "row": getattr(cell, "row", row_idx + 1),
                                    "col": getattr(cell, "col", 0),
                                    "value": str(getattr(cell, "value", "")),
                                    "page": page_num
                                })
                
                if cell_provenance:
                    metadata["cell_provenance"] = cell_provenance
        return Chunk(
            chunk_id=self._make_chunk_id(doc_id, idx, text),
            text=text,
            token_count=self.estimate_tokens(text),
            char_count=len(text),
            chunk_type=ChunkType.STRUCTURAL,
            strategy=strat,
            provenance=prov,
            score=score,
            section_title=title,
            metadata=metadata,
        )

    # -------------------------------
    # Helpers
    # -------------------------------
    def _make_chunk_id(self, doc_id: str, idx: int, text: str) -> str:
        raw = f"{doc_id}|{idx}|{text[:60]}"
        return f"chunk_{doc_id}_{idx}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)
