import ftfy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
from .base import LoaderBaseModel

@dataclass
class TableCell(LoaderBaseModel):

    value: str = ""
    row: Optional[int] = None
    col: Optional[int] = None
    bbox: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def normalize(self, config: 'Optional[dict]' = None) -> 'TableCell':
        from cleantext import clean
        import ftfy
        # Chuẩn hóa value
        if self.value:
            self.value = clean(ftfy.fix_text(str(self.value)).strip())
                # Tách câu bằng spaCy nếu value đủ dài
            if len(self.value) > 40:
                    try:
                        from loaders.normalizers.spacy_utils import sent_tokenize
                        self.sentences = sent_tokenize(self.value)
                    except Exception:
                        self.sentences = []
        # Chuẩn hóa bbox
        if self.bbox and isinstance(self.bbox, (tuple, list)) and len(self.bbox) == 4:
            self.bbox = tuple(round(float(x), 2) for x in self.bbox)
        # Chuẩn hóa metadata
        if self.metadata:
            self.metadata = {k: clean(ftfy.fix_text(str(v)).strip()) for k, v in self.metadata.items() if k}
        return self
@dataclass
class TableRow(LoaderBaseModel):

    def normalize(self, config: 'Optional[dict]' = None) -> 'TableRow':
        # Chuẩn hóa từng cell
        if self.cells:
            for c in self.cells:
                if hasattr(c, 'normalize') and callable(c.normalize):
                    c.normalize(config=config)
        return self
    cells: List[TableCell] = field(default_factory=list)
    row_idx: Optional[int] = None


@dataclass
class TableSchema(LoaderBaseModel):
    @staticmethod
    def merge_split_tables(tables: List['TableSchema']) -> List['TableSchema']:
        """
        Ghép các bảng bị ngắt trang nếu:
        - Số cột giống nhau
        - Header bảng sau rỗng hoặc rất ngắn, hoặc chỉ là phần tiếp nối
        - Hai bảng nằm trên các trang liên tiếp
        - Nội dung bảng sau tiếp nối logic bảng trước
        """
        if not tables:
            return []
        merged = []
        i = 0
        while i < len(tables):
            curr = tables[i]
            # Kiểm tra bảng tiếp theo có thể ghép không
            if i + 1 < len(tables):
                nxt = tables[i+1]
                # Điều kiện ghép: cùng số cột, trang liên tiếp, header bảng sau rỗng hoặc ngắn
                same_cols = len(curr.header) == len(nxt.header) or (nxt.header and len(nxt.header) <= 2)
                consecutive_page = (nxt.page_number == curr.page_number + 1)
                nxt_header_empty = not nxt.header or all(not h.strip() for h in nxt.header) or (len(nxt.header) <= 2)
                if same_cols and consecutive_page and nxt_header_empty:
                    # Ghép rows bảng sau vào bảng trước
                    curr.rows.extend(nxt.rows)
                    i += 2
                    # Có thể cập nhật metadata nếu cần
                    continue
            merged.append(curr)
            i += 1
        return merged

    def normalize(self, config: 'Optional[dict]' = None) -> 'TableSchema':
        from cleantext import clean
        
        # 1. Chuẩn hóa header
        if self.header:
            self.header = [clean(ftfy.fix_text(str(h)).strip()) for h in self.header if h]
                # Tách câu cho từng header nếu đủ dài
            try:
                from loaders.normalizers.spacy_utils import sent_tokenize
                self.header_sentences = [sent_tokenize(h) if len(h) > 40 else [h] for h in self.header]
            except Exception:
                    self.header_sentences = [[h] for h in self.header]
        # 2. Chuẩn hóa rows
        if self.rows:
            for r in self.rows:
                if hasattr(r, 'normalize') and callable(r.normalize):
                    r.normalize(config=config)
        # 3. Chuẩn hóa metadata
        if self.metadata:
            self.metadata = {k: clean(ftfy.fix_text(str(v)).strip()) for k, v in self.metadata.items() if k}
        # 4. Chuẩn hóa markdown
        if self.markdown:
            self.markdown = clean(ftfy.fix_text(str(self.markdown)).strip())
        # 5. Tính lại stable_id, content_sha256 nếu có đủ thông tin
        if hasattr(self, 'file_path') and hasattr(self, 'page_number'):
            from ..ids import table_stable_id
            self.stable_id = table_stable_id(self)
            import hashlib
            self.content_sha256 = hashlib.sha256((str(self.header) + str(self.rows)).encode("utf-8")).hexdigest()
        return self
    """Lightweight table container with helpers for extraction and markdown rendering.
    API is kept stable for downstream code.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    page_number: int = 0
    header: List[str] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)
    bbox: Optional[Any] = None
    markdown: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    stable_id: Optional[str] = None
    content_sha256: Optional[str] = None
    text_source: Optional[str] = None
    def validate(self):
        # Có thể mở rộng kiểm tra, hiện tại chỉ trả về True để khớp base
        return True

    # ---------------- Normalization ----------------
    @staticmethod
    def normalize_table_cells(table: List[List[Any]]) -> List[List[str]]:
        """Normalize a raw 2D table (list of rows) to dense strings and trim empties."""
        cleaned: List[List[str]] = []
        for row in table:
            new_row: List[str] = []
            for cell in row:
                text = '' if cell is None else str(cell)
                # collapse internal whitespace
                text = ' '.join(text.split())
                new_row.append(text)
            cleaned.append(new_row)
        trimmed = TableSchema._drop_empty_columns(cleaned)
        # drop rows that are entirely empty
        return [row for row in trimmed if any(cell.strip() for cell in row)]

    @staticmethod
    def _drop_empty_columns(table: List[List[str]]) -> List[List[str]]:
        if not table:
            return table
        max_cols = max(len(row) for row in table)
        keep_indices = [
            idx
            for idx in range(max_cols)
            if any(idx < len(row) and row[idx].strip() for row in table)
        ]
        if not keep_indices:
            return table
        trimmed: List[List[str]] = []
        for row in table:
            trimmed.append([row[idx] if idx < len(row) else '' for idx in keep_indices])
        return trimmed

    # ---------------- Extraction backends ----------------
    @staticmethod
    def extract_tables_camelot(file_path: str, page_num_1based: int, camelot_module) -> List[Dict[str, Any]]:
        import logging
        logger = logging.getLogger("TableSchema")
        out: List[Dict[str, Any]] = []
        try:
            read_pdf = getattr(camelot_module, "read_pdf", None)
            if callable(read_pdf):
                tables = read_pdf(file_path, pages=str(page_num_1based), flavor='lattice')
                logger.info(f"Camelot lattice: page={page_num_1based}, found={getattr(tables, 'n', 0)} tables")
                if getattr(tables, 'n', 0) == 0:
                    tables = read_pdf(file_path, pages=str(page_num_1based), flavor='stream')
                    logger.info(f"Camelot stream: page={page_num_1based}, found={getattr(tables, 'n', 0)} tables")
                import collections.abc
                if getattr(tables, 'n', 0) > 0 and isinstance(tables, collections.abc.Iterable):
                    for t in tables:
                        # Use camelot table data directly
                        matrix = t.df.values.tolist() if hasattr(t, 'df') else []
                        # Try to get bbox from camelot table
                        bbox = None
                        if hasattr(t, '_bbox'):
                            bbox = tuple(t._bbox)
                        out.append({'matrix': matrix, 'bbox': bbox})
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
        return out

    @staticmethod
    def extract_tables_pdfplumber(plumber_pdf: Any, page_num_1based: int) -> List[Dict[str, Any]]:
        import logging
        logger = logging.getLogger("TableSchema")
        out: List[Dict[str, Any]] = []
        try:
            page = plumber_pdf.pages[page_num_1based - 1]
            tables = page.extract_tables() or []
            logger.info(f"pdfplumber: page={page_num_1based}, found={len(tables)} tables")
            # Try to get table bboxes from pdfplumber
            table_settings = page.find_tables()
            for idx, t in enumerate(tables):
                # Use table data directly (pdfplumber returns List[List[str]])
                matrix = t if isinstance(t, list) else []
                bbox = None
                if table_settings and idx < len(table_settings):
                    tb = table_settings[idx]
                    if hasattr(tb, 'bbox'):
                        bbox = tuple(tb.bbox)
                out.append({'matrix': matrix, 'bbox': bbox})
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        return out

    @staticmethod
    def extract_tables_for_page(
        file_path: str,
        page_num_1based: int,
        plumber_pdf: Any,
        tables_engine: str,
        camelot_module=None
    ) -> List[Dict[str, Any]]:
        """Engine selector with sensible defaults (auto → pdfplumber then camelot). 
        Returns list of dict with keys: 'matrix' (List[List[str]]) and 'bbox' (tuple or None)."""
        out: List[Dict[str, Any]] = []
        if tables_engine == "auto":
            use_plumber = (plumber_pdf is not None)
            if use_plumber:
                try:
                    tables = TableSchema.extract_tables_pdfplumber(plumber_pdf, page_num_1based)
                    if tables:
                        # keep reasonably shaped tables (>= header + 1 data row, >= 1 cols)
                        filtered_tables = [t for t in tables if len(t.get('matrix', [])) >= 2]  # Just need header + 1 data row
                        if filtered_tables:
                            return filtered_tables
                except Exception:
                    pass
            use_camelot = (camelot_module is not None)
            if use_camelot:
                try:
                    tables = TableSchema.extract_tables_camelot(file_path, page_num_1based, camelot_module)
                    if tables:
                        return tables
                except Exception:
                    pass
        else:
            use_camelot = (tables_engine in ("camelot", "auto")) and (camelot_module is not None)
            if use_camelot:
                try:
                    tables = TableSchema.extract_tables_camelot(file_path, page_num_1based, camelot_module)
                    if tables:
                        return tables
                except Exception:
                    pass
            use_plumber = (tables_engine in ("pdfplumber", "auto")) and (plumber_pdf is not None)
            if use_plumber:
                try:
                    tables = TableSchema.extract_tables_pdfplumber(plumber_pdf, page_num_1based)
                    if tables:
                        return tables
                except Exception:
                    pass
        return out

    # ---------------- Constructors & rendering ----------------
    @staticmethod
    def from_matrix(
        matrix: List[List[str]],
        file_path: str = "",
        page_number: int = 0,
        bbox: Any = None,
        markdown: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'TableSchema':
        if not matrix or not matrix[0]:
            return TableSchema(file_path=file_path, page_number=page_number, bbox=bbox, markdown=markdown, metadata=metadata or {})
        header = matrix[0]
        rows = [
            TableRow(
                cells=[TableCell(value=cell, row=i+1, col=j) for j, cell in enumerate(row)],
                row_idx=i+1
            )
            for i, row in enumerate(matrix[1:])
        ]
        return TableSchema(
            file_path=file_path,
            page_number=page_number,
            header=header,
            rows=rows,
            bbox=bbox,
            markdown=markdown,
            metadata=metadata or {}
        )

    def to_markdown(self) -> str:
        if not self.header:
            return ""
        md = "| " + " | ".join(self.header) + " |\n"
        md += "|" + ("---|" * len(self.header)) + "\n"
        for row in self.rows:
            md += "| " + " | ".join(cell.value for cell in row.cells) + " |\n"
        return md

    def to_matrix(self) -> List[List[str]]:
        """Convert TableSchema back to matrix format for compatibility."""
        if not self.header:
            return []
        
        matrix = [self.header]
        for row in self.rows:
            row_data = [cell.value for cell in row.cells]
            matrix.append(row_data)
        
        return matrix

    # ---------------- Table manipulation utilities ----------------
    @staticmethod
    def extract_leading_number(value: Any) -> Optional[int]:
        """Extract leading number from a string value."""
        import re
        if not isinstance(value, str):
            return None
        
        match = re.match(r'^\s*(\d+)', value)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def header_looks_like_continuation(tbl: 'TableSchema') -> bool:
        """Check if table header looks like a continuation of previous table."""
        header = getattr(tbl, 'header', None) or []
        rows = getattr(tbl, 'rows', None) or []
        if not rows:
            return False
        
        first_num = None
        if header and header[0].strip():
            first_num = TableSchema.extract_leading_number(header[0])
        
        if first_num is None:
            first_row_value = ''
            first_row = rows[0]
            if getattr(first_row, 'cells', None):
                first_row_value = first_row.cells[0].value or ''
            first_num = TableSchema.extract_leading_number(first_row_value)
        
        if first_num is None:
            return False
        
        next_num = None
        for row in rows:
            cells = getattr(row, 'cells', None) or []
            if not cells:
                continue
            candidate = TableSchema.extract_leading_number(cells[0].value or '')
            if candidate is not None:
                next_num = candidate
                break
        
        if next_num is None:
            return first_num > 1
        
        if next_num == first_num or next_num == first_num + 1:
            return True
        
        if first_num > 1 and next_num > first_num:
            return True
        
        return False

    @staticmethod
    def make_row(values: List[str], row_idx: int) -> TableRow:
        """Create a TableRow from list of values."""
        cells = []
        for col_idx, val in enumerate(values, start=1):
            cells.append(TableCell(value=val, row=row_idx, col=col_idx, bbox=None, metadata={}))
        return TableRow(cells=cells, row_idx=row_idx)

    @staticmethod
    def reindex_rows(rows: List[TableRow]) -> None:
        """Reindex row and cell indices for consistency."""
        for r_idx, row in enumerate(rows, start=1):
            row.row_idx = r_idx
            for c_idx, cell in enumerate(row.cells, start=1):
                cell.row = r_idx
                cell.col = c_idx

    @staticmethod
    def match_row_to_columns(row: TableRow, target_len: int) -> None:
        """Adjust row to have exactly target_len columns."""
        cells = list(getattr(row, "cells", []) or [])
        if target_len <= 0:
            return
        
        if len(cells) > target_len:
            merged = " ".join(cell.value for cell in cells[target_len-1:]).strip()
            new_cells = cells[:target_len-1] + [TableCell(value=merged, row=row.row_idx, col=target_len, bbox=None, metadata={})]
        elif len(cells) < target_len:
            new_cells = cells + [TableCell(value='', row=row.row_idx, col=len(cells)+i+1, bbox=None, metadata={}) for i in range(target_len - len(cells))]
        else:
            new_cells = cells
        
        for idx, cell in enumerate(new_cells, start=1):
            cell.row = row.row_idx
            cell.col = idx
        row.cells = new_cells

    @staticmethod
    def rebuild_markdown(header: List[str], rows: List[TableRow]) -> str:
        """Rebuild markdown table from header and rows."""
        md_lines: List[str] = []
        if header:
            md_lines.append('| ' + ' | '.join(header) + ' |')
            md_lines.append('|' + ('---|' * len(header)))
        for row in rows:
            md_lines.append('| ' + ' | '.join(cell.value for cell in row.cells) + ' |')
        return '\n'.join(md_lines) + ('\n' if md_lines else '')
