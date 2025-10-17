# Loaders Module - PDF Document Processing# RAG Loaders



Module xử lý và trích xuất dữ liệu từ file PDF theo chuẩn OOP, hỗ trợ trích xuất text và bảng với các tính năng lọc và chuẩn hóa dữ liệu.Module chịu trách nhiệm trích xuất dữ liệu thô từ PDF (loader) và chuẩn hóa/biến đổi (normalizer) qua các class chuyên biệt. Các model trung gian (PDFDocument, PDFPage, Block, TableSchema, ...) đều hỗ trợ mở rộng normalization ở tầng class.

## 📁 Cấu trúc thư mục## Cấu trúc

``````text

loaders/rag/loaders/

├── __init__.py              # Package exports├── pdf_loader.py         # Loader: chỉ load và parse PDF, KHÔNG normalize

├── pdf_loader.py            # PDFLoader class chính├── config.py            # YAML config loader

├── ids.py                   # ID generation utilities├── ids.py               # ID generation utilities

├── model/                   # Data models├── model/               # Data models

│   ├── __init__.py│   ├── base.py         # Base classes

│   ├── base.py             # Base classes│   ├── document.py     # PDFDocument model

│   ├── document.py         # PDFDocument│   ├── page.py         # PDFPage model

│   ├── page.py             # PDFPage│   ├── block.py        # Text/Table blocks

│   ├── block.py            # TextBlock, TableBlock│   └── table.py        # Table schema

│   ├── table.py            # TableSchema, TableRow, TableCell├── normalizers/         # Data normalization

│   └── text.py             # Text models│   ├── text.py         # Text processing

└── normalizers/             # Data normalization utilities│   ├── tables.py       # Table processing

    ├── block_utils.py      # Block filtering & processing│   └── layout.py       # Layout analysis

    ├── table_utils.py      # Table extraction & cleaning├── __init__.py         # Package init

    ├── text_utils.py       # Text normalization└── README.md           # File này

    └── spacy_utils.py      # NLP utilities (sentence splitting)```

```

## Chức năng chính

## 🎯 Chức năng chính

### PDFLoader (pdf_loader.py)

### PDFLoader - Class trích xuất PDF

- **Chỉ load và parse PDF thành dữ liệu thô**

**Thiết kế OOP thuần túy:**- KHÔNG thực hiện normalize, KHÔNG xử lý tables, KHÔNG chunking

- ✅ Dependency injection - Config thông qua constructor- Sử dụng PyMuPDF để extract blocks thô cho từng trang

- ✅ Factory methods - Preset configurations- Trả về các model trung gian (PDFDocument, PDFPage, Block, ...)

- ✅ Runtime configuration - Dynamic config updates- Phù hợp cho pipeline custom hoặc các bước xử lý tiếp theo

- ✅ No external dependencies - Không phụ thuộc YAML- Nếu muốn chuẩn hóa, hãy sử dụng các method `.normalize()` ở từng class model hoặc qua normalizer riêng biệt.

- ✅ Type-safe - Full type hints

- ✅ Testable - Dễ dàng mock và test### Data Models (model/)



**Tính năng:**- **PDFDocument**: Container cho toàn bộ document, có thể mở rộng chuẩn hóa qua `.normalize()`

- Trích xuất text từ PDF (PyMuPDF)- **PDFPage**: Đại diện cho một trang PDF, hỗ trợ chuẩn hóa layout/text qua `.normalize()`

- Trích xuất bảng với nhiều engines (pdfplumber, camelot, pymupdf)- **Block**: Text hoặc table blocks với position info, có thể chuẩn hóa text qua `.normalize()`

- Lọc block lặp lại (header/footer)- **TableSchema**: Cấu trúc bảng với rows/columns, chuẩn hóa header/rows qua `.normalize()`

- Lọc block ngắn/noise

- Gán caption cho bảng tự động### Normalizers (normalizers/)

- Chuẩn hóa text (unicode, whitespace)

- Deterministic IDs cho mọi element- **TextNormalizer**: Class/hàm chuẩn hóa text content, có thể dùng độc lập hoặc gọi từ model

- **TableNormalizer**: Chuẩn hóa bảng, header, rows

## 🚀 Cách sử dụng- **LayoutNormalizer**: Chuẩn hóa vị trí, bbox, reading order



### 1. Sử dụng cơ bản với cấu hình mặc định## Loại bỏ dữ liệu trùng lặp, nhiễu, header/footer (Block/Table Filtering)



```python### Các bước đã thực hiện để làm sạch dữ liệu

from loaders import PDFLoader

#### Đối với Block (text)

# Factory method - cấu hình mặc định

loader = PDFLoader.create_default()- **Chuẩn hóa text ở Block.normalize:**

- Sử dụng `clean-text` và `ftfy` để chuẩn hóa unicode, loại bỏ ký tự vô hình, emoji, ký tự đặc biệt.

# Hoặc khởi tạo trực tiếp (all params có default)- Loại bỏ các chuỗi dấu chấm lặp ("......") thường gặp ở TOC.

loader = PDFLoader()- Chuẩn hóa whitespace, loại bỏ nhiều khoảng trắng/thừa dòng.

- Giữ lại line-break hợp lý để phân biệt đoạn/câu.

# Load PDF- **Chuyển đổi block tuple thành Block object trước khi normalize:**

document = loader.load("path/to/file.pdf")- Đảm bảo mọi block đều được chuẩn hóa text trước khi lọc.

- **Lọc block lặp lại (header/footer):**

# Truy cập dữ liệu- Tính hash cho từng block text toàn document, đếm số lần xuất hiện.

for page in document.pages:- Nếu một block xuất hiện >= `repeated_block_threshold` (configurable, mặc định 3), block đó sẽ bị loại bỏ (trừ khi là nội dung thực sự dài).

    print(f"Page {page.page_number}:")- **Lọc block ngắn/noise:**

    - Loại bỏ block có độ dài nhỏ hơn `min_text_length` (configurable, mặc định 10).

    # Text blocks- Loại bỏ block chỉ chứa whitespace, số trang, hoặc bbox quá nhỏ.

    for block in page.text_blocks:- **Lọc block theo vị trí (header/footer):**

        print(f"  Text: {block.text[:50]}...")- Nếu block nằm ở top/bottom của trang và ngắn, sẽ bị loại bỏ.

    - **Lọc block trùng lặp cross-document:**

    # Tables- Các block header/table header/TOC lặp lại ở nhiều file sẽ vẫn còn, nhưng đã loại bỏ phần lớn noise trong từng document.

    for table in page.tables:- **Tất cả tham số lọc đều cấu hình qua YAML (`config/preprocessing.yaml`).**

        print(f"  Table: {len(table.rows)} rows")

```#### Đối với Table



### 2. Cấu hình tùy chỉnh- **Chuẩn hóa bảng ở TableSchema.normalize:**

- Chuẩn hóa text từng cell, header, row bằng `clean-text` và các rule tương tự block.

```python- Loại bỏ các dòng/cột trống hoàn toàn.

# Chỉ trích xuất text- Loại bỏ các dòng/cột chỉ chứa ký tự noise (dấu chấm, gạch ngang, ký tự đặc biệt).

text_loader = PDFLoader(- Loại bỏ các dòng/cột lặp lại hoàn toàn trong bảng.

    extract_text=True,- Chuẩn hóa lại header, merge header nếu bị split.

    extract_tables=False,- **Lọc bảng noise:**

    min_text_length=20,- Bảng chỉ có 1 dòng hoặc 1 cột, hoặc toàn bộ cell trùng lặp sẽ bị loại bỏ.

    enable_repeated_block_filter=True- Bảng không có giá trị thực (sau khi clean) sẽ bị loại bỏ.

)- **Tất cả tham số lọc bảng đều cấu hình qua YAML (`config/preprocessing.yaml`).**



# Chỉ trích xuất bảng### Kết quả thực nghiệm

table_loader = PDFLoader(

    extract_text=False,- Đã loại bỏ ~50% block noise/trùng lặp trên các file PDF mẫu.

    extract_tables=True,- Đã loại bỏ phần lớn bảng noise, bảng trùng header/footer, bảng chỉ có 1 dòng/cột hoặc toàn ký tự đặc biệt.

    tables_engine="camelot",  # or "pdfplumber", "pymupdf", "auto"- Các block và bảng còn lại chủ yếu là nội dung thực, table, hoặc header/table header cross-document.

    table_settings={

        "flavor": "lattice",## Output Schema

        "line_scale": 40

    }Mỗi chunk/model có thể có:

)

- `stable_id`: Deterministic ID (hash-based)

# Tùy chỉnh đầy đủ- `metadata["citation"]`: Human-readable citation (e.g., "doc-title, p.12")

custom_loader = PDFLoader(- `bbox_norm`: Normalized bounding box

    extract_text=True,- `source`: Full source attribution (doc_id, page_number, etc.)

    extract_tables=True,- `content_sha256`: Content hash for stability

    tables_engine="auto",

    min_repeated_text_threshold=5,  # Block xuất hiện >= 5 lần sẽ bị lọc## Config

    min_text_length=10,              # Lọc text < 10 chars

    enable_repeated_block_filter=True,- `rag/config/preprocessing.yaml`: Cấu hình preprocessing cho loader

    enable_short_block_filter=True,- `rag/config/chunking.yaml`: Cấu hình chunking/normalization (nếu cần)

    enable_bbox_filter=True,

    table_settings={## Cách sử dụng

        "flavor": "lattice",

        "line_scale": 40```python

    }from rag.loaders.pdf_loader import PDFLoader

)

```# Load PDF thô với cấu hình mặc định

loader = PDFLoader.create_default()

### 3. Factory methods (preset configs)doc = loader.load_pdf("path/to/document.pdf")



```python# Hoặc với cấu hình tùy chỉnh

# Text only - không trích xuất bảngloader = PDFLoader(

loader = PDFLoader.create_text_only()    extract_text=True,

    extract_tables=True,

# Tables only - không trích xuất text    min_repeated_text_threshold=5

loader = PDFLoader.create_tables_only())

doc = loader.load_pdf("path/to/document.pdf")

# Default - trích xuất cả text và bảng

loader = PDFLoader.create_default()# Chuẩn hóa toàn bộ document (nếu muốn)

```doc_norm = doc.normalize()  # Yêu cầu các class model đã implement .normalize()



### 4. Quản lý cấu hình runtime# Hoặc chuẩn hóa từng page/block

for page in doc.pages:

```python    page_norm = page.normalize()

loader = PDFLoader.create_default()```



# Xem cấu hình hiện tại## Tích hợp

config = loader.get_config()

print(config)- Input: Raw PDF files từ `data/pdf/`

- Output: `PDFDocument` objects cho `DocumentService`

# Cập nhật cấu hình- Loader chỉ trả về dữ liệu thô, không chunking, không normalize

loader.update_config(- Nếu muốn chuẩn hóa, hãy gọi `.normalize()` ở tầng model hoặc dùng normalizer riêng

    extract_tables=False,

    min_text_length=15,

    tables_engine="camelot" Duplicate Statistics:

)   • Total blocks: 230

   • Unique texts: 224

# Enable/disable tất cả filters   • Duplicate blocks: 6

loader.enable_all_filters()   • Duplicate rate: 2.6%

loader.disable_all_filters()

```✅ LOW DUPLICATE RATE (2.6%)

   Recommendation: Deduplication optional

## 📊 Data Models

💡 DEDUPLICATION STRATEGY:

### PDFDocument   1. Group blocks by content_sha256

```python   2. Keep first occurrence per hash

class PDFDocument:   3. Remove 6 duplicate blocks

    doc_id: str                    # Unique document ID   4. Result: 224 unique blocks

    file_path: str                 # Source file path
    metadata: Dict[str, Any]       # Document metadata
    pages: List[PDFPage]           # List of pages
```

### PDFPage
```python
class PDFPage:
    page_id: str                   # Unique page ID
    page_number: int               # Page number (1-indexed)
    text_blocks: List[TextBlock]   # Text blocks
    tables: List[TableSchema]      # Tables
    width: float                   # Page width
    height: float                  # Page height
```

### TextBlock
```python
class TextBlock:
    block_id: str                  # Unique block ID
    text: str                      # Block text content
    bbox: Tuple[float, ...]        # Bounding box (x0, y0, x1, y1)
    block_type: str                # Block type
    metadata: Dict[str, Any]       # Additional metadata
```

### TableSchema
```python
class TableSchema:
    table_id: str                  # Unique table ID
    caption: Optional[str]         # Table caption (auto-assigned)
    header: Optional[TableRow]     # Table header
    rows: List[TableRow]           # Table rows
    bbox: Optional[Tuple[float]]   # Bounding box
```

## 🧹 Lọc và Chuẩn hóa dữ liệu

### Block Filtering

**1. Lọc block lặp lại (Header/Footer):**
- Tính hash cho mỗi block text
- Đếm số lần xuất hiện trong document
- Lọc block xuất hiện >= `min_repeated_text_threshold` lần
- Giữ lại block dài (> 200 chars) dù lặp lại

**2. Lọc block ngắn/noise:**
- Lọc block < `min_text_length` chars
- Lọc block chỉ chứa whitespace
- Lọc block chỉ chứa số trang

**3. Lọc theo bbox:**
- Lọc block có bbox quá nhỏ (area < threshold)
- Lọc block ở vị trí header/footer area

**Kết quả:**
- ✅ Loại bỏ ~50% block noise/duplicate
- ✅ Giữ lại nội dung có giá trị
- ✅ Duplicate rate < 3% sau filtering

### Table Processing

**1. Table Extraction:**
- Hỗ trợ 3 engines: pdfplumber, camelot, pymupdf
- Auto engine selection dựa trên PDF structure
- Configurable extraction parameters

**2. Table Cleaning:**
- Chuẩn hóa text trong cells
- Loại bỏ rows/columns trống
- Loại bỏ duplicate rows
- Merge split headers

**3. Caption Assignment:**
- Tự động tìm và gán caption cho bảng
- Tìm kiếm text blocks phía trên/dưới bảng
- Sử dụng pattern matching (Table X, Bảng X)
- Scoring candidates dựa trên distance và keywords

### Text Normalization

**Utilities trong `normalizers/text_utils.py`:**
- Unicode normalization (ftfy)
- Clean special characters (clean-text)
- Remove zero-width characters
- Normalize whitespace
- De-hyphenation (merge từ bị ngắt dòng)
- Remove repeated dots (......) từ TOC

## ⚙️ Configuration Parameters

### PDFLoader Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_text` | bool | True | Trích xuất text blocks |
| `extract_tables` | bool | True | Trích xuất tables |
| `tables_engine` | str | "auto" | Table engine: "auto", "pdfplumber", "camelot", "pymupdf" |
| `min_repeated_text_threshold` | int | 3 | Block xuất hiện >= N lần sẽ bị lọc |
| `min_text_length` | int | 10 | Lọc block < N chars |
| `enable_repeated_block_filter` | bool | True | Enable lọc block lặp lại |
| `enable_short_block_filter` | bool | True | Enable lọc block ngắn |
| `enable_bbox_filter` | bool | True | Enable lọc theo bbox |
| `table_settings` | dict | {} | Custom settings cho table engine |

### Table Engine Settings

**pdfplumber:**
```python
table_settings = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3
}
```

**camelot:**
```python
table_settings = {
    "flavor": "lattice",  # or "stream"
    "line_scale": 40,
    "copy_text": ["v", "h"]
}
```

## 🧪 Testing

```powershell
# Activate virtual environment
& C:/Users/ENGUYEHWC/Downloads/RAG/RAG/.venv/Scripts/Activate.ps1

# Run tests
python -m pytest tests/test_loader.py -v

# Run with coverage
python -m pytest tests/ --cov=loaders --cov-report=html
```

## 📦 Dependencies

```
pymupdf>=1.23.0      # PDF text extraction
pdfplumber>=0.10.0   # Table extraction (engine 1)
camelot-py>=0.11.0   # Table extraction (engine 2)
opencv-python        # Required by camelot
ghostscript          # Required by camelot
ftfy>=6.1.0         # Unicode normalization
clean-text>=0.6.0   # Text cleaning
spacy>=3.7.0        # NLP utilities
```

## 🔧 Troubleshooting

### Camelot không hoạt động
```powershell
# Install ghostscript
# Download from: https://www.ghostscript.com/download/gsdnld.html
# Add to PATH: C:\Program Files\gs\gs9.xx\bin
```

### Table extraction kém
```python
# Thử engine khác
loader = PDFLoader(tables_engine="pdfplumber")  # hoặc "camelot"

# Hoặc tune settings
loader = PDFLoader(
    tables_engine="camelot",
    table_settings={
        "flavor": "stream",  # Thử stream thay vì lattice
        "edge_tol": 50
    }
)
```

### Quá nhiều block bị lọc
```python
# Giảm threshold
loader = PDFLoader(
    min_repeated_text_threshold=10,  # Tăng từ 3 lên 10
    min_text_length=5,                # Giảm từ 10 xuống 5
    enable_repeated_block_filter=False  # Tắt filter
)
```

## 📝 Migration từ code cũ

**Code cũ (YAML-based):**
```python
# Cũ - phụ thuộc YAML config
loader = PDFLoader()  # Load từ config/preprocessing.yaml
```

**Code mới (OOP):**
```python
# Mới - dependency injection
loader = PDFLoader.create_default()  # Equivalent behavior

# Hoặc explicit config
loader = PDFLoader(
    extract_text=True,
    extract_tables=True,
    min_repeated_text_threshold=3
)
```

## 🎯 Development Focus

### ✅ Hoàn thành
- [x] PDFLoader refactored to pure OOP
- [x] Removed YAML dependency
- [x] Factory methods
- [x] Block filtering (repeated, short, bbox)
- [x] Table extraction (3 engines)
- [x] Caption assignment
- [x] Text normalization
- [x] Deterministic IDs

### 🔄 Đang phát triển
- [ ] Complete test coverage (>90%)
- [ ] Performance benchmarking
- [ ] Memory optimization

### 📋 TODO
- [ ] Advanced layout analysis
- [ ] Multi-column detection
- [ ] Language detection
- [ ] OCR support for scanned PDFs

## 📚 Tài liệu liên quan

- **API Documentation**: See docstrings in `pdf_loader.py`
- **Model Schemas**: See `model/` directory
- **Utility Functions**: See `normalizers/` directory
- **Copilot Instructions**: See `.github/copilot-instructions.md`

## 🤝 Contributing

Khi phát triển module này:
1. **Focus on loaders only** - Không touch chunkers/pipeline
2. **OOP first** - All functionality trong classes
3. **Type hints** - Full type annotations
4. **Tests** - Coverage >90%
5. **Documentation** - Update README khi có changes

## 📄 License

Theo license của project RAG.
