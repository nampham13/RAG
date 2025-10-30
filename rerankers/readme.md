# Rerankers — Cấu trúc thư mục

Mô-đun rerankers chứa các implement để tái xếp hạng kết quả tìm kiếm (reranking). Dưới đây là cấu trúc thư mục chuẩn và mô tả ngắn các thành phần.

Thư mục:
```
rerankers/
├── init.py
├── i_reranker.py # Giao diện (interface) chung cho các reranker
├── readme.md # Tệp này
├── reranker_factory.py # Factory để tạo instance reranker theo type
├── reranker_type.py # Định nghĩa kiểu/enum reranker
├── model/
│ ├── init.py
│ └── reranker_profile.py # Định nghĩa profile/cấu hình model reranker
├── providers/
│ ├── init.py
│ ├── base_reranker.py # Lớp cơ sở provider (Single Responsibility)
│ ├── bge_reranker.py # Provider sử dụng BGE embeddings
│ └── jina_reranker.py # Provider tích hợp Jina hoặc dịch vụ tương tự
└── pycache/ # Bytecode cache (tự sinh)
```

notice: after installing new lib in codespace delete all the cuda components to save space for models