# Thư mục data

Chứa dữ liệu đầu vào và đầu ra cho hệ thống RAG.

## Cấu trúc

data/
├── pdf/                    # File PDF nguồn để load và chunking
│   ├── Process_Risk Management.pdf
│   ├── Process_Service Configuration Management.pdf
│   ├── Process_Service Delivery.pdf
│   └── Process_Service Management.pdf
└── embedded/               # Index và metadata đã được embedding
    ├── Process_Risk Management.index
    ├── Process_Risk Management.meta.json
    ├── Process_Risk Management.rows.map.json
    ├── Process_Service Configuration Management.index
    ├── Process_Service Configuration Management.meta.json
    ├── Process_Service Configuration Management.rows.map.json
    ├── Process_Service Delivery.index
    ├── Process_Service Delivery.meta.json
    ├── Process_Service Delivery.rows.map.json
    ├── Process_Service Management.index
    ├── Process_Service Management.meta.json
    └── Process_Service Management.rows.map.json

## pdf/

- Chứa các file PDF gốc dùng để xây dựng index
- Thêm file PDF mới vào đây để tạo index từ tài liệu mới
- Các file PDF sẽ được load, chunk và embed tự động

## embedded/

- Chứa kết quả embedding đã sinh (FAISS index + metadata)
- File `.index`: FAISS vector index
- File `.meta.json`: Metadata của chunks
- File `.rows.map.json`: Mapping giữa chunk ID và vector index
- Các file này được tạo tự động bởi `EmbeddingPipeline`

## Lưu ý

- Thư mục `embedded/` được tạo tự động khi chạy pipeline
- Không nên chỉnh sửa thủ công các file trong `embedded/`
- Nếu muốn rebuild index, xóa thư mục `embedded/` và chạy lại pipeline
