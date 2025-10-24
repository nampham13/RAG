from huggingface_hub import snapshot_download
 
repo_id = "BAAI/bge-reranker-v2-m3"  # hoặc mô hình bạn cần
local_dir = "/workspaces/RAG/test/model/jina"
snapshot_download(repo_id=repo_id, local_dir=local_dir)