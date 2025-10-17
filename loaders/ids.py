import hashlib
from loaders.model.block import Block
from loaders.model.table import TableSchema

def make_stable_id(*parts) -> str:
    raw = '|'.join(str(p) for p in parts)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]

def content_sha256(text: str) -> str:
    return hashlib.sha256((text or '').encode('utf-8')).hexdigest()

def block_stable_id(block: Block) -> str:
    return make_stable_id(block.text, block.bbox)

def table_stable_id(table: TableSchema) -> str:
    return make_stable_id(table.file_path, table.page_number, table.header, table.rows)
