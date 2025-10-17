#!/usr/bin/env python3
"""Check actual chunk metadata from metadata_map.pkl"""
import pickle
from pathlib import Path

# Load metadata
metadata_files = list(Path('data/vectors').glob('*metadata_map*.pkl'))
if not metadata_files:
    print("No metadata files found")
    exit(1)

metadata_file = metadata_files[-1]
print(f"Loading: {metadata_file.name}")

with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

# Look at chunks 12, 13, 14
print(f"Total chunks in metadata: {len(metadata_map)}\n")

for i, (chunk_id, meta) in enumerate(list(metadata_map.items())):
    if i < 11 or i > 13:
        continue
    
    text = meta.get('text', '')
    print(f"=== CHUNK {i+1} (Index {i}) ===")
    print(f"Chunk ID: {chunk_id}")
    print(f"Text start (100 chars): {repr(text[:100])}")
    print(f"Text end (100 chars): {repr(text[-100:])}")
    print(f"Pages: {meta.get('pages')}")
    print(f"Token count: {meta.get('token_count')}")
    print(f"Char count: {len(text)}")
    print()
