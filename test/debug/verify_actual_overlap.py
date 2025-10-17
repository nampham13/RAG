#!/usr/bin/env python3
"""Check actual chunk 13 end text to understand the overlap issue"""
import pickle
from pathlib import Path

# Load metadata
metadata_files = list(Path('data/vectors').glob('*metadata_map*.pkl'))
if not metadata_files:
    print("No metadata files found")
    exit(1)

metadata_file = metadata_files[-1]
print(f"Loading: {metadata_file.name}\n")

with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

# Get chunks 13 and 14
chunks_list = list(metadata_map.items())
if len(chunks_list) > 13:
    chunk_13_id, chunk_13_meta = chunks_list[12]  # Index 12 = chunk 13
    chunk_14_id, chunk_14_meta = chunks_list[13]  # Index 13 = chunk 14
    
    chunk_13_text = chunk_13_meta.get('text', '')
    chunk_14_text = chunk_14_meta.get('text', '')
    
    print("=== CHUNK 13 ===")
    print(f"Last 200 chars: {repr(chunk_13_text[-200:])}")
    print()
    print("=== CHUNK 14 ===")
    print(f"First 200 chars: {repr(chunk_14_text[:200])}")
    print()
    
    # Look for "Instead of" in chunk 13
    if "Instead of" in chunk_13_text:
        idx = chunk_13_text.rfind("Instead of")
        print(f"Found 'Instead of' at position {idx}")
        print(f"Context: {repr(chunk_13_text[max(0, idx-50):idx+100])}")
    
    # Check if chunk 14 starts with corrupted version
    if chunk_14_text.startswith("d of"):
        print("\n✗ CORRUPTED: Chunk 14 starts with 'd of'")
        # Find where this comes from in chunk 13
        corrupt_start = chunk_13_text.rfind("d of trusting")
        if corrupt_start >= 0:
            print(f"Found partial text at position {corrupt_start} in chunk 13")
            print(f"Context: {repr(chunk_13_text[corrupt_start-20:corrupt_start+50])}")
    else:
        print("\n✓ CLEAN: Chunk 14 starts properly")
