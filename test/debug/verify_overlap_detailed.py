#!/usr/bin/env python3
"""Debug the overlap calculation and see what's happening"""
import pickle
from pathlib import Path

# Load metadata
metadata_files = list(Path('data/vectors').glob('*metadata_map*.pkl'))
if not metadata_files:
    print("No metadata files found")
    exit(1)

metadata_file = metadata_files[-1]

with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

# Look at chunks 12, 13, 14
items = list(metadata_map.items())
for i in [11, 12, 13]:
    chunk_id, meta = items[i]
    text = meta.get('text', '')
    
    print(f"\n=== CHUNK {i+1} TEXT ===")
    print(f"First 200 chars: {repr(text[:200])}")
    print(f"Last 200 chars: {repr(text[-200:])}")
    
    # Check if text has overlap from previous
    if i > 0:
        prev_text = items[i-1][1].get('text', '')
        # Check if end of prev is in start of curr
        prev_last_100 = prev_text[-100:]
        if prev_last_100 in text:
            print(f"✓ OVERLAP FOUND: Previous chunk's last 100 chars appear in current chunk")
            overlap_start = text.find(prev_last_100)
            print(f"  Overlap starts at position {overlap_start}")
        else:
            # Try shorter sequences
            for check_len in [50, 30, 20, 10]:
                check_text = prev_text[-check_len:]
                if check_text in text and len(check_text.strip()) > 2:
                    print(f"✓ PARTIAL OVERLAP: Previous chunk's last {check_len} chars appear in current")
                    print(f"  Overlap text: {repr(check_text)}")
                    break
            else:
                print(f"✗ NO OVERLAP DETECTED")
