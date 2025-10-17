#!/usr/bin/env python3
"""Check if overlap is properly applied between chunks 13 and 14"""
import re

chunks_file = r'data/chunks/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_chunks_20251016_174255.txt'

with open(chunks_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find CHUNK 13 and 14
chunk_13_match = re.search(r'CHUNK 13:.*?(?=CHUNK 14:|$)', content, re.DOTALL)
chunk_14_match = re.search(r'CHUNK 14:.*?(?=CHUNK 15:|$)', content, re.DOTALL)

if chunk_13_match and chunk_14_match:
    chunk_13_text = chunk_13_match.group(0)
    chunk_14_text = chunk_14_match.group(0)
    
    # Extract content after the header line (---------)
    c13_content_start = chunk_13_text.find('---') + 50  # Skip ------- line
    c13_content = chunk_13_text[c13_content_start:chunk_13_text.rfind('\n\n')]
    
    c14_content_start = chunk_14_text.find('---') + 50  # Skip ------- line
    c14_content = chunk_14_text[c14_content_start:chunk_14_text.rfind('\n\n')]
    
    # Get last 150 chars of chunk 13
    print("=== CHUNK 13 LAST 150 CHARS ===")
    print(repr(c13_content[-150:]))
    print()
    print("=== CHUNK 14 FIRST 150 CHARS ===")
    print(repr(c14_content[:150]))
    print()
    
    # Check for overlap
    print("=== OVERLAP CHECK ===")
    # Get last 100 words of chunk 13
    c13_words = c13_content.split()[-20:]  # Last 20 words
    c14_words = c14_content.split()[:20]  # First 20 words
    
    c13_tail = ' '.join(c13_words)
    c14_head = ' '.join(c14_words)
    
    print("CHUNK 13 LAST 20 WORDS:")
    print(c13_tail)
    print()
    print("CHUNK 14 FIRST 20 WORDS:")
    print(c14_head)
    print()
    
    # Check if they share text (overlap)
    if c13_tail in c14_content or any(word in c14_head for word in c13_words):
        print("✓ OVERLAP DETECTED")
    else:
        print("✗ NO OVERLAP - chunks are separate")
