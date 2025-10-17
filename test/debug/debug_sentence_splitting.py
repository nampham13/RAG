#!/usr/bin/env python3
"""Debug sentence splitting for the problematic paragraph"""
from loaders.pdf_loader import PDFLoader
from chunkers.semantic_chunker import SemanticChunker
from pathlib import Path

# Load PDF
pdf_path = list(Path('data/pdf').glob('*.pdf'))[-1] if Path('data/pdf').exists() else None

if not pdf_path:
    print("No PDF found")
    exit(1)

print(f"Loading: {pdf_path.name}")

loader = PDFLoader.create_default()
doc = loader.load(str(pdf_path))

# Create chunker
chunker = SemanticChunker(max_tokens=250, overlap_tokens=35)

# Get the problematic text (pages 8-9, around "Self-Reasoning")
print("\n=== Looking for 'Self-Reasoning' section ===")

for page_idx, page in enumerate(doc.pages, 1):
    for block in page.blocks:
        if hasattr(block, 'text') and 'Self-Reasoning' in (block.text or ''):
            print(f"Found in page {page_idx}")
            print(f"Block text preview: {block.text[:200]}...\n")
            
            # Split into sentences
            sentences = chunker._split_into_sentences(block.text)
            print(f"Sentences found: {len(sentences)}")
            for i, sent in enumerate(sentences):
                if 'Instead' in sent or 'trusting' in sent:
                    print(f"  [{i}] {sent[:100]}...")
