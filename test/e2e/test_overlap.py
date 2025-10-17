from chunkers.hybrid_chunker import HybridChunker, ChunkerMode
from loaders.pdf_loader import PDFLoader

# Load document
loader = PDFLoader.create_default()
doc = loader.load('data/pdf/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy.pdf')

# Chunk with overlap
chunker = HybridChunker(max_tokens=250, overlap_tokens=35, mode=ChunkerMode.SEMANTIC_FIRST)
chunk_set = chunker.chunk(doc)

print(f'Total chunks: {len(chunk_set.chunks)}')
print('Checking overlap between consecutive chunks:')

for i in range(min(3, len(chunk_set.chunks)-1)):  # Just check first 3 pairs
    chunk1 = chunk_set.chunks[i]
    chunk2 = chunk_set.chunks[i+1]
    
    print(f'\n--- Checking chunks {i} -> {i+1} ---')
    print(f'Chunk {i} full text ends with: "...{chunk1.text[-100:]}"')
    print(f'Chunk {i+1} full text starts with: "{chunk2.text[:100]}..."')
    
    # Simulate what _apply_overlap does
    prev_tokens = chunk1.text.split()  # Simple tokenization
    overlap_token_count = min(35, len(prev_tokens) // 2)
    overlap_tokens = prev_tokens[-overlap_token_count:]
    expected_overlap_text = " ".join(overlap_tokens)
    
    print(f'Expected overlap text ({len(expected_overlap_text)} chars): "{expected_overlap_text}"')
    
    # Check if chunk2 starts with this overlap text
    if chunk2.text.startswith(expected_overlap_text):
        print(f'✓ OVERLAP CONFIRMED: Chunk {i+1} starts with overlap from chunk {i}')
    else:
        print(f'✗ NO OVERLAP: Chunk {i+1} does not start with expected overlap')
        print(f'  Chunk2 actually starts with: "{chunk2.text[:len(expected_overlap_text)]}"')