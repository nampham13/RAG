"""
Test Fixed Size Chunker
======================
Verify fixed size chunking implementation with tiktoken.
"""

import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .fixed_size_chunker import FixedSizeChunker
from loaders.model.block import Block
import sys

def create_test_block(text, block_id, page_num=1):
    """Create a simple test block"""
    block = Block(text=text)
    block.stable_id = block_id
    block.metadata = {'page_number': page_num}
    return block

def test_fixed_size_chunker():
    """Simple test for the fixed size chunker"""
    
    # Sample text for testing
    text1 = """
This is a test document for the fixed size chunker. It will be divided into chunks based on token count. 
We'll use tiktoken for accurate token counting if available.

The fixed size chunker is a simple but effective method for dividing text when no obvious structural 
or semantic boundaries exist. It can be used as a fallback when other chunking methods are not applicable.
    """
    
    text2 = """
Each chunk will have a maximum token count, with an optional overlap between chunks.
This ensures that context is not lost at chunk boundaries.

The chunker can respect sentence boundaries, ensuring that sentences are not cut off in the middle.
This improves the readability and usability of the chunks for downstream tasks like embedding or retrieval.
    """
    
    # Create test blocks
    blocks = [
        create_test_block(text1, "block_1", 1),
        create_test_block(text2, "block_2", 1)
    ]
    
    # Create chunker with different configurations
    chunker1 = FixedSizeChunker(max_tokens=100, overlap_tokens=20)
    chunker2 = FixedSizeChunker(max_tokens=50, overlap_tokens=10, respect_sentence_boundary=False)
    
    print("===== Testing Fixed Size Chunker =====")
    
    # Test chunking with first config
    print("\n1. Chunking with max_tokens=100, overlap=20, respect_sentence_boundary=True")
    chunks1 = chunker1.chunk_blocks(blocks, "test_doc")
    print(f"Created {len(chunks1)} chunks:")
    for i, chunk in enumerate(chunks1):
        print(f"\nChunk {i+1} ({chunk.token_count} tokens):")
        print(f"Text: {chunk.text[:100]}...")
        if chunk.score:
            print(f"Score: coherence={chunk.score.coherence_score:.2f}, "
                 f"completeness={chunk.score.completeness_score:.2f}, "
                 f"token_ratio={chunk.score.token_ratio:.2f}")
        else:
            print("Score: Not available")
    
    # Test chunking with second config
    print("\n\n2. Chunking with max_tokens=50, overlap=10, respect_sentence_boundary=False")
    chunks2 = chunker2.chunk_blocks(blocks, "test_doc")
    print(f"Created {len(chunks2)} chunks:")
    for i, chunk in enumerate(chunks2):
        print(f"\nChunk {i+1} ({chunk.token_count} tokens):")
        print(f"Text: {chunk.text[:100]}...")
        if chunk.score:
            print(f"Score: coherence={chunk.score.coherence_score:.2f}, "
                 f"completeness={chunk.score.completeness_score:.2f}, "
                 f"token_ratio={chunk.score.token_ratio:.2f}")
        else:
            print("Score: Not available")

if __name__ == "__main__":
    test_fixed_size_chunker()