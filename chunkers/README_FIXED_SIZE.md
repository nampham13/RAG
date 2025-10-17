# Fixed Size Chunker Implementation

## Enhancements Made

The FixedSizeChunker has been enhanced with the following features:

1. **Accurate Token Counting with tiktoken**
   - Uses OpenAI's tiktoken library for precise token counts
   - Supports different encoding models (cl100k_base for GPT-4, p50k_base for GPT-3)
   - Provides accurate token accounting for optimal chunk sizing

2. **Automatic Fallback Mechanism**
   - Falls back to character-based estimation if tiktoken is unavailable
   - Maintains consistent behavior regardless of available libraries

3. **Sentence Boundary Respect**
   - Option to respect sentence boundaries when creating chunks
   - Prevents chunks from breaking in the middle of sentences
   - Improves readability and usability of chunks for downstream tasks

4. **Enhanced Boundary Detection**
   - Improved algorithm for finding optimal chunk boundaries
   - Prioritizes paragraph breaks, sentence endings, and other natural boundaries
   - Works at both character and token level

5. **Improved Chunk Scoring**
   - More sophisticated scoring system for evaluating chunk quality
   - Assesses completeness based on sentence/paragraph structure
   - Provides metadata about paragraph and sentence counts

## Usage

```python
from chunkers.fixed_size_chunker import FixedSizeChunker

# Create a chunker with specified parameters
chunker = FixedSizeChunker(
    max_tokens=512,               # Maximum tokens per chunk
    overlap_tokens=50,            # Overlap between chunks
    encoding_name="cl100k_base",  # Tiktoken encoding model
    respect_sentence_boundary=True # Don't break mid-sentence
)

# Process a document
chunks = chunker.chunk(document)
```

## Implementation Details

The FixedSizeChunker now uses a two-stage approach:

1. **Token-Based Chunking (with tiktoken)**
   - Pre-encodes text to get precise token boundaries
   - Maps token positions to character positions
   - Creates chunks based on token counts with proper boundaries

2. **Character-Based Chunking (fallback)**
   - Uses a character-to-token estimation (~4 chars per token)
   - Applies similar boundary detection logic

This approach ensures consistent chunking across different environments and optimal chunk sizes for LLM processing.
