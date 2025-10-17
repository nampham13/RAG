import pickle
from pathlib import Path

# Load metadata to check if overlap is actually in chunks
metadata_file = Path('data/vectors/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_metadata_map_20251016_164029.pkl')
with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

print("Checking overlap between chunks 12 and 13 (pages 8-9):")
print("=" * 100)

chunk_12 = metadata_map[12]
chunk_13 = metadata_map[13]

text_12 = chunk_12.get('text', '')
text_13 = chunk_13.get('text', '')

print(f"\nChunk 12 (Page {chunk_12.get('page_number')})")
print(f"Length: {len(text_12)} chars")
print(f"Last 150 chars:")
print(repr(text_12[-150:]))

print(f"\n{'='*100}")
print(f"\nChunk 13 (Page {chunk_13.get('page_number')})")
print(f"Length: {len(text_13)} chars")
print(f"First 150 chars:")
print(repr(text_13[:150]))

print(f"\n{'='*100}")
print("\nChecking if chunk 13 starts with overlap from chunk 12:")

# Get last 100 chars of chunk 12
last_100_chunk12 = text_12[-100:]
# Get first 100 chars of chunk 13
first_100_chunk13 = text_13[:100]

# Check if there's any common text
common_found = False
for i in range(5, 50):  # Check different overlap sizes
    overlap_candidate = last_100_chunk12[-i:]
    if first_100_chunk13.startswith(overlap_candidate.strip()):
        print(f"Found {i}-char overlap: '{overlap_candidate}'")
        common_found = True
        break

if not common_found:
    print("NO OVERLAP FOUND between chunk 12 and 13!")
    print(f"Chunk 12 ends with: '{last_100_chunk12}'")
    print(f"Chunk 13 starts with: '{first_100_chunk13}'")