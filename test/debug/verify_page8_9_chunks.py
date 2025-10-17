import pickle

# Load metadata to check chunks on page 8-9
metadata_file = 'data/vectors/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_metadata_map_20251016_164029.pkl'
with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

print("Chunks on pages 8-9:")
print("=" * 100)

for chunk_idx, meta in metadata_map.items():
    page = meta.get('page_number', '?')
    if page in [8, 9]:
        text = meta.get('text', '')
        print(f"\n{'=' * 100}")
        print(f"Chunk {chunk_idx} - Page {page}")
        print(f"Text length: {len(text)}")
        print(f"First 100 chars: {text[:100]}...")
        print(f"Last 100 chars: ...{text[-100:]}")
        print(f"\nFull text:")
        print(text)
        print("=" * 100)