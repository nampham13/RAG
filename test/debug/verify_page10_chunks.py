import pickle

# Load the metadata map
metadata_file = 'data/vectors/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_metadata_map_20251016_161342.pkl'
with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

print('Total chunks:', len(metadata_map))
print('Type of metadata_map:', type(metadata_map))
print()

# Find chunks on page 10
page_10_chunks = [chunk_idx for chunk_idx, meta in metadata_map.items() if meta.get('page_number') == 10]
print(f'Chunks on page 10: {page_10_chunks}')
print()

# Show content of page 10 chunks
for chunk_idx in page_10_chunks:
    meta = metadata_map[chunk_idx]
    print(f'Chunk {chunk_idx}:')
    print(f'  Page: {meta.get("page_number")}')
    print(f'  Content preview: {meta.get("text", "")[:400]}...')
    print()

# Also search for "Query Rewriting" in all chunks
print('Searching for "Query Rewriting" in all chunks:')
found_chunks = []
for chunk_idx, meta in metadata_map.items():
    text = meta.get('text', '').lower()
    if 'query rewriting' in text:
        found_chunks.append(chunk_idx)
        print(f'Found in Chunk {chunk_idx} (Page {meta.get("page_number")}):')
        # Show the full text to see the actual content
        full_text = meta.get('text', '')
        print(f'  Full Content: {full_text}')
        print()

print(f'Total chunks containing "query rewriting": {len(found_chunks)}')

# Let's also search for just "rewriting" to see if there are other variations
print('\nSearching for "rewriting" in all chunks:')
rewriting_chunks = []
for chunk_idx, meta in metadata_map.items():
    text = meta.get('text', '').lower()
    if 'rewriting' in text:
        rewriting_chunks.append(chunk_idx)
        print(f'Found "rewriting" in Chunk {chunk_idx} (Page {meta.get("page_number")}):')
        # Find the context around "rewriting"
        full_text = meta.get('text', '')
        start_idx = text.find('rewriting')
        context_start = max(0, start_idx - 100)
        context_end = min(len(full_text), start_idx + 200)
        print(f'  Context: ...{full_text[context_start:context_end]}...')
        print()

print(f'Total chunks containing "rewriting": {len(rewriting_chunks)}')