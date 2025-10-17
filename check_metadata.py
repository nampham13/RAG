import pickle
import sys
sys.path.append('C:/Users/ENGUYEHWC/Prototype/Version_03/RAG')

# Load the metadata for the working FAISS file
try:
    with open('C:/Users/ENGUYEHWC/Prototype/Version_03/RAG/data/vectors/Process_Service Configuration Management_metadata_map_20251015_170035.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print('Number of chunks in metadata:', len(metadata))
    print()

    # Show first few chunks
    for i, (chunk_id, chunk_data) in enumerate(list(metadata.items())[:5]):
        print(f'Chunk {i+1}:')
        print(f'  ID: {chunk_id}')
        text = chunk_data.get('text', '')
        print(f'  Text length: {len(text)}')
        print(f'  Text preview: {repr(text[:200])}')
        print(f'  Page: {chunk_data.get("page_number", "N/A")}')
        print(f'  Keys: {list(chunk_data.keys())}')
        print()
except Exception as e:
    print(f'Error loading metadata: {e}')