import pickle
import numpy as np
import faiss
from embedders import EmbedderFactory

# Load the FAISS index
faiss_file = 'data/vectors/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_vectors_20251016_161342.faiss'
index = faiss.read_index(faiss_file)

# Load the metadata map
metadata_file = 'data/vectors/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy_metadata_map_20251016_161342.pkl'
with open(metadata_file, 'rb') as f:
    metadata_map = pickle.load(f)

# Initialize embedder
factory = EmbedderFactory()
embedder = factory.create_gemma()

# Test query
query = "Query Rewriting"
print(f"Testing query: '{query}'")

# Generate embedding for the query
query_embedding = embedder.embed(query)
query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

# Normalize the query embedding
faiss.normalize_L2(query_embedding)

# Search the FAISS index
k = 10
distances, indices = index.search(query_embedding, k)

print(f"Top {k} results:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if idx < len(metadata_map):
        meta = metadata_map[idx]
        similarity = 1 - dist  # Convert distance to similarity
        print(f"{i+1}. Chunk {idx}: Page {meta.get('page_number')}, Similarity: {similarity:.4f}")
        print(f"   Preview: {meta.get('text', '')[:100]}...")
    else:
        print(f"{i+1}. Chunk {idx}: INVALID INDEX")
    print()

# Check if chunks 17 and 18 are in the results
print("Checking if chunks 17 and 18 are in results:")
found_17 = any(idx == 17 for idx in indices[0])
found_18 = any(idx == 18 for idx in indices[0])
print(f"Chunk 17 found: {found_17}")
print(f"Chunk 18 found: {found_18}")

# Let's also check the embeddings of chunks 17 and 18 directly
print("\nDirect check of chunks 17 and 18 embeddings:")
for chunk_idx in [17, 18]:
    if chunk_idx < index.ntotal:
        # Get the vector for this chunk
        vector = index.reconstruct(chunk_idx)
        vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        # Compute similarity with query
        similarity = np.dot(query_embedding, vector.T)[0][0]
        print(f"Chunk {chunk_idx}: Direct similarity = {similarity:.4f}")
    else:
        print(f"Chunk {chunk_idx}: No embedding found in FAISS index")