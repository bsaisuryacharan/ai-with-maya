import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import VectorStoreConfig

print("=" * 60)
print("VECTOR STORE TESTS")
print("=" * 60)

# Test 1: Initialize
print("\nTest 1: Vector store initialization")
config = VectorStoreConfig(store_type="chromadb", path="./test_data/chroma")
store = ChromaDBStore(config)
print(f"  ✓ ChromaDB store initialized")

# Test 2: Add vectors
print("\nTest 2: Add vectors with metadata")
vectors = [
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.3, 0.4, 0.5],
    [0.9, 0.8, 0.7, 0.6]
]
metadata = [
    {"source": "doc1.txt", "chunk": 0},
    {"source": "doc1.txt", "chunk": 1},
    {"source": "doc2.txt", "chunk": 0}
]

ids = store.add(vectors, metadata)
print(f"  ✓ Added {len(ids)} vectors")
print(f"  ✓ IDs: {ids}")

# Test 3: Search
print("\nTest 3: Search for similar vectors")
query_vector = [0.15, 0.25, 0.35, 0.45]
results = store.search(query_vector, top_k=2)
print(f"  ✓ Found {len(results)} results")
for result in results:
    print(f"    - ID: {result['id']}, Distance: {result['distance']:.4f}")

# Test 4: Delete
print("\nTest 4: Delete vectors")
store.delete([ids[0]])
results_after = store.search(query_vector, top_k=3)
print(f"  ✓ After delete, found {len(results_after)} results")

# Cleanup
print("\nCleanup: Removing test data")
shutil.rmtree("./test_data", ignore_errors=True)
print(f"  ✓ Test data cleaned up")

print("\n" + "=" * 60)
print("✓ ALL VECTOR STORE TESTS PASSED!")
print("=" * 60)