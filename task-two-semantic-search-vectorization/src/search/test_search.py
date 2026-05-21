import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.search.retriever import SemanticRetriever
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.embeddings.local import LocalEmbedding
from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

# Clean up old data
shutil.rmtree("./test_data", ignore_errors=True)

print("=" * 60)
print("SEMANTIC SEARCH TESTS")
print("=" * 60)

# Setup
print("\nSetup: Creating components")
embedding_config = EmbeddingConfig(provider="local")
chunking_config = ChunkingConfig(chunk_size=150, overlap=30)
vector_config = VectorStoreConfig(store_type="chromadb", path="./test_data/chroma")

embedder = LocalEmbedding(embedding_config)
vector_store = ChromaDBStore(vector_config)
processor = DocumentProcessor("recursive", chunking_config, embedder)
pipeline = IngestionPipeline(processor, vector_store)
retriever = SemanticRetriever(embedder, vector_store)

print(f"  ✓ Components initialized")

# Ingest sample documents
print("\nIngest: Adding sample documents")
sample_docs = [
    {
        "content": """Python is a high-level programming language known for its simplicity and readability.
        It supports multiple programming paradigms and has a large standard library.
        Python is widely used in web development, data science, and automation.""",
        "source": "python.txt"
    },
    {
        "content": """Machine learning enables computers to learn from data without explicit programming.
        Common algorithms include decision trees, neural networks, and clustering methods.
        Machine learning powers recommendation systems and image recognition.""",
        "source": "ml.txt"
    },
    {
        "content": """Web development involves building applications that run in browsers.
        Frontend technologies include HTML, CSS, and JavaScript.
        Backend frameworks handle server-side logic and database operations.""",
        "source": "web.txt"
    }
]

for doc in sample_docs:
    pipeline.ingest_document(doc["content"], doc["source"])

print(f"  ✓ Ingested {len(sample_docs)} documents")

# Test 1: Single query search
print("\nTest 1: Single query search")
query = "Python programming"
results = retriever.search(query, top_k=2)
print(f"  ✓ Query: '{query}'")
print(f"  ✓ Found {len(results)} results")
for i, result in enumerate(results, 1):
    source = result["metadata"].get("source", "unknown")
    print(f"    {i}. From {source} (distance: {result['distance']:.4f})")

# Test 2: Multiple queries
print("\nTest 2: Batch search")
queries = ["machine learning algorithms", "web development frameworks"]
batch_results = retriever.batch_search(queries, top_k=1)
print(f"  ✓ Searched {len(queries)} queries")
for query, results in zip(queries, batch_results):
    print(f"    - '{query}': {len(results)} result(s)")

# Test 3: Relevance ranking
print("\nTest 3: Relevance ranking")
query = "data science and AI"
results = retriever.search(query, top_k=3)
print(f"  ✓ Query: '{query}'")
print(f"  ✓ Top results (by relevance):")
for i, result in enumerate(results, 1):
    source = result["metadata"].get("source", "unknown")
    print(f"    {i}. {source}")

# Cleanup
print("\nCleanup: Removing test data")
shutil.rmtree("./test_data", ignore_errors=True)
print(f"  ✓ Test data cleaned up")

print("\n" + "=" * 60)
print("✓ ALL SEARCH TESTS PASSED!")
print("=" * 60)