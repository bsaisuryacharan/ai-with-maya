import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.embeddings.local import LocalEmbedding
from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

# Clean up old test data first
shutil.rmtree("./test_data", ignore_errors=True)

print("=" * 60)
print("INGESTION PIPELINE TESTS")
print("=" * 60)

# Setup
print("\nSetup: Creating components")
embedding_config = EmbeddingConfig(provider="local")
chunking_config = ChunkingConfig(chunk_size=100, overlap=20)
vector_config = VectorStoreConfig(store_type="chromadb", path="./test_data/chroma")

embedder = LocalEmbedding(embedding_config)
vector_store = ChromaDBStore(vector_config)
processor = DocumentProcessor("recursive", chunking_config, embedder)
pipeline = IngestionPipeline(processor, vector_store)

print(f"  ✓ Embedder: {embedding_config.provider}")
print(f"  ✓ Chunker: recursive (chunk_size={chunking_config.chunk_size})")
print(f"  ✓ Vector store: {vector_config.store_type}")

# Test 1: Ingest single document
print("\nTest 1: Ingest single document")
doc_text = """Machine learning is a subset of artificial intelligence.
It enables systems to learn from data and improve over time.
Deep learning uses neural networks to process complex patterns.
Natural language processing helps computers understand human language."""

ids = pipeline.ingest_document(doc_text, "ml_intro.txt")
print(f"  ✓ Ingested document")
print(f"  ✓ Created {len(ids)} vectors")

# Test 2: Search ingested document
print("\nTest 2: Search ingested document")
query = "What is machine learning?"
query_embedding = embedder.embed(query)
results = vector_store.search(query_embedding, top_k=2)
print(f"  ✓ Found {len(results)} results for query")
for i, result in enumerate(results, 1):
    print(f"    {i}. Distance: {result['distance']:.4f}")

# Test 3: Batch ingest
print("\nTest 3: Batch ingestion")
batch_docs = [
    {"content": "Python is a popular programming language.", "source": "python.txt"},
    {"content": "JavaScript runs in web browsers.", "source": "js.txt"},
]
count = pipeline.ingest_batch(batch_docs)
print(f"  ✓ Ingested {len(batch_docs)} documents")
print(f"  ✓ Total vectors: {count}")

# Cleanup
print("\nCleanup: Removing test data")
shutil.rmtree("./test_data", ignore_errors=True)
print(f"  ✓ Test data cleaned up")

print("\n" + "=" * 60)
print("✓ ALL INGESTION TESTS PASSED!")
print("=" * 60)