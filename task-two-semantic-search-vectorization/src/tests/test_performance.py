import sys
from pathlib import Path
import shutil
import tempfile
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.document_processor import DocumentProcessor
from src.search.retriever import SemanticRetriever
from src.embeddings.local import LocalEmbedding
from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

print("=" * 70)
print("PERFORMANCE BENCHMARKS")
print("=" * 70)

test_dir = tempfile.mkdtemp()

try:
    # Setup
    embedding_config = EmbeddingConfig(provider="local")
    chunking_config = ChunkingConfig(chunk_size=256, overlap=64)
    vector_config = VectorStoreConfig(store_type="chromadb", path=f"{test_dir}/chroma")
    
    embedder = LocalEmbedding(embedding_config)
    vector_store = ChromaDBStore(vector_config)
    processor = DocumentProcessor("recursive", chunking_config, embedder)
    pipeline = IngestionPipeline(processor, vector_store)
    retriever = SemanticRetriever(embedder, vector_store)
    
    # Benchmark 1: Ingestion speed
    print("\n[BENCHMARK 1] Ingestion Speed")
    doc_text = "Lorem ipsum dolor sit amet. " * 100
    
    start = time.time()
    pipeline.ingest_document(doc_text, "perf_test.txt")
    ingest_time = time.time() - start
    
    print(f"  Time to ingest document: {ingest_time*1000:.2f}ms")
    assert ingest_time < 5.0, "Ingestion too slow"
    
    # Benchmark 2: Search speed
    print("\n[BENCHMARK 2] Search Speed")
    query = "Lorem ipsum"
    
    start = time.time()
    results = retriever.search(query, top_k=5)
    search_time = time.time() - start
    
    print(f"  Time for search: {search_time*1000:.2f}ms")
    print(f"  Results returned: {len(results)}")
    assert search_time < 2.0, "Search too slow"
    
    # Benchmark 3: Batch search efficiency
    print("\n[BENCHMARK 3] Batch Search Efficiency")
    queries = [f"query {i}" for i in range(10)]
    
    start = time.time()
    batch_results = retriever.batch_search(queries, top_k=3)
    batch_time = time.time() - start
    
    print(f"  Time for 10 queries: {batch_time*1000:.2f}ms")
    print(f"  Avg time per query: {batch_time*100:.2f}ms")
    assert batch_time < 5.0, "Batch search too slow"
    
    print("\n" + "=" * 70)
    print("✓ ALL PERFORMANCE BENCHMARKS PASSED!")
    print("=" * 70)

finally:
    shutil.rmtree(test_dir, ignore_errors=True)