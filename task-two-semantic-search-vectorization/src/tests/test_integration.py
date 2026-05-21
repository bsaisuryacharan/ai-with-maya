import sys
from pathlib import Path
import shutil
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.document_processor import DocumentProcessor
from src.search.retriever import SemanticRetriever
from src.embeddings.local import LocalEmbedding
from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

print("=" * 70)
print("END-TO-END INTEGRATION TESTS")
print("=" * 70)

# Create temp directory
test_dir = tempfile.mkdtemp()

try:
    # Setup
    print("\n[SETUP] Initializing system")
    embedding_config = EmbeddingConfig(provider="local")
    chunking_config = ChunkingConfig(chunk_size=200, overlap=50)
    vector_config = VectorStoreConfig(store_type="chromadb", path=f"{test_dir}/chroma")
    
    embedder = LocalEmbedding(embedding_config)
    vector_store = ChromaDBStore(vector_config)
    processor = DocumentProcessor("recursive", chunking_config, embedder)
    pipeline = IngestionPipeline(processor, vector_store)
    retriever = SemanticRetriever(embedder, vector_store)
    print("  ✓ System initialized")
    
    # Test 1: Full pipeline
    print("\n[TEST 1] Full ingestion → search pipeline")
    documents = [
        {
            "content": """Artificial Intelligence is transforming industries.
            Machine learning algorithms learn from data patterns.
            Deep learning uses neural networks with many layers.""",
            "source": "ai_intro.md"
        },
        {
            "content": """Natural Language Processing helps computers understand text.
            Named entity recognition identifies people, places, and things.
            Sentiment analysis determines emotional tone.""",
            "source": "nlp_guide.md"
        },
        {
            "content": """Computer Vision enables machines to interpret images.
            Object detection identifies and locates objects.
            Image segmentation divides images into regions.""",
            "source": "cv_basics.md"
        }
    ]
    
    for doc in documents:
        pipeline.ingest_document(doc["content"], doc["source"])
    
    print(f"  ✓ Ingested {len(documents)} documents")
    
    # Test 2: Multi-query search
    print("\n[TEST 2] Multi-query search")
    queries = [
        "machine learning and neural networks",
        "text processing and emotion detection",
        "image analysis techniques"
    ]
    
    for query in queries:
        results = retriever.search(query, top_k=2)
        print(f"  ✓ Query '{query}': {len(results)} results")
        assert len(results) > 0, f"No results for query: {query}"
    
    # Test 3: Batch search
    print("\n[TEST 3] Batch search efficiency")
    batch_results = retriever.batch_search(queries, top_k=2)
    print(f"  ✓ Batch searched {len(queries)} queries")
    print(f"  ✓ Total results: {sum(len(r) for r in batch_results)}")
    
    # Test 4: Result quality
    print("\n[TEST 4] Result quality validation")
    query = "deep learning networks"
    results = retriever.search(query, top_k=1)
    
    if results:
        top_result = results[0]
        source = top_result["metadata"].get("source", "unknown")
        distance = top_result["distance"]
        print(f"  ✓ Top result from: {source}")
        print(f"  ✓ Relevance distance: {distance:.4f}")
        assert distance < 1.0, "Invalid distance value"
    
    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)

finally:
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)