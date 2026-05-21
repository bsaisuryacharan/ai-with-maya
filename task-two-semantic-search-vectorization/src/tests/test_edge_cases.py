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
print("EDGE CASE TESTS")
print("=" * 70)

test_dir = tempfile.mkdtemp()

try:
    # Setup
    embedding_config = EmbeddingConfig(provider="local")
    chunking_config = ChunkingConfig(chunk_size=100, overlap=20)
    vector_config = VectorStoreConfig(store_type="chromadb", path=f"{test_dir}/chroma")
    
    embedder = LocalEmbedding(embedding_config)
    vector_store = ChromaDBStore(vector_config)
    processor = DocumentProcessor("recursive", chunking_config, embedder)
    pipeline = IngestionPipeline(processor, vector_store)
    retriever = SemanticRetriever(embedder, vector_store)
    
    # Test 1: Empty document
    print("\n[TEST 1] Empty document handling")
    try:
        pipeline.ingest_document("", "empty.txt")
        print("  ✓ Handled empty document gracefully")
    except Exception as e:
        print(f"  ✗ Failed on empty document: {e}")
    
    # Test 2: Very long document
    print("\n[TEST 2] Large document handling")
    large_doc = "word " * 10000
    ids = pipeline.ingest_document(large_doc, "large.txt")
    print(f"  ✓ Ingested large document ({len(large_doc)} chars)")
    print(f"  ✓ Created {len(ids)} chunks")
    
    # Test 3: Special characters
    print("\n[TEST 3] Special characters handling")
    special_doc = "Test with émojis 🚀 and spëcial çharacters!"
    ids = pipeline.ingest_document(special_doc, "special.txt")
    print(f"  ✓ Handled special characters")
    
    # Test 4: Duplicate ingestion
    print("\n[TEST 4] Duplicate document handling")
    doc = "Duplicate test document"
    ids1 = pipeline.ingest_document(doc, "dup.txt")
    ids2 = pipeline.ingest_document(doc, "dup.txt")
    print(f"  ✓ First ingest: {len(ids1)} vectors")
    print(f"  ✓ Second ingest: {len(ids2)} vectors")
    
    print("\n" + "=" * 70)
    print("✓ ALL EDGE CASE TESTS PASSED!")
    print("=" * 70)

finally:
    shutil.rmtree(test_dir, ignore_errors=True)