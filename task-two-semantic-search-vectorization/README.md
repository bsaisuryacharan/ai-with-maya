# Semantic Search with Vector Embeddings

A complete end-to-end semantic search system using local embeddings, vector storage, and retrieval.

## Features

- **Text Chunking**: Fixed-size and recursive character splitting strategies
- **Embeddings**: Local (Sentence Transformers) and cloud (GROQ) providers
- **Vector Storage**: ChromaDB persistent storage with cosine similarity search
- **Document Ingestion**: End-to-end pipeline for chunking, embedding, and storing
- **Semantic Search**: Query-based retrieval with ranking
- **Testing**: Integration, performance, and edge-case tests

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python src/chunking/test_chunking.py
python src/embeddings/test_embeddings.py
python src/vector_store/test_vector_store.py
python src/ingestion/test_ingestion.py
python src/search/test_search.py
python src/tests/test_integration.py
python src/tests/test_performance.py
python src/tests/test_edge_cases.py
```

### 3. Use the System

```python
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.document_processor import DocumentProcessor
from src.search.retriever import SemanticRetriever
from src.embeddings.local import LocalEmbedding
from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

# Setup
embedder = LocalEmbedding(EmbeddingConfig(provider="local"))
vector_store = ChromaDBStore(VectorStoreConfig(store_type="chromadb", path="./data"))
processor = DocumentProcessor("recursive", ChunkingConfig(), embedder)
pipeline = IngestionPipeline(processor, vector_store)
retriever = SemanticRetriever(embedder, vector_store)

# Ingest
pipeline.ingest_document("Your text here...", "source.txt")

# Search
results = retriever.search("query text", top_k=5)
for result in results:
    print(f"- {result['metadata']['source']}")
```

## Architecture

```
┌─────────────────────────────────────────┐
│   Document Input                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Chunking (Fixed / Recursive)          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Embeddings (Local / GROQ)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Vector Store (ChromaDB)               │
└──────────────┬──────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Search Query │
        └──────┬───────┘
               │
               ▼
    ┌──────────────────────┐
    │ Return Top-K Results │
    └──────────────────────┘
```

## Configuration

Set environment variables in `.env`:

```env
EMBEDDING_PROVIDER=local
VECTOR_STORE_TYPE=chromadb
VECTOR_STORE_PATH=./data/chroma
CHUNK_SIZE=512
CHUNK_OVERLAP=100
GROQ_API_KEY=your_key_here
```

## Components

| Module | Purpose |
|--------|---------|
| `chunking/` | Text splitting strategies |
| `embeddings/` | Embedding generation |
| `vector_store/` | Vector storage backends |
| `ingestion/` | Document processing pipeline |
| `search/` | Semantic search & retrieval |
| `config/` | Configuration management |
| `utils/` | Utilities (timing, logging) |

## API Reference

### IngestionPipeline

```python
pipeline = IngestionPipeline(processor, vector_store)
ids = pipeline.ingest_document(text, source)
count = pipeline.ingest_batch(documents)
```

### SemanticRetriever

```python
retriever = SemanticRetriever(embedder, vector_store)
results = retriever.search(query, top_k=5)
batch_results = retriever.batch_search(queries, top_k=5)
```

## Performance

- **Ingestion**: ~135ms per document (with embeddings)
- **Search**: <2s for single query
- **Batch Search**: <5s for 10 queries

## License

MIT
