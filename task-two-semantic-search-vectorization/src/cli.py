import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.document_loader import DocumentLoader
from src.search.retriever import SemanticRetriever
from src.embeddings.local import LocalEmbedding
from src.vector_store.chromadb_store import ChromaDBStore
from src.config.settings import EmbeddingConfig, ChunkingConfig, VectorStoreConfig

def setup_system(embedding_type="local"):
    """Initialize the search system."""
    embedding_config = EmbeddingConfig(provider=embedding_type)
    chunking_config = ChunkingConfig()
    vector_config = VectorStoreConfig()

    embedder = LocalEmbedding(embedding_config)
    vector_store = ChromaDBStore(vector_config)
    processor = DocumentProcessor(chunking_config.splitter_type, chunking_config, embedder)
    pipeline = IngestionPipeline(processor, vector_store)
    retriever = SemanticRetriever(embedder, vector_store)

    return pipeline, retriever

def ingest_command(args):
    """Ingest documents."""
    pipeline, _ = setup_system(args.embedder)

    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
        ids = pipeline.ingest_document(text, args.file)
        print(f"✓ Ingested {args.file}")
        print(f"✓ Created {len(ids)} vectors")

    elif args.directory:
        docs = DocumentLoader.load_text_files(args.directory)
        count = pipeline.ingest_batch(docs)
        print(f"✓ Ingested {len(docs)} documents")
        print(f"✓ Created {count} total vectors")

def search_command(args):
    """Search documents."""
    _, retriever = setup_system(args.embedder)

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        distance = result["distance"]
        content = result.get("content") if result.get("content") is not None else metadata.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        preview = content[:100] + ("..." if len(content) > 100 else "")

        print(f"{i}. [{source}] (distance: {distance:.4f})")
        print(f"   {preview}\n")

def main():
    parser = argparse.ArgumentParser(description="Semantic Search System")
    parser.add_argument("--embedder", choices=["local"], default="local")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest = subparsers.add_parser("ingest", help="Ingest documents")
    ingest.add_argument("--file", help="Single file to ingest")
    ingest.add_argument("--directory", help="Directory of files to ingest")

    # Search command
    search = subparsers.add_parser("search", help="Search documents")
    search.add_argument("query", help="Search query")
    search.add_argument("--top-k", type=int, default=5, help="Top K results")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
