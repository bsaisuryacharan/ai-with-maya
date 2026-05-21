from typing import List, Dict, Any
from src.ingestion.document_processor import DocumentProcessor
from src.vector_store.base import VectorStore
from src.utils.timing import log_timing

class IngestionPipeline:
    """End-to-end document ingestion pipeline."""
    
    def __init__(self, processor: DocumentProcessor, vector_store: VectorStore):
        self.processor = processor
        self.vector_store = vector_store
    
    @log_timing("ingest_document")
    def ingest_document(self, text: str, source: str) -> List[str]:
        """Ingest a single document."""
        processed = self.processor.process(text, source)
        
        vectors = [p["embedding"] for p in processed]
        metadata = [
            {
                **p["metadata"],
                "content": p["content"],
            }
            for p in processed
        ]
        
        ids = self.vector_store.add(vectors, metadata)
        return ids
    
    @log_timing("ingest_batch")
    def ingest_batch(self, docs: List[Dict[str, Any]]) -> int:
        """Ingest multiple documents."""
        total_ids = []
        for doc in docs:
            ids = self.ingest_document(doc["content"], doc["source"])
            total_ids.extend(ids)
        
        return len(total_ids)