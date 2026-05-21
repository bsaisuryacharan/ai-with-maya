from typing import List, Dict, Any
from src.chunking.base import Chunk
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveCharacterChunker
from src.embeddings.base import EmbeddingProvider
from src.config.settings import ChunkingConfig
from src.utils.timing import log_timing

class DocumentProcessor:
    """Process documents: chunk and embed."""
    
    def __init__(self, chunker_type: str, chunking_config: ChunkingConfig, embedder: EmbeddingProvider):
        self.embedder = embedder
        
        if chunker_type == "fixed":
            self.chunker = FixedSizeChunker(
                chunk_size=chunking_config.chunk_size,
                overlap=chunking_config.overlap
            )
        elif chunker_type == "recursive":
            self.chunker = RecursiveCharacterChunker(
                chunk_size=chunking_config.chunk_size,
                overlap=chunking_config.overlap
            )
        else:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
    
    @log_timing("process_document")
    def process(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Chunk text and generate embeddings."""
        # Step 1: Chunk
        chunks = self.chunker.chunk(text, source)
        
        # Step 2: Extract chunk texts
        chunk_texts = [c.content for c in chunks]
        
        # Step 3: Generate embeddings
        embeddings = self.embedder.embed_batch(chunk_texts)
        
        # Step 4: Combine
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                "content": chunk.content,
                "embedding": embedding,
                "metadata": {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index
                }
            })
        
        return result