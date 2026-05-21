from typing import List
from src.chunking.base import Chunk, ChunkingStrategy

class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with configurable overlap."""

    def __init__(self, chunk_size: int, overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in range [0, chunk_size)")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():  # Skip empty chunks
                chunks.append(Chunk(
                    content=chunk_text,
                    source=source,
                    chunk_index=len(chunks)
                ))

        return chunks
