from typing import List
from src.chunking.base import Chunk, ChunkingStrategy

class RecursiveCharacterChunker(ChunkingStrategy):
    """Recursive character splitting that respects text boundaries."""
    
    def __init__(self, chunk_size: int, overlap: int = 0, separators: List[str] = None):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in range [0, chunk_size)")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Default separators: paragraph, newline, space, character
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def chunk(self, text: str, source: str) -> List[Chunk]:
        """Split text recursively on separators, preferring better boundaries."""
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                for separator in self.separators:
                    if not separator:
                        continue
                    split_at = text.rfind(separator, start, end)
                    if split_at > start:
                        end = split_at + len(separator)
                        break

            if end <= start:
                end = min(start + self.chunk_size, text_len)

            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text,
                    source=source,
                    chunk_index=len(chunks)
                ))

            if end >= text_len:
                break

            next_start = end - self.overlap if self.overlap > 0 else end
            start = max(next_start, 0)

        return chunks
