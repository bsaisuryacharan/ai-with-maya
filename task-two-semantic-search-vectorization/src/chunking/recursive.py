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
        chunk_index = 0
        
        def _split_text(full_text: str, separators: List[str]) -> List[str]:
            """Recursively split on separators, trying easier splits first."""
            separator = separators[-1]
            
            for sep in separators:
                if sep in full_text:
                    separator = sep
                    break
            
            if separator:
                splits = full_text.split(separator)
            else:
                splits = list(full_text)
            
            # Filter out empty strings
            return [s for s in splits if s.strip()]
        
        splits = _split_text(text, self.separators)
        merged = ""
        
        for split in splits:
            if len(merged) + len(split) < self.chunk_size:
                merged += split
            else:
                if merged.strip():
                    chunks.append(Chunk(
                        content=merged,
                        source=source,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                merged = split
        
        # Add last chunk
        if merged.strip():
            chunks.append(Chunk(
                content=merged,
                source=source,
                chunk_index=chunk_index
            ))
        
        return chunks