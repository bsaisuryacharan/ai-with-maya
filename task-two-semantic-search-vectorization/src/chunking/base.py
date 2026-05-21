from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List

@dataclass
class Chunk:
    """Represents a single chunk of text with metadata."""
    content: str
    source: str
    chunk_index: int
    
    def __post_init__(self):
        if not self.content:
            raise ValueError("Chunk content cannot be empty")


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            source: Source document identifier
            
        Returns:
            List of Chunk objects with metadata
        """
        pass