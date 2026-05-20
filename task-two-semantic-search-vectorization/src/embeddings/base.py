from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """Abstract base class for embedding generation."""
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the embedding vectors."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        pass