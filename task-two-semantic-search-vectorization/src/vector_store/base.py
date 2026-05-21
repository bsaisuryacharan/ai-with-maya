from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    @abstractmethod
    def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        """Store vectors with metadata. Returns IDs."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors. Returns results with metadata."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors."""
        pass