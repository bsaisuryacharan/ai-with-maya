from typing import List, Dict, Any
from src.embeddings.base import EmbeddingProvider
from src.vector_store.base import VectorStore
from src.utils.timing import log_timing

class SemanticRetriever:
    """Retrieve relevant documents using semantic search."""
    
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
    
    @log_timing("semantic_search")
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of results with content and metadata
        """
        # Generate embedding for query
        query_embedding = self.embedder.embed(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
    
    @log_timing("batch_search")
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Search for multiple queries efficiently."""
        # Generate embeddings for all queries
        query_embeddings = self.embedder.embed_batch(queries)
        
        # Search for each query
        all_results = []
        for embedding in query_embeddings:
            results = self.vector_store.search(embedding, top_k=top_k)
            all_results.append(results)
        
        return all_results