from typing import List, Dict, Any
from uuid import uuid4
import chromadb
from src.vector_store.base import VectorStore
from src.config.settings import VectorStoreConfig
from src.utils.timing import log_timing

class ChromaDBStore(VectorStore):
    """Vector store implementation using ChromaDB."""

    def __init__(self, config: VectorStoreConfig):
        if config.store_type != "chromadb":
            raise ValueError("ChromaDBStore requires store_type='chromadb'")

        self.client = chromadb.PersistentClient(path=config.path)
        self.collection_name = config.collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    @log_timing("vector_add")
    def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors with metadata to ChromaDB."""
        ids = [str(uuid4()) for _ in range(len(vectors))]
        documents = [str(m.get("content", "")) for m in metadata]
        self.collection.add(
            embeddings=vectors,
            metadatas=metadata,
            documents=documents,
            ids=ids
        )
        return ids
    
    @log_timing("vector_search")
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["distances", "metadatas", "documents"]
        )
        
        output = []
        if results["ids"] and len(results["ids"]) > 0:
            documents = results.get("documents")
            metadatas = results.get("metadatas")
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = metadatas[0][i] if metadatas and i < len(metadatas[0]) else {}
                if documents and i < len(documents[0]) and documents[0][i]:
                    content = documents[0][i]
                else:
                    content = str(metadata.get("content", ""))
                output.append({
                    "id": doc_id,
                    "distance": results["distances"][0][i],
                    "metadata": metadata,
                    "content": content
                })
        
        return output
    
    @log_timing("vector_delete")
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        self.collection.delete(ids=ids)
    
    def clear(self) -> None:
        """Clear all vectors."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
