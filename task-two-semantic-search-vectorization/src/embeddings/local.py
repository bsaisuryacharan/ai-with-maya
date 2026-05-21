from typing import List
from sentence_transformers import SentenceTransformer
from src.embeddings.base import EmbeddingProvider
from src.config.settings import EmbeddingConfig
from src.utils.timing import log_timing

class LocalEmbedding(EmbeddingProvider):
    """Local embedding generation using Sentence Transformers."""
    
    def __init__(self, config: EmbeddingConfig):
        if config.provider != "local":
            raise ValueError("LocalEmbedding requires provider='local'")
        
        self.model = SentenceTransformer(config.model_name)
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            self._embedding_dim = int(self.model.get_sentence_embedding_dimension())
        else:
            self._embedding_dim = config.embedding_dim
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @log_timing("embed_single")
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    @log_timing("embed_batch")
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [e.tolist() for e in embeddings]
