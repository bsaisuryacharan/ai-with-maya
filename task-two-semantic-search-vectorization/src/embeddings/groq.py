from typing import List
from groq import Groq
from src.embeddings.base import EmbeddingProvider
from src.config.settings import EmbeddingConfig
from src.utils.timing import log_timing

class GroqEmbedding(EmbeddingProvider):
    """Cloud-based embedding generation using GROQ API."""
    
    def __init__(self, config: EmbeddingConfig):
        if config.provider != "groq":
            raise ValueError("GroqEmbedding requires provider='groq'")
        if not config.groq_api_key:
            raise ValueError("groq_api_key is required")
        
        self.client = Groq(api_key=config.groq_api_key)
        self.model = "text-embedding-3-small"
        self._embedding_dim = 1536
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @log_timing("groq_embed_single")
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text via GROQ."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    @log_timing("groq_embed_batch")
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts via GROQ."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]