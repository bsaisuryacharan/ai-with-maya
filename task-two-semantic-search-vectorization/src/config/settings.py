from dataclasses import dataclass, field
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: Literal["local", "groq"] = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    model_name: str = "all-MiniLM-L6-v2"
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    embedding_dim: int = 384  # Dimension for all-MiniLM-L6-v2


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    splitter_type: Literal["fixed", "recursive"] = "recursive"


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    store_type: Literal["chromadb", "faiss"] = field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "chromadb"))
    path: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./data/chroma"))
    collection_name: str = "documents"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))