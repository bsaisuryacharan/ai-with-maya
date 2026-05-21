from src.config.settings import AppConfig, EmbeddingConfig

# Try creating a config
config = AppConfig()
print(f"Embedding provider: {config.embedding.provider}")
print(f"Chunk size: {config.chunking.chunk_size}")
print(f"Vector store: {config.vector_store.store_type}")
print(f"Log level: {config.log_level}")
print("✓ Configuration loaded successfully!")