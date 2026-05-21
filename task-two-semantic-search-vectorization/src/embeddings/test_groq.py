import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import patch, MagicMock
from src.embeddings.groq import GroqEmbedding
from src.config.settings import EmbeddingConfig

print("=" * 60)
print("GROQ EMBEDDING TESTS (MOCKED)")
print("=" * 60)

# Test 1: Initialize
print("\nTest 1: GROQ Embedder initialization")
config = EmbeddingConfig(provider="groq", groq_api_key="test-key")
embedder = GroqEmbedding(config)
print(f"  ✓ Embedding dimension: {embedder.embedding_dim}")
print(f"  ✓ Model: {embedder.model}")

# Test 2: Mock single embedding
print("\nTest 2: Single embedding (mocked)")
with patch('src.embeddings.groq.Groq') as MockGroq:
    mock_client = MagicMock()
    MockGroq.return_value = mock_client
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1536)
    ]
    
    config = EmbeddingConfig(provider="groq", groq_api_key="test-key")
    embedder = GroqEmbedding(config)
    embedding = embedder.embed("Test text")
    
    print(f"  ✓ Got embedding with {len(embedding)} dimensions")
    print(f"  ✓ API called: {mock_client.embeddings.create.called}")

# Test 3: Mock batch embedding
print("\nTest 3: Batch embedding (mocked)")
with patch('src.embeddings.groq.Groq') as MockGroq:
    mock_client = MagicMock()
    MockGroq.return_value = mock_client
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
        MagicMock(embedding=[0.3] * 1536)
    ]
    
    config = EmbeddingConfig(provider="groq", groq_api_key="test-key")
    embedder = GroqEmbedding(config)
    embeddings = embedder.embed_batch(["Text 1", "Text 2", "Text 3"])
    
    print(f"  ✓ Got {len(embeddings)} embeddings")
    print(f"  ✓ Each with {len(embeddings[0])} dimensions")

print("\n" + "=" * 60)
print("✓ ALL GROQ TESTS PASSED (MOCKED)!")
print("=" * 60)
print("\nNote: Real GROQ API requires GROQ_API_KEY env variable")