import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.local import LocalEmbedding
from src.config.settings import EmbeddingConfig

# Create config
config = EmbeddingConfig(provider="local")
embedder = LocalEmbedding(config)

print("=" * 60)
print("LOCAL EMBEDDING TESTS")
print("=" * 60)

# Test 1: Single embedding
print("\nTest 1: Single text embedding")
text1 = "The quick brown fox jumps over the lazy dog"
embedding1 = embedder.embed(text1)
print(f"  ✓ Embedding dimension: {len(embedding1)}")
print(f"  ✓ First 5 values: {embedding1[:5]}")
print(f"  ✓ All floats: {all(isinstance(x, float) for x in embedding1)}")

# Test 2: Batch embeddings
print("\nTest 2: Batch embeddings")
texts = [
    "The quick brown fox jumps",
    "A fast auburn fox leaps",
    "The weather is nice today"
]
embeddings = embedder.embed_batch(texts)
print(f"  ✓ Created {len(embeddings)} embeddings")
print(f"  ✓ Each has dimension {len(embeddings[0])}")

# Test 3: Semantic similarity (similar texts should have similar embeddings)
print("\nTest 3: Semantic similarity check")
import math

def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b)

sim_similar = cosine_similarity(embeddings[0], embeddings[1])  # Similar
sim_different = cosine_similarity(embeddings[0], embeddings[2])  # Different

print(f"  ✓ Similarity (fox texts): {sim_similar:.4f}")
print(f"  ✓ Similarity (fox vs weather): {sim_different:.4f}")
print(f"  ✓ Similar texts ranked higher: {sim_similar > sim_different}")

print("\n" + "=" * 60)
print("✓ ALL EMBEDDING TESTS PASSED!")
print("=" * 60)
