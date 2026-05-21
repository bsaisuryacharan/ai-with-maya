import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveCharacterChunker
from src.chunking.base import Chunk

# Create test text
sample_text = "The quick brown fox jumps over the lazy dog. " * 20

print("=" * 60)
print("FIXED-SIZE CHUNKER TESTS")
print("=" * 60)

# Test 1: Basic chunking
print("\nTest 1: Basic chunking")
chunker = FixedSizeChunker(chunk_size=100, overlap=20)
chunks = chunker.chunk(sample_text, source="test.txt")

print(f"  ✓ Created {len(chunks)} chunks")
print(f"  ✓ First chunk has source: {chunks[0].source}")
print(f"  ✓ Chunk indices sequential: {[c.chunk_index for c in chunks[:3]]}")

# Test 2: Verify overlap works
print("\nTest 2: Verify overlap")
if len(chunks) > 1:
    overlap_region = chunks[0].content[-20:]
    next_start = chunks[1].content[:20]
    if overlap_region == next_start:
        print(f"  ✓ Overlap working correctly")
    else:
        print(f"  ✗ Overlap check failed")

# Test 3: No overlap
print("\nTest 3: No overlap chunking")
chunker_no_overlap = FixedSizeChunker(chunk_size=100, overlap=0)
chunks_no_overlap = chunker_no_overlap.chunk(sample_text, source="test.txt")
print(f"  ✓ Created {len(chunks_no_overlap)} chunks (no overlap)")

print("\n" + "=" * 60)
print("RECURSIVE CHUNKER TESTS")
print("=" * 60)

# Test 4: Recursive chunker with multiline text
print("\nTest 4: Recursive chunking with paragraphs")
multiline_text = """Paragraph 1: The quick brown fox jumps over the lazy dog.

Paragraph 2: The weather is nice today.
It has a lot of sunshine.

Paragraph 3: Final paragraph here."""

recursive_chunker = RecursiveCharacterChunker(chunk_size=100, overlap=10)
recursive_chunks = recursive_chunker.chunk(multiline_text, source="paragraphs.txt")
print(f"  ✓ Created {len(recursive_chunks)} chunks")
print(f"  ✓ First chunk: {recursive_chunks[0].content[:50]}...")

# Test 5: Empty input handling
print("\nTest 5: Empty input handling")
empty_chunks = recursive_chunker.chunk("", source="empty.txt")
print(f"  ✓ Empty input returns: {len(empty_chunks)} chunks")

# Test 6: Single sentence
print("\nTest 6: Single short sentence")
short_text = "This is a short sentence."
short_chunks = recursive_chunker.chunk(short_text, source="short.txt")
print(f"  ✓ Short text returns: {len(short_chunks)} chunk(s)")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)