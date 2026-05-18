import argparse
from pathlib import Path

def load_document(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')

def chunk_text(text:str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def main():
    parser = argparse.ArgumentParser(description="Single document RAG demo")
    parser.add_argument("--document", required=True,help="Path to the text document")
    parser.add_argument("--question", required=True, help="Question to ask about the document")
    args = parser.parse_args()

    document_text = load_document(args.document)
    chunks = chunk_text(document_text)

    print("Document content:")
    print(f"Characters: {len(document_text)}")
    print(f"Chunks: {len(chunks)}")
    #print(f"First chunk preview: {chunks[0][:200] if chunks else 'No chunks found'}")
    print(f"\nQuestion: {args.question}")

if __name__ == "__main__":
    main()