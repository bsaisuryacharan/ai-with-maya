import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def load_document(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')

def chunk_text(text: str, chunk_size: int = 200):
    words = text.split()
    return [
        " ".join( words[i:i+chunk_size] for i in range(0, len(words), chunk_size) )
    ]


def embed_chunks(chunks: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    # embeddings means the vector representations of the text chunks, which can be used for similarity search or other downstream tasks.
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return model, embeddings
    
def embed_query(question: str, model: SentenceTransformer):
    q_model = model.encode( [question], convert_to_numpy=True, normalize_embeddings=True)
    return q_model[0]

def retrieve_relevant_chunks(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, top_k: int = 3):
    # chunk_embeddings:  (N, D), query_emb: (D, )
    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
    similarities = query_embedding @ chunk_embeddings.T # dot product which will give us the similarity between the query and each chunk
    top_index = np.argsort(similarities)[:top_k] # get the indices of the top k most similar chunks
    top_scores = similarities[top_index] # get the similarity scores of the top k chunks
    return top_index.tolist(), top_scores.tolist()

def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Single document RAG demo")
    parser.add_argument("--document", required=True,help="Path to the text document")
    parser.add_argument("--question", required=True, help="Question to ask about the document")
    args = parser.parse_args()

    # Step 2: Load the data
    document_text = load_document(args.document)

    # Step 3: Chunk the document (length of the document in characters, and the number of chunks created Ex: 301 words and chunk size of 200 would create 2 chunks)
    chunks = chunk_text(document_text)
    
    # Step 4: Embed the chunks
    model, data_embeddings = embed_chunks(chunks)

    # Step 5: Embed the query
    query_embedding = embed_query(args.question, model)

    # Step 6: Retrieve relevant chunks
    top_indices, top_scores = retrieve_relevant_chunks(query_embedding, data_embeddings)

    print("Document content:")
    print(f"Characters: {len(document_text)}")
    print(f"Chunks: {len(chunks)}")
    print(f"\nQuestion: {args.question}")
    print(f"Embeddings shape: {data_embeddings.shape}")
    print(f"Top relevant chunks: {top_indices}")
    print(f"Top scores: {top_scores}")
    print("Chunks retrieved:")
    for i in top_indices:
        print(f"  - {chunks[i]}")

if __name__ == "__main__":
    main()