import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import requests

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")


def load_document(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')

def chunk_text(text: str, chunk_size: int = 200):
    words = text.split()
    return [
        " ".join( words[i:i+chunk_size]) for i in range(0, len(words), chunk_size) 
    ]


def embed_chunks(chunks: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    # embeddings means the vector representations of the text chunks, which can be used for similarity search or other downstream tasks.
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return embedder, embeddings
    
def embed_query(question: str, model: SentenceTransformer):
    q_model = model.encode( [question], convert_to_numpy=True, normalize_embeddings=True)
    return q_model[0]

def retrieve_relevant_chunks(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, top_k: int = 3):
    # chunk_embeddings:  (N, D), query_emb: (D, )
    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
    similarities = query_embedding @ chunk_embeddings.T # dot product which will give us the similarity between the query and each chunk
    top_index = np.argsort(-similarities)[:top_k] # get the indices of the top k most similar chunks
    top_scores = similarities[top_index] # get the similarity scores of the top k chunks
    return top_index.tolist(), top_scores.tolist()


def build_prompt(chunks: list[str], top_indices: list[int], question: str) -> str:
    context_parts = []
    for rank, index in enumerate( top_indices, start = 1 ):
        context_parts.append(f"[Chunk {rank}: {chunks[index]}]")
    context = "\n".join(context_parts)
    prompt = (
        "You are given CONTEXT extracted from a single document. "
        "Answer the QUESTION using ONLY the information in the CONTEXT. "
        "If the answer cannot be found in the CONTEXT, respond exactly: "
        "\"I don't know based on the provided document.\" Do not use outside knowledge.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n"
        "Provide a concise answer and optionally mention which chunk(s) you used."
    )
    return prompt

def call_groq(prompt: str, model: str="llama-3.1-8b-instant"):
    if not GROQ_API_KEY:
        raise ValueError("Missing required environment variable: GROQ_API_KEY")
    if not GROQ_API_URL:
        raise ValueError("Missing required environment variable: GROQ_API_URL")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected Groq API response format") from exc


def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Single document RAG demo")
    parser.add_argument("--document", required=True,help="Path to the text document")
    parser.add_argument("--question", required=True, help="Question to ask about the document")
    args = parser.parse_args()

    # Step 2: Load the data
    document_text = load_document(args.document)

    # Step 3: Chunk the document by words (e.g., 301 words with chunk size 200 -> 2 chunks)
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
    # print("Chunks retrieved:")
    # for i in top_indices:
    #     print(f"  - {chunks[i]}")

    # Step 7: Build the prompt
    prompt = build_prompt(chunks, top_indices, args.question)
    print("\n--- LLM Prompt (preview) ---\n")
    print(prompt[:4000])   # show up to 4000 chars for review
    print("\n--- end prompt preview ---\n")

    # Step 8: Call the LLM
    answer = call_groq(prompt)
    print(f"Answer:\n{answer}")

if __name__ == "__main__":
    main()
