# AI-Engineering
Repository for foundational RAG tasks as part of the AI Engineering curriculum.
Task 1: Simple RAG
Description: Make a simple `task-one-simple-rag/single_doc_rag.py` where it takes document path and query as parameters and it
  1. loads data
  2. chunks doc content
  3. embed doc and query
  4. retrieve top similar indices and scores
  5. build a system prompt with instructions and chunks of data
  6. Generate answer from llm

Example:
`python task-one-simple-rag/single_doc_rag.py --document task-one-simple-rag/data/test_doc.txt --question "What is this file about?"`
