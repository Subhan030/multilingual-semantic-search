Multilingual Semantic Search with Cross-Lingual RAG

An interactive multilingual semantic search application that allows users to search documents by meaning rather than keywords, even across different languages.
The app also supports Retrieval-Augmented Generation (RAG) using Groq LLMs to generate concise answers grounded strictly in retrieved content.

Features

Cross-lingual semantic search

Search English queries over Hindi documents (and vice versa)

Powered by multilingual sentence embeddings

FAISS vector search

Efficient similarity search using cosine similarity

Sentence-level relevance

Retrieves the most relevant 2–3 sentences instead of large chunks

Retrieval-Augmented Generation (RAG)

Optional answer generation using Groq LLMs

Answers are grounded strictly in retrieved context

Language-aware answers

RAG answers are generated in the query’s language

User-friendly UI

Paste text, upload .txt files, or load sample documents

Deployment-ready

Works locally and on Streamlit Cloud

How It Works

Document ingestion
Documents are split into chunks

Embedding
Each chunk is converted into a multilingual vector representation

Vector indexing
FAISS is used with cosine similarity (inner product)

Semantic search
Queries are embedded and matched against document vectors

Sentence re-ranking
The most relevant sentences are extracted from matched chunks

RAG (optional)
Retrieved sentences are passed to a Groq LLM to generate an answer

Tech Stack

Python

Streamlit – interactive UI

SentenceTransformers – multilingual embeddings

FAISS (CPU) – vector similarity search

Groq API – fast LLM inference for RAG
