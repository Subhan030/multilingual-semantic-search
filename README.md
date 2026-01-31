Multilingual Semantic Search with Cross-Lingual RAG

An interactive multilingual semantic search application that allows users to search documents by meaning rather than keywords, even across different languages.
The app also supports Retrieval-Augmented Generation (RAG) using Groq LLMs to generate concise answers grounded strictly in retrieved content.

Features

Cross-Lingual Semantic Search

Search English queries over Hindi documents (and vice versa)

Powered by multilingual sentence embeddings

FAISS Vector Search

Efficient similarity search using cosine similarity

Sentence-Level Relevance

Retrieves the most relevant 2–3 sentences instead of large chunks

Retrieval-Augmented Generation (RAG)

Optional answer generation using Groq LLMs

Answers are grounded strictly in retrieved context

Language-Aware Answers

RAG answers are generated in the query’s language

User-Friendly Interface

Paste text, upload .txt files, or load sample documents

Deployment-Ready

Works locally and on Streamlit Cloud

How It Works

Document Ingestion
Documents are split into manageable chunks.

Embedding
Each chunk is converted into a multilingual vector representation.

Vector Indexing
FAISS is used with cosine similarity (inner product).

Semantic Search
Queries are embedded and matched against document vectors.

Sentence Re-Ranking
The most relevant sentences are extracted from matched chunks.

Retrieval-Augmented Generation (Optional)
Retrieved sentences are passed to a Groq LLM to generate a grounded answer.

Tech Stack

Python

Streamlit – Interactive UI

SentenceTransformers – Multilingual embeddings

FAISS (CPU) – Vector similarity search

Groq API – Fast LLM inference for RAG
