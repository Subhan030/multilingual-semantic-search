# Multilingual Semantic Search with Cross-Lingual RAG

This project implements a multilingual semantic search system that enables cross-lingual information retrieval using dense vector embeddings, FAISS similarity search, and Retrieval-Augmented Generation (RAG).

The application allows users to query documents by semantic meaning rather than keyword matching, supporting English and Hindi cross-language search. It also provides an optional RAG pipeline powered by Groq large language models, generating answers strictly grounded in retrieved content.

## Core Features

Multilingual semantic search with cross-lingual retrieval (English ↔ Hindi)

Dense vector embeddings using SentenceTransformers

FAISS vector indexing with cosine similarity for fast nearest-neighbor search

Sentence-level relevance extraction to reduce noisy results

Optional Retrieval-Augmented Generation (RAG) using Groq API

Language-aware answer generation based on query language

Interactive web interface built with Streamlit

Deployment-ready architecture with secure API key handling

## Technical Overview

The system follows a modern vector-based information retrieval architecture:

Document ingestion and chunking

Multilingual embedding generation

Vector indexing using FAISS

Semantic similarity search with cosine similarity

Sentence-level re-ranking

Optional RAG-based answer synthesis

Technologies Used

Python

Streamlit

SentenceTransformers

FAISS (CPU)

Groq API
