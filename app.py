# =======================
# Imports
# =======================
import os
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import subprocess


# =======================
# Page config
# =======================
st.set_page_config(
    page_title="Multilingual Semantic Search",
    layout="centered"
)


# =======================
# Helper: best sentence from chunk
# =======================
def best_sentence_from_chunk(chunk_text, query, model):
    sentences = [s.strip() for s in chunk_text.split(".") if len(s.strip()) > 5]

    if not sentences:
        return chunk_text

    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])

    sentence_embeddings = sentence_embeddings / np.linalg.norm(
        sentence_embeddings, axis=1, keepdims=True
    )
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    similarities = sentence_embeddings @ query_embedding.T
    best_idx = int(np.argmax(similarities))

    return sentences[best_idx] + "."


# =======================
# Load static resources
# =======================
@st.cache_resource
def load_resources():
    embeddings = np.load("data/embeddings.npy")
    metadata = pd.read_csv("data/chunk_metadata.csv")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index, metadata


model, static_index, metadata = load_resources()


# =======================
# Chunking & indexing helpers
# =======================
def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, text_chunks


# =======================
# RAG (optional)
# =======================
# =======================
# FREE RAG using Ollama (local LLM)
# =======================
def free_rag_answer(query, retrieved_sentences, model_name="phi"):
    """
    Uses a local Ollama model to generate an answer
    strictly from retrieved sentences.
    """
    context = "\n".join(f"- {s}" for s in retrieved_sentences)

    prompt = f"""
You are an assistant that answers ONLY using the provided context.
If the answer is not present, say:
"The provided documents do not contain this information."

Context:
{context}

Question:
{query}

Answer (concise, factual):
"""

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()



# =======================
# UI
# =======================
st.title("🌐 Multilingual Semantic Search")
st.write(
    "Search documents by **meaning**, not keywords. "
    "Works across languages (English ↔ Hindi)."
)

doc_mode = st.radio(
    "📄 Document Source",
    ["Paste text", "Upload .txt file"]
)

query = st.text_input(
    "Enter your query:",
    placeholder="e.g. employee benefits / कर्मचारी लाभ"
)

use_rag = st.checkbox("🧠 Answer using retrieved context (RAG)")
top_k = st.slider("Number of results", 1, 5, 3)


# =======================
# Session state
# =======================
if "dynamic_index" not in st.session_state:
    st.session_state.dynamic_index = None
    st.session_state.dynamic_chunks = None


# =======================
# Document input
# =======================
if doc_mode == "Paste text":
    user_text = st.text_area("Paste your document text:", height=200)

    if st.button("Index pasted text") and user_text.strip():
        chunks = chunk_text(user_text)
        index, chunks = build_faiss_index(chunks)

        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks.")


elif doc_mode == "Upload .txt file":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file and st.button("Index uploaded document"):
        text = uploaded_file.read().decode("utf-8")
        chunks = chunk_text(text)
        index, chunks = build_faiss_index(chunks)

        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks.")


# =======================
# Search
# =======================
if st.button("Search") and query.strip():

    query_embedding = model.encode([query])

    # ---- Dynamic documents ----
    if st.session_state.dynamic_index is None:
        st.warning("Please index a document first.")
    else:
        k = min(top_k, st.session_state.dynamic_index.ntotal)
        distances, indices = st.session_state.dynamic_index.search(
            query_embedding, k
        )

        retrieved_sentences = [
            best_sentence_from_chunk(
                st.session_state.dynamic_chunks[idx],
                query,
                model
            )
            for idx in indices[0]
        ]

        if use_rag:
            st.subheader("🧠 Answer (RAG – Local, Free)")
            with st.spinner("Generating answer locally..."):
                st.write(free_rag_answer(query, retrieved_sentences))


        st.subheader("Results (Your Document)")
        for i, sentence in enumerate(retrieved_sentences, 1):
            st.markdown(f"**{i}.**")
            st.write(sentence)
            st.caption(f"Distance: {distances[0][i-1]:.3f}")
