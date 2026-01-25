import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Multilingual Semantic Search",
    layout="centered"
)

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

def chunk_text(text, chunk_size=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, text_chunks

st.title("🌐 Multilingual Semantic Search")
st.write(
    "Search documents by **meaning**, not keywords. "
    "Works across languages (English ↔ Hindi)."
)

st.subheader("📄 Document Source")

doc_mode = st.radio(
    "Choose how to provide documents:",
    ["Use existing dataset", "Paste text", "Upload .txt file"]
)

query = st.text_input(
    "Enter your query:",
    placeholder="e.g. employee benefits / कर्मचारी लाभ"
)

top_k = st.slider("Number of results", 1, 5, 3)

if "dynamic_index" not in st.session_state:
    st.session_state.dynamic_index = None
    st.session_state.dynamic_chunks = None

if doc_mode == "Paste text":
    user_text = st.text_area(
        "Paste your document text here:",
        height=200
    )

    if st.button("Index pasted text") and user_text.strip():
        chunks = chunk_text(user_text)
        index, chunks = build_faiss_index(chunks)

        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks from pasted text.")
elif doc_mode == "Upload .txt file":
    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"]
    )

    if uploaded_file and st.button("Index uploaded document"):
        text = uploaded_file.read().decode("utf-8")
        chunks = chunk_text(text)
        index, chunks = build_faiss_index(chunks)

        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks from uploaded document.")

if st.button("Search") and query.strip():

    query_embedding = model.encode([query])
    if doc_mode == "Use existing dataset":
        distances, indices = static_index.search(query_embedding, top_k)

        st.subheader("Results (Dataset)")
        for i, idx in enumerate(indices[0], 1):
            st.markdown(f"**{i}. ({metadata.iloc[idx]['language']})**")
            st.write(metadata.iloc[idx]["text"])
            st.caption(f"Distance: {distances[0][i-1]:.3f}")

    else:
        if st.session_state.dynamic_index is None:
            st.warning("Please index a document first.")
        else:
            distances, indices = st.session_state.dynamic_index.search(
                query_embedding, top_k
            )

            st.subheader("Results (Your Document)")
            for i, idx in enumerate(indices[0], 1):
                st.markdown(f"**{i}.**")
                st.write(st.session_state.dynamic_chunks[idx])
                st.caption(f"Distance: {distances[0][i-1]:.3f}")
