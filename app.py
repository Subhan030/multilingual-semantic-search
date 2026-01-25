import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Multilingual Semantic Search", layout="centered")

# ------------------ Load resources (cached) ------------------
@st.cache_resource
def load_resources():
    embeddings = np.load("data/embeddings.npy")
    metadata = pd.read_csv("data/chunk_metadata.csv")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index, metadata

model, index, metadata = load_resources()

# ------------------ UI ------------------
st.title("🌐 Multilingual Semantic Search")
st.write("Search documents by **meaning**, not keywords. Works across languages (English ↔ Hindi).")

query = st.text_input("Enter your query (any language):", placeholder="e.g., employee benefits / कर्मचारी लाभ")
top_k = st.slider("Number of results", 1, 5, 3)

# ------------------ Search ------------------
def semantic_search(query, top_k):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for d, idx in zip(distances[0], indices[0]):
        results.append({
            "distance": float(d),
            "language": metadata.iloc[idx]["language"],
            "text": metadata.iloc[idx]["text"]
        })
    return results

# Simple threshold to handle unrelated queries
DISTANCE_THRESHOLD = 5

if st.button("Search") and query.strip():
    results = semantic_search(query, top_k)

    
    st.subheader("Results")
    for i, r in enumerate(results, 1):
        st.markdown(f"**{i}. ({r['language']})**")
        st.write(r["text"])
        st.caption(f"Distance: {r['distance']:.3f}")
