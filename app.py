import numpy as np
import faiss
import streamlit as st
import subprocess
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Multilingual Semantic Search",
    layout="centered"
)

DISTANCE_THRESHOLD = 37.0

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


model = load_model()

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
def build_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, text_chunks

def best_sentences_from_chunk(chunk_text, query, top_n=3):
    sentences = [s.strip() for s in chunk_text.split(".") if len(s.strip()) > 5]

    if not sentences:
        return chunk_text

    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])

    sentence_embeddings = sentence_embeddings / np.linalg.norm(
        sentence_embeddings, axis=1, keepdims=True
    )
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    similarities = (sentence_embeddings @ query_embedding.T).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    top_indices = sorted(top_indices)

    return " ".join(sentences[i] + "." for i in top_indices)


def free_rag_answer(query, retrieved_sentences, model_name="phi"):
    context = "\n".join(f"- {s}" for s in retrieved_sentences)

    prompt = f"""
You are an assistant that answers ONLY using the provided context.
If the answer is not present, say:
"The provided documents do not contain this information."

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )

        if result.stdout.strip():
            return result.stdout.strip()

        if result.stderr.strip():
            return f"⚠️ Ollama error:\n{result.stderr.strip()}"

        return "⚠️ Ollama returned no output."

    except subprocess.TimeoutExpired:
        return "⚠️ Ollama took too long to respond."

st.title("🌐 Multilingual Semantic Search")
st.write(
    "Upload or paste documents and search them by **meaning**, "
    "not keywords. Works across languages."
)

doc_mode = st.radio(
    "📄 Document Input",
    ["Paste text", "Upload .txt file"]
)

query = st.text_input(
    "Enter your query:",
    placeholder="e.g. What is a gorilla?"
)

use_rag = st.checkbox(" Answer using retrieved context (RAG)")
top_k = st.slider("Number of results", 1, 5, 3)


if "dynamic_index" not in st.session_state:
    st.session_state.dynamic_index = None
    st.session_state.dynamic_chunks = None

if doc_mode == "Paste text":
    user_text = st.text_area("Paste your document text:", height=200)

    if st.button("Index text") and user_text.strip():
        chunks = chunk_text(user_text)
        index, chunks = build_faiss_index(chunks)

        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks.")


elif doc_mode == "Upload .txt file":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file and st.button("Index document"):
        text = uploaded_file.read().decode("utf-8")
        chunks = chunk_text(text)
        index, chunks = build_faiss_index(chunks)

        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks.")

if st.button("Search") and query.strip():

    if st.session_state.dynamic_index is None:
        st.warning("Please paste or upload a document first.")
    else:
        query_embedding = model.encode([query])
        k = min(top_k, st.session_state.dynamic_index.ntotal)

        distances, indices = st.session_state.dynamic_index.search(
            query_embedding, k
        )

        filtered_results = []

        for rank, idx in enumerate(indices[0]):
            distance = distances[0][rank]

            if distance <= DISTANCE_THRESHOLD:
                text = best_sentences_from_chunk(
                    st.session_state.dynamic_chunks[idx],
                    query,
                    top_n=3
                )
                filtered_results.append((text, distance))

        if not filtered_results:
            st.warning("No relevant results found. Try a different query.")
            st.stop()

        if use_rag:
            st.subheader(" Answer (RAG – Local, Free)")
            with st.spinner("Generating answer locally..."):
                st.write(
                    free_rag_answer(
                        query,
                        [text for text, _ in filtered_results]
                    )
                )

        st.subheader("Results")
        for i, (text, distance) in enumerate(filtered_results, 1):
            st.markdown(f"**{i}.**")
            st.write(text)
            st.caption(f"Distance: {distance:.3f}")
