import os
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="Multilingual Semantic Search",
    layout="centered"
)

SIMILARITY_THRESHOLD = 0.25

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_AVAILABLE = GROQ_API_KEY is not None

client = Groq(api_key=GROQ_API_KEY) if GROQ_AVAILABLE else None

SAMPLE_DOCUMENT = """
गोरिल्ला बड़े और मुख्य रूप से शाकाहारी महान कपि होते हैं जो भूमध्यरेखीय अफ्रीका के उष्णकटिबंधीय जंगलों में रहते हैं।
गोरिल्ला वंश को पूर्वी और पश्चिमी प्रजातियों में विभाजित किया गया है।

गोरिल्ला वर्तमान में जीवित सबसे बड़े प्राइमेट होते हैं और आमतौर पर समूहों में रहते हैं, जिनका नेतृत्व एक सिल्वरबैक करता है।
वे अपनी बुद्धिमत्ता और मजबूत सामाजिक संबंधों के लिए जाने जाते हैं।

गोरिल्ला उप-सहारा अफ्रीका के विभिन्न ऊँचाई वाले क्षेत्रों में पाए जाते हैं।
आवास विनाश और अवैध शिकार के कारण गोरिल्ला की कई प्रजातियाँ संकटग्रस्त हैं।
"""

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, text_chunks

def best_sentences_from_chunk(chunk_text, query, top_n=3):
    sentences = [s.strip() for s in chunk_text.split(".") if len(s.strip()) > 5]
    if not sentences:
        return chunk_text

    sent_emb = model.encode(sentences)
    query_emb = model.encode([query])

    sent_emb = sent_emb / np.linalg.norm(sent_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.linalg.norm(query_emb)

    sims = (sent_emb @ query_emb.T).flatten()
    top_idx = sorted(sims.argsort()[::-1][:top_n])

    return " ".join(sentences[i] + "." for i in top_idx)

def groq_rag_answer(query, retrieved_sentences):
    context = "\n".join(f"- {s}" for s in retrieved_sentences)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a question answering assistant.\n"
                    "Follow these steps strictly:\n"
                    "1. From the provided context, identify the relevant information that answers the question.\n"
                    "2. Combine that information into a SHORT PARAGRAPH of 2–3 sentences.\n"
                    "3. Translate the paragraph into ENGLISH if needed.\n\n"
                    "Rules:\n"
                    "- The final answer MUST be in ENGLISH.\n"
                    "- The answer MUST contain at least TWO sentences.\n"
                    "- Do NOT copy the context verbatim if it is not English.\n"
                    "- Use ONLY information from the context.\n"
                    "- If the answer is not present, say: The provided documents do not contain this information."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()






st.title("🌐 Multilingual Semantic Search")
st.write(
    "Upload or paste documents and search them by meaning, "
    "not keywords. Works across languages."
)

doc_mode = st.radio("Document Input", ["Paste text", "Upload .txt file"])

show_sample = st.checkbox("Show sample document")

if show_sample:
    st.text_area("Sample document", SAMPLE_DOCUMENT, height=200, disabled=True)

if st.button("Load sample document"):
    chunks = chunk_text(SAMPLE_DOCUMENT)
    index, chunks = build_faiss_index(chunks)
    st.session_state.dynamic_index = index
    st.session_state.dynamic_chunks = chunks
    st.success("Sample document loaded")

query = st.text_input("Enter your query", placeholder="e.g. What is a gorilla?")

if GROQ_AVAILABLE:
    use_rag = st.checkbox("Answer using retrieved context (RAG)")
else:
    use_rag = False
    st.info("RAG is disabled because GROQ_API_KEY is not set")

top_k = st.slider("Number of results", 1, 5, 3)

if "dynamic_index" not in st.session_state:
    st.session_state.dynamic_index = None
    st.session_state.dynamic_chunks = None

if doc_mode == "Paste text":
    user_text = st.text_area("Paste your document text", height=200)
    if st.button("Index text") and user_text.strip():
        chunks = chunk_text(user_text)
        index, chunks = build_faiss_index(chunks)
        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks
        st.success(f"Indexed {len(chunks)} chunks")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file and st.button("Index document"):
        text = uploaded_file.read().decode("utf-8")
        chunks = chunk_text(text)
        index, chunks = build_faiss_index(chunks)
        st.session_state.dynamic_index = index
        st.session_state.dynamic_chunks = chunks
        st.success(f"Indexed {len(chunks)} chunks")

if st.button("Search") and query.strip():
    if st.session_state.dynamic_index is None:
        st.warning("Please paste, upload, or load the sample document first")
    else:
        query_emb = model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb)

        k = min(top_k, st.session_state.dynamic_index.ntotal)
        scores, indices = st.session_state.dynamic_index.search(query_emb, k)

        results = []
        for r, idx in enumerate(indices[0]):
            score = scores[0][r]
            if score >= SIMILARITY_THRESHOLD:
                txt = best_sentences_from_chunk(
                    st.session_state.dynamic_chunks[idx],
                    query,
                    top_n=3
                )
                results.append((txt, score))

        if not results:
            st.warning("No relevant results found")
            st.stop()

        if use_rag and GROQ_AVAILABLE:
            st.subheader("Answer")
            with st.spinner("Generating answer"):
                st.write(groq_rag_answer(query, [t for t, _ in results]))

        st.subheader("Results")
        for i, (text, score) in enumerate(results, 1):
            st.markdown(f"{i}.")
            st.write(text)
            st.caption(f"Similarity: {score:.3f}")
