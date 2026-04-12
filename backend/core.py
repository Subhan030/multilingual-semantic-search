import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

SIMILARITY_THRESHOLD = 0.20

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_AVAILABLE = GROQ_API_KEY is not None

client = Groq(api_key=GROQ_API_KEY) if GROQ_AVAILABLE else None

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

class EngineState:
    def __init__(self):
        self.index = None
        self.chunks = None
        
state = EngineState()

def chunk_text(text, chunk_size=70, overlap=15):
    words = text.split()
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

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
    if not client:
        return "Groq client is not initialized. Please provide GROQ_API_KEY."
        
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
                    "2. Combine that information into a SHORT PARAGRAPH.\n"
                    "3. Translate the paragraph into ENGLISH if needed.\n"
                    "4. Be descriptive and provide a detailed answer.\n\n"
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
