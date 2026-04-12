from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.models import SearchRequest, SearchResponse, SearchResult, IndexTextRequest
from backend.core import (
    chunk_text, build_faiss_index, groq_rag_answer, 
    model, state, GROQ_AVAILABLE, SIMILARITY_THRESHOLD
)
import numpy as np

app = FastAPI(title="Multilingual Semantic Search API")

@app.post("/index/text")
def index_text(request: IndexTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    chunks = chunk_text(request.text)
    index, chunks = build_faiss_index(chunks)
    state.index = index
    state.chunks = chunks
    return {"message": f"Indexed {len(chunks)} chunks"}

@app.post("/index/file")
async def index_file(file: UploadFile = File(...)):
    text = await file.read()
    try:
        text = text.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="File cannot be empty")
        
    chunks = chunk_text(text)
    index, chunks = build_faiss_index(chunks)
    state.index = index
    state.chunks = chunks
    return {"message": f"Indexed {len(chunks)} chunks"}

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if state.index is None or state.chunks is None:
        raise HTTPException(status_code=400, detail="Please index a document first")
        
    query_emb = model.encode([request.query])
    query_emb = query_emb / np.linalg.norm(query_emb)

    k = min(request.top_k, state.index.ntotal)
    scores, indices = state.index.search(query_emb, k)

    results = []
    for r, idx in enumerate(indices[0]):
        score = float(scores[0][r])
        if score >= SIMILARITY_THRESHOLD:
            txt = state.chunks[idx]
            results.append(SearchResult(text=txt, score=score))

    if not results:
        return SearchResponse(results=[])

    answer = None
    if request.use_rag and GROQ_AVAILABLE:
        answer = groq_rag_answer(request.query, [r.text for r in results])

    return SearchResponse(results=results, answer=answer)
