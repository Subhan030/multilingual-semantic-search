from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    use_rag: bool = False

class SearchResult(BaseModel):
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    answer: Optional[str] = None

class IndexTextRequest(BaseModel):
    text: str
