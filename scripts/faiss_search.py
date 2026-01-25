import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


embeddings = np.load("data/embeddings.npy")
metadata = pd.read_csv("data/chunk_metadata.csv")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built with", index.ntotal, "vectors")

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "chunk_id": metadata.iloc[idx]["chunk_id"],
            "language": metadata.iloc[idx]["language"],
            "text": metadata.iloc[idx]["text"]
        })
    return results

query = "Social Security"
results = semantic_search(query)

print("\nQuery:", query)
print("Results:")
for r in results:
    print("-", r["text"])
