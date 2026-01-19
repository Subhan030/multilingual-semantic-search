import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/chunks.csv")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

texts = df["text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True
)

np.save("data/embeddings.npy", embeddings)

df.to_csv("data/chunk_metadata.csv", index=False, encoding="utf-8")

print("Embeddings created!")
print("Embedding shape:", embeddings.shape)
