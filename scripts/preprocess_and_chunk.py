import pandas as pd
import re

CHUNK_SIZE = 150 
df = pd.read_csv("data/documents.csv")

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

chunks = []
chunk_id = 1

for _, row in df.iterrows():
    text = clean_text(row["text"])
    words = text.split()

    for i in range(0, len(words), CHUNK_SIZE):
        chunk_text = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": row["doc_id"],
            "language": row["language"],
            "text": chunk_text
        })
        chunk_id += 1

chunk_df = pd.DataFrame(chunks)
chunk_df.to_csv("data/chunks.csv", index=False, encoding="utf-8")

print("Total chunks created:", len(chunk_df))
print(chunk_df.head())
