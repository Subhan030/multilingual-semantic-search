import pandas as pd

df = pd.read_csv("data/documents.csv")

print("Documents loaded:", len(df))
print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nSample rows:")
print(df.head())