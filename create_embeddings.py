from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("dataset/web_data.csv")
docs = df['text'].dropna().tolist()[:500]   # limit for speed

embeddings = model.encode(docs)

np.save("dataset/web_embeddings.npy", embeddings)

print("✅ Embeddings saved!")

