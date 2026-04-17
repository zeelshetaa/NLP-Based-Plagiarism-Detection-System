import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("dataset/cleaned_train.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

q1_emb = model.encode(df['q1_clean'].tolist()[:500])
q2_emb = model.encode(df['q2_clean'].tolist()[:500])

np.save("dataset/q1_emb.npy", q1_emb)
np.save("dataset/q2_emb.npy", q2_emb)

print("Embeddings saved ✅")