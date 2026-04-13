from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/paws_sample.csv")

# Load pre-trained BERT model (Sentence-BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert sentences to embeddings
embeddings1 = model.encode(df["sentence1"].tolist())
embeddings2 = model.encode(df["sentence2"].tolist())

# Compute similarity
similarity_scores = cosine_similarity(embeddings1, embeddings2).diagonal()

# Add results
df["bert_similarity"] = similarity_scores

print(df[["sentence1", "sentence2", "bert_similarity"]])