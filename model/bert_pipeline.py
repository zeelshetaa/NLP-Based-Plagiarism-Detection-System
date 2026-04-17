import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preprocess import clean_text

# Load dataset
df = pd.read_csv("../dataset/data.csv")

# Apply preprocessing
df["s1_clean"] = df["sentence1"].apply(clean_text)
df["s2_clean"] = df["sentence2"].apply(clean_text)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')


def compare_documents(doc1, doc2):
    doc1_emb = model.encode([clean_text(doc1)])
    doc2_emb = model.encode([clean_text(doc2)])
    
    score = cosine_similarity(doc1_emb, doc2_emb)[0][0]
    return score

    
# Generate embeddings
emb1 = model.encode(df["s1_clean"].tolist(), convert_to_tensor=False)
emb2 = model.encode(df["s2_clean"].tolist(), convert_to_tensor=False)

# Compute similarity
similarities = cosine_similarity(emb1, emb2).diagonal()

df["similarity_score"] = similarities

# Advanced classification logic
def classify(score):
    if score > 0.80:
        return "High Plagiarism"
    elif score > 0.60:
        return "Moderate Similarity"
    else:
        return "Low / No Plagiarism"

df["result"] = df["similarity_score"].apply(classify)

# Save results
df.to_csv("../results/output.csv", index=False)

print(df[["sentence1", "sentence2", "similarity_score", "result"]])



if __name__ == "__main__":
    doc1 = "Machine learning is very useful in artificial intelligence."
    doc2 = "AI uses machine learning techniques for many applications."

    score = compare_documents(doc1, doc2)

    print("Similarity Score:", score)