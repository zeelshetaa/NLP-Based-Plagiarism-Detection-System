import pandas as pd
from preprocessing.preprocess import clean_text

df = pd.read_csv("dataset/reference/train.csv")

# ✅ REMOVE NULL VALUES FIRST
df = df[['question1', 'question2']].dropna()

# OPTIONAL (extra safe)
df = df[df['question1'].astype(str).str.strip() != ""]
df = df[df['question2'].astype(str).str.strip() != ""]

# cleaning
df['q1_clean'] = df['question1'].apply(clean_text)
df['q2_clean'] = df['question2'].apply(clean_text)

# save
df.to_csv("dataset/cleaned_train.csv", index=False)

print("✅ Cleaned dataset saved")