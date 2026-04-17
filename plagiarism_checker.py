from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

def check_single_doc(doc, model, df, q1_emb, q2_emb):

    sentences = sent_tokenize(doc)

    total_score = 0
    matches = []

    for sent in sentences:
        emb = model.encode([sent])

        sim1 = cosine_similarity(emb, q1_emb)[0]
        sim2 = cosine_similarity(emb, q2_emb)[0]

        max_score = max(max(sim1), max(sim2))
        total_score += max_score

        if max_score > 0.75:
            matches.append((sent, max_score))

    final_score = total_score / len(sentences)
    percent = final_score * 100
    plag_count = 0

    for sent in sentences:
        emb = model.encode([sent])

        sim1 = cosine_similarity(emb, q1_emb)[0]
        sim2 = cosine_similarity(emb, q2_emb)[0]

        max_score = max(max(sim1), max(sim2))

        if max_score > 0.60:
            plag_count += 1
            matches.append((sent, max_score))

    if len(sentences) == 0:
        return 0, []

    percent = (plag_count / len(sentences)) * 100

    return percent, matches