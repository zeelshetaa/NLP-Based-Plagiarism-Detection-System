from sklearn.metrics.pairwise import cosine_similarity

def compute_semantic_similarity(uploaded_embedding, reference_embeddings):
    """
    Compute cosine similarity between uploaded document
    and reference document embeddings
    """

    similarities = cosine_similarity(uploaded_embedding, reference_embeddings)

    return similarities[0]