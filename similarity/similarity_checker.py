"""
similarity/similarity_checker.py
----------------------------------
Computes cosine similarity between an uploaded document vector
and a set of reference document vectors.

Cosine similarity measures the angle between two vectors in TF-IDF
space — documents with similar term distributions score close to 1.0,
while unrelated documents score close to 0.0.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse


def compute_cosine_similarity(
    uploaded_vector: scipy.sparse.csr_matrix,
    reference_vectors: scipy.sparse.csr_matrix,
) -> tuple[float, int]:
    """
    Compute cosine similarity between an uploaded document and all
    reference documents, then return the best match.

    Args:
        uploaded_vector  (csr_matrix): TF-IDF vector of the uploaded
                                       document.  Shape: (1, vocab_size).
        reference_vectors (csr_matrix): TF-IDF matrix of all reference
                                        documents.
                                        Shape: (n_refs, vocab_size).

    Returns:
        tuple[float, int]:
            - highest_score (float): Best cosine similarity score in [0, 1].
            - best_match_index (int): Row index into reference_vectors of
                                      the most similar document.

    Raises:
        ValueError: If either input matrix is empty.
    """
    if uploaded_vector.shape[0] == 0 or reference_vectors.shape[0] == 0:
        raise ValueError("Input matrices must not be empty.")

    # similarity_scores has shape (1, n_refs)
    similarity_scores: np.ndarray = cosine_similarity(
        uploaded_vector, reference_vectors
    )

    # Flatten to 1-D for easy argmax
    scores_flat: np.ndarray = similarity_scores.flatten()

    best_match_index: int = int(np.argmax(scores_flat))
    highest_score: float = float(scores_flat[best_match_index])

    return highest_score, best_match_index
