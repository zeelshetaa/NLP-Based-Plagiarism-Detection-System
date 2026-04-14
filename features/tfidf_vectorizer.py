"""
features/tfidf_vectorizer.py
-----------------------------
Wraps sklearn's TfidfVectorizer to provide fit/transform utilities.
The vectorizer is stored as a module-level singleton so that
fit_vectorizer() and transform_documents() share the same vocabulary.

Future upgrade path:
    Replace TF-IDF vectors with sentence-transformer / BERT embeddings
    by keeping the same fit_vectorizer / transform_documents interface.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

# Module-level singleton vectorizer instance
_vectorizer: TfidfVectorizer = None


def fit_vectorizer(reference_docs: list[str]) -> TfidfVectorizer:
    """
    Fit TF-IDF vectorizer on the reference corpus.

    Builds the vocabulary and IDF weights from the provided documents.
    Must be called before transform_documents().

    Args:
        reference_docs (list[str]): List of cleaned reference document texts.

    Returns:
        TfidfVectorizer: The fitted vectorizer instance.

    Raises:
        ValueError: If reference_docs is empty.
    """
    global _vectorizer

    if not reference_docs:
        raise ValueError("reference_docs must contain at least one document.")

    _vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),   # unigrams + bigrams for richer features
        min_df=1,
        sublinear_tf=True,    # apply 1 + log(tf) dampening
    )
    _vectorizer.fit(reference_docs)
    return _vectorizer


def transform_documents(documents: list[str]) -> scipy.sparse.csr_matrix:
    """
    Transform text documents into TF-IDF feature vectors.

    Must be called after fit_vectorizer().

    Args:
        documents (list[str]): List of cleaned text strings to vectorize.

    Returns:
        scipy.sparse.csr_matrix: Sparse TF-IDF matrix, shape (n_docs, vocab_size).

    Raises:
        RuntimeError: If the vectorizer has not been fitted yet.
        ValueError: If documents list is empty.
    """
    global _vectorizer

    if _vectorizer is None:
        raise RuntimeError(
            "Vectorizer is not fitted. Call fit_vectorizer() first."
        )

    if not documents:
        raise ValueError("documents list must not be empty.")

    return _vectorizer.transform(documents)
