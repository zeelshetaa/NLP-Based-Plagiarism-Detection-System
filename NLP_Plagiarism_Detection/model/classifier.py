"""
model/classifier.py
--------------------
Logistic Regression–based plagiarism classifier.

The model is trained on synthetic (score, label) pairs and then used
to classify new similarity scores as PLAGIARISED or ORIGINAL.

Design note:
    Logistic Regression operates on a single numeric feature (the
    cosine similarity score). This keeps the pipeline simple and
    interpretable while still providing probabilistic output.

Future upgrade path:
    Replace the single-feature input with a richer feature vector
    (e.g., BERT similarity + TF-IDF score + n-gram overlap) without
    changing the train_model / predict interface.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

# Module-level classifier singleton
_model: LogisticRegression = None

# Default threshold — scores >= THRESHOLD are considered plagiarised
PLAGIARISM_THRESHOLD: float = 0.5


def train_model(threshold: float = PLAGIARISM_THRESHOLD) -> LogisticRegression:
    """
    Train a Logistic Regression classifier on synthetic similarity data.

    Synthetic training data spans the full [0.0, 1.0] similarity range:
        - Scores < threshold  → label 0 (ORIGINAL)
        - Scores >= threshold → label 1 (PLAGIARISED)

    Args:
        threshold (float): Decision boundary. Defaults to 0.5.

    Returns:
        LogisticRegression: The fitted classifier instance.
    """
    global _model

    # Generate evenly spaced similarity scores in [0.0, 1.0]
    scores: np.ndarray = np.linspace(0.0, 1.0, num=200).reshape(-1, 1)

    # Binary labels based on threshold
    labels: np.ndarray = (scores.flatten() >= threshold).astype(int)

    _model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    _model.fit(scores, labels)

    return _model


def predict(similarity_score: float) -> dict:
    """
    Classify a similarity score as plagiarised or original.

    Args:
        similarity_score (float): Cosine similarity score in [0.0, 1.0].

    Returns:
        dict: {
            "label"       : int   — 1 (PLAGIARISED) or 0 (ORIGINAL),
            "verdict"     : str   — human-readable verdict,
            "probability" : float — probability of being plagiarised,
        }

    Raises:
        RuntimeError: If the model has not been trained (train_model not called).
        ValueError: If similarity_score is outside [0.0, 1.0].
    """
    global _model

    if _model is None:
        raise RuntimeError(
            "Classifier not trained. Call train_model() first."
        )

    if not (0.0 <= similarity_score <= 1.0):
        raise ValueError(
            f"similarity_score must be in [0.0, 1.0], got {similarity_score}"
        )

    feature: np.ndarray = np.array([[similarity_score]])

    label: int = int(_model.predict(feature)[0])
    probability: float = float(_model.predict_proba(feature)[0][1])

    verdict: str = "PLAGIARISED" if label == 1 else "ORIGINAL"

    return {
        "label": label,
        "verdict": verdict,
        "probability": round(probability, 4),
    }
