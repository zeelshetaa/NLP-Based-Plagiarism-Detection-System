"""
main.py
--------
Entry point for the NLP-Based Intelligent Plagiarism Detection System.

Pipeline:
    1. Load reference documents from dataset/reference/
    2. Load the uploaded document from dataset/uploaded/
    3. Clean all texts via preprocessing.text_cleaner
    4. Fit TF-IDF vectorizer on reference corpus
    5. Transform both reference and uploaded documents
    6. Compute cosine similarity
    7. Train Logistic Regression classifier
    8. Classify the result
    9. Generate and print structured report
"""

import os
import sys

# ── Module imports ─────────────────────────────────────────────────────────────
from preprocessing.text_cleaner import load_text, clean_text
from features.tfidf_vectorizer import fit_vectorizer, transform_documents
from similarity.similarity_checker import compute_cosine_similarity
from model.classifier import train_model, predict
from results.report_generator import generate_report, print_report

# ── Path configuration ─────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR: str = os.path.join(BASE_DIR, "dataset", "reference")
UPLOADED_DIR: str = os.path.join(BASE_DIR, "dataset", "uploaded")

# BURT Embeddings
from features.bert_embeddings import BertEmbedder
from similarity.semantic_similarity import compute_semantic_similarity


def load_reference_documents() -> tuple[list[str], list[str]]:
    """
    Load and clean all .txt documents from the reference directory.

    Returns:
        tuple[list[str], list[str]]:
            - cleaned_docs  : list of cleaned text strings
            - filenames     : list of corresponding file names
    """
    if not os.path.isdir(REFERENCE_DIR):
        raise FileNotFoundError(
            f"Reference directory not found: {REFERENCE_DIR}"
        )

    txt_files = sorted(
        [f for f in os.listdir(REFERENCE_DIR) if f.endswith(".txt")]
    )

    if not txt_files:
        raise ValueError(
            f"No .txt files found in reference directory: {REFERENCE_DIR}"
        )

    cleaned_docs: list[str] = []
    filenames: list[str] = []

    for fname in txt_files:
        fpath = os.path.join(REFERENCE_DIR, fname)
        raw = load_text(fpath)
        cleaned = clean_text(raw)
        cleaned_docs.append(cleaned)
        filenames.append(fname)
        print(f"  [✓] Loaded reference: {fname}")

    return cleaned_docs, filenames


def load_uploaded_document() -> tuple[str, str]:
    """
    Load and clean the first .txt file found in the uploaded directory.

    Returns:
        tuple[str, str]:
            - cleaned_text : cleaned text string
            - filename     : name of the uploaded file
    """
    if not os.path.isdir(UPLOADED_DIR):
        raise FileNotFoundError(
            f"Uploaded directory not found: {UPLOADED_DIR}"
        )

    txt_files = sorted(
        [f for f in os.listdir(UPLOADED_DIR) if f.endswith(".txt")]
    )

    if not txt_files:
        raise ValueError(
            f"No .txt files found in uploaded directory: {UPLOADED_DIR}"
        )

    # Use the first file found (single-document pipeline)
    fname = txt_files[0]
    fpath = os.path.join(UPLOADED_DIR, fname)
    raw = load_text(fpath)
    cleaned = clean_text(raw)
    print(f"  [✓] Loaded uploaded:   {fname}")
    return cleaned, fname


def run_pipeline() -> None:
    """
    Execute the full plagiarism detection pipeline and print the report.
    """
    print("\n" + "=" * 65)
    print("   NLP-Based Intelligent Plagiarism Detection System")
    print("=" * 65)

    # ── Step 1 & 3: Load + clean reference docs ────────────────────────────
    print("\n[1/6] Loading reference documents …")
    ref_docs, ref_filenames = load_reference_documents()

    # ── Step 2 & 3: Load + clean uploaded doc ─────────────────────────────
    print("\n[2/6] Loading uploaded document …")
    uploaded_text, uploaded_filename = load_uploaded_document()

    # ── Step 4: BERT feature extraction ─────────────────────────────────
    print("[3/6] Loading BERT embedder...")
    bert = BertEmbedder()

    print("[4/6] Generating embeddings...")
    reference_embeddings = bert.encode_documents(ref_docs)
    uploaded_embedding = bert.encode_single(uploaded_text)

    # ── Step 5: Cosine similarity ──────────────────────────────────────────
    print("[5/6] Computing semantic similarity...")
    similarity_scores = compute_semantic_similarity(
        uploaded_embedding,
        reference_embeddings
    )
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[best_match_index]
    print("Best Match:", ref_filenames[best_match_index])
    print("Similarity Score:", best_score)
    # ── Step 6 & 7: Train classifier + classify ────────────────────────────
    print("\n[6/6] Classifying result …")
    train_model(threshold=0.5)
    classification = predict(best_score)

    # ── Step 7: Generate & print report ───────────────────────────────────
    report = generate_report(
        uploaded_filename=uploaded_filename,
        reference_filename=ref_filenames[best_match_index],
        similarity_score=best_score,
        classification=classification,
    )
    print_report(report)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
