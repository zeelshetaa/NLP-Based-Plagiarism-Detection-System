"""
results/report_generator.py
----------------------------
Generates a structured plagiarism report from analysis results.
Returns a Python dictionary — easy to serialise to JSON, CSV, or PDF.
"""


def generate_report(
    uploaded_filename: str,
    reference_filename: str,
    similarity_score: float,
    classification: dict,
) -> dict:
    """
    Build a structured plagiarism report.

    Args:
        uploaded_filename  (str):  Name / path of the uploaded document.
        reference_filename (str):  Name / path of the best-matching reference.
        similarity_score   (float): Cosine similarity score in [0.0, 1.0].
        classification     (dict):  Output from model.classifier.predict().
                                    Expected keys: label, verdict, probability.

    Returns:
        dict: Structured report with the following keys —
            - uploaded_document   : filename of the document under review
            - matched_document    : filename of the most similar reference
            - similarity_score    : raw cosine similarity (4 d.p.)
            - similarity_percent  : similarity expressed as a percentage string
            - plagiarism_verdict  : "PLAGIARISED" or "ORIGINAL"
            - confidence_percent  : model confidence as a percentage string
            - summary             : one-line human-readable summary
    """
    similarity_percent: float = round(similarity_score * 100, 2)
    confidence_percent: float = round(classification["probability"] * 100, 2)
    verdict: str = classification["verdict"]

    # Build a one-line natural language summary
    if verdict == "PLAGIARISED":
        summary = (
            f'"{uploaded_filename}" is {similarity_percent}% similar to '
            f'"{reference_filename}" and is classified as PLAGIARISED '
            f"(confidence: {confidence_percent}%)."
        )
    else:
        summary = (
            f'"{uploaded_filename}" is {similarity_percent}% similar to '
            f'"{reference_filename}" and is classified as ORIGINAL '
            f"(confidence: {100 - confidence_percent}% not plagiarised)."
        )

    report: dict = {
        "uploaded_document": uploaded_filename,
        "matched_document": reference_filename,
        "similarity_score": round(similarity_score, 4),
        "similarity_percent": f"{similarity_percent}%",
        "plagiarism_verdict": verdict,
        "confidence_percent": f"{confidence_percent}%",
        "summary": summary,
    }

    return report


def print_report(report: dict) -> None:
    """
    Pretty-print a plagiarism report to stdout.

    Args:
        report (dict): Structured report returned by generate_report().
    """
    border = "=" * 65
    print(f"\n{border}")
    print("          🔍  PLAGIARISM DETECTION REPORT")
    print(border)
    print(f"  Uploaded Document : {report['uploaded_document']}")
    print(f"  Matched Reference : {report['matched_document']}")
    print(f"  Similarity Score  : {report['similarity_score']}  "
          f"({report['similarity_percent']})")
    print(f"  Verdict           : {report['plagiarism_verdict']}")
    print(f"  Model Confidence  : {report['confidence_percent']}")
    print(f"\n  Summary:\n  {report['summary']}")
    print(f"{border}\n")
