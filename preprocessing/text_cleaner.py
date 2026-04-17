"""
preprocessing/text_cleaner.py
------------------------------
Handles loading raw text from files and cleaning/normalizing it.
Designed to be swappable — future upgrades (e.g., NLTK stemming,
BERT tokenization) only require changes here.
"""

import os
import re
import string


def load_text(file_path: str) -> str:
    """
    Load raw text content from a file.

    Args:
        file_path (str): Absolute or relative path to the text file.

    Returns:
        str: Raw text content of the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If the file cannot be read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def clean_text(text: str) -> str:
    """
    Normalize and clean raw text for NLP processing.

    Steps performed:
        1. Convert to lowercase.
        2. Remove punctuation characters.
        3. Collapse multiple whitespace/newlines into a single space.
        4. Strip leading/trailing whitespace.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned, normalized text string.
    """
    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3 — Remove extra whitespace (spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text)

    # Step 4 — Strip edges
    text = text.strip()

    return text
