"""
Text Preprocessing Utilities
Cleaning, normalization, and format helpers for CNN/DailyMail
"""

import re
import unicodedata
from typing import List, Optional


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode to ASCII-compatible form."""
    return unicodedata.normalize("NFKC", text)


def clean_article(text: str) -> str:
    """
    Clean a raw CNN/DailyMail article.
    - Removes bylines (CNN, Reuters, etc.)
    - Strips boilerplate phrases
    - Normalizes whitespace
    """
    # Remove CNN/Reuters style bylines
    text = re.sub(
        r"^(CNN|Reuters|AP|AFP)\s*[-–—]\s*",
        "",
        text.strip(),
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Remove "(CNN)" markers
    text = re.sub(r"\(CNN\)\s*", "", text, flags=re.IGNORECASE)
    # Remove editor notes
    text = re.sub(
        r"Editor'?s?\s+note:.*?(?=\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove "(Watch|Read) (more|the|video)" patterns
    text = re.sub(r"\(Watch [^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Read [^)]+\)", "", text, flags=re.IGNORECASE)
    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Strip
    text = text.strip()
    return normalize_unicode(text)


def clean_summary(text: str) -> str:
    """
    Clean a CNN/DailyMail highlight/summary.
    - Strips bullet points and leading dashes
    - Handles multi-sentence summaries
    """
    # Remove bullet points and dashes
    text = re.sub(r"^[\-•\*]\s*", "", text.strip(), flags=re.MULTILINE)
    # Remove "NEW:" prefixes
    text = re.sub(r"\bNEW:\s*", "", text, flags=re.IGNORECASE)
    # Remove @author handles
    text = re.sub(r"@\w+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return normalize_unicode(text)


def truncate_to_word_limit(text: str, max_words: int) -> str:
    """Truncate text to max_words, respecting sentence boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Try to end at sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_words * 3 // 4:  # If period is in last 25%
        return truncated[: last_period + 1]
    return truncated + "..."


# ─────────────────────────────────────────────
# Dataset Preprocessing Function
# ─────────────────────────────────────────────

def preprocess_example(
    article: str,
    highlights: str,
    max_article_words: int = 800,
    max_summary_words: int = 150,
) -> dict:
    """
    Full preprocessing pipeline for one CNN/DailyMail example.
    Returns cleaned article and summary.
    """
    article_clean = clean_article(article)
    summary_clean = clean_summary(highlights)

    article_clean = truncate_to_word_limit(article_clean, max_article_words)
    summary_clean = truncate_to_word_limit(summary_clean, max_summary_words)

    return {
        "article": article_clean,
        "summary": summary_clean,
        "article_word_count": len(article_clean.split()),
        "summary_word_count": len(summary_clean.split()),
    }


def batch_preprocess(examples: dict) -> dict:
    """
    HuggingFace Dataset.map-compatible batch preprocessing.
    """
    results = {
        "article": [],
        "summary": [],
        "article_word_count": [],
        "summary_word_count": [],
    }
    for article, highlights in zip(examples["article"], examples["highlights"]):
        processed = preprocess_example(article, highlights)
        results["article"].append(processed["article"])
        results["summary"].append(processed["summary"])
        results["article_word_count"].append(processed["article_word_count"])
        results["summary_word_count"].append(processed["summary_word_count"])
    return results


# ─────────────────────────────────────────────
# Tokenization Quality Filters
# ─────────────────────────────────────────────

def filter_by_token_length(
    example: dict,
    tokenizer,
    min_tokens: int = 50,
    max_tokens: int = 1200,
) -> bool:
    """Filter examples by tokenized length of article."""
    tokens = tokenizer(
        example["article"],
        truncation=False,
        return_length=True,
    )["length"][0]
    return min_tokens <= tokens <= max_tokens


# ─────────────────────────────────────────────
# BPE Tokenization Stats
# ─────────────────────────────────────────────

def compute_tokenization_stats(
    texts: List[str],
    tokenizer,
    sample_size: int = 1000,
) -> dict:
    """
    Compute BPE tokenization statistics for a text corpus.
    """
    import numpy as np

    sample = texts[:sample_size]
    lengths = []

    for text in sample:
        tokens = tokenizer(text, truncation=False, return_length=True)
        lengths.append(tokens["length"][0])

    arr = np.array(lengths)
    return {
        "mean_tokens": float(arr.mean()),
        "median_tokens": float(np.median(arr)),
        "p95_tokens": float(np.percentile(arr, 95)),
        "p99_tokens": float(np.percentile(arr, 99)),
        "max_tokens": int(arr.max()),
        "min_tokens": int(arr.min()),
        "pct_under_1024": float((arr <= 1024).mean() * 100),
        "pct_under_512": float((arr <= 512).mean() * 100),
    }
