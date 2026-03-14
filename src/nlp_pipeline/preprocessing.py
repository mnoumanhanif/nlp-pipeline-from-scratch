"""
Text preprocessing utilities for the NLP pipeline.

Includes web scraping, tokenization, regex extraction, text normalization,
stopword removal, POS tagging, lemmatization/stemming, and NER.
"""

import re
import string
import urllib.request
from collections import Counter, defaultdict

from bs4 import BeautifulSoup


def fetch_clean_text(url: str) -> str:
    """
    Downloads HTML from a URL and extracts plain text.

    Args:
        url: The URL to fetch and clean.

    Returns:
        Cleaned plain text extracted from the HTML.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        html_bytes = response.read()
        html = html_bytes.decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text(separator=" ", strip=True)
    return plain_text


def extract_emails(text: str) -> list:
    """
    Extracts unique email addresses from text using regex.

    Args:
        text: Input text to search.

    Returns:
        List of unique email addresses found.
    """
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(email_pattern, text)))


def extract_phones(text: str) -> list:
    """
    Extracts unique US phone numbers from text using regex.

    Args:
        text: Input text to search.

    Returns:
        List of unique phone numbers found.
    """
    phone_pattern = (
        r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.\s]?\d{4}\b"
    )
    return list(set(re.findall(phone_pattern, text)))


def normalize_text(text: str) -> str:
    """
    Applies a text normalization pipeline:
    1. Lowercasing
    2. Replace digits with <NUM>
    3. Remove punctuation
    4. Remove extra whitespace

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text.
    """
    norm_text = text.lower()
    norm_text = re.sub(r"\d+", "<NUM>", norm_text)
    norm_text = norm_text.translate(str.maketrans("", "", string.punctuation))
    norm_text = re.sub(r"\s+", " ", norm_text).strip()
    return norm_text


def compute_corpus_statistics(tokens: list, stop_words: set = None) -> dict:
    """
    Computes corpus statistics before and after optional stopword removal.

    Args:
        tokens: List of tokens.
        stop_words: Optional set of stopwords to remove.

    Returns:
        Dictionary with token counts, vocabulary size, TTR, and top-25 frequencies.
    """
    total_tokens = len(tokens)
    vocab = set(tokens)
    vocab_size = len(vocab)
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0
    freq = Counter(tokens)
    top_25 = freq.most_common(25)

    result = {
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "ttr": ttr,
        "top_25": top_25,
    }

    if stop_words is not None:
        filtered = [w for w in tokens if w not in stop_words]
        filtered_vocab = set(filtered)
        result["filtered_tokens"] = len(filtered)
        result["filtered_vocab_size"] = len(filtered_vocab)
        result["filtered_ttr"] = (
            len(filtered_vocab) / len(filtered) if filtered else 0
        )
        result["filtered_top_25"] = Counter(filtered).most_common(25)

    return result


def extract_named_entities(tokens: list) -> dict:
    """
    Performs Named Entity Recognition on tokens using NLTK.

    Args:
        tokens: List of word tokens (case-sensitive for better NER).

    Returns:
        Dictionary mapping entity types to lists of unique entity names.
    """
    import nltk

    pos_tags = nltk.pos_tag(tokens)
    ner_tree = nltk.ne_chunk(pos_tags)

    extracted_entities = defaultdict(list)
    for chunk in ner_tree:
        if hasattr(chunk, "label"):
            entity_name = " ".join(c[0] for c in chunk)
            entity_label = chunk.label()
            if entity_label in ["GPE", "LOCATION"]:
                entity_label = "GPE/LOCATION"
            extracted_entities[entity_label].append(entity_name)

    return {
        label: list(set(entities))
        for label, entities in extracted_entities.items()
    }
