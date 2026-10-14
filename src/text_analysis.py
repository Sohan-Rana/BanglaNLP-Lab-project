"""Text statistics, word frequencies and n-grams for Bangla text."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from nltk import FreqDist
from nltk.util import ngrams as nltk_ngrams

from .preprocessing import (
    BN_DIGITS,
    EN_DIGITS,
    load_stopwords,
    normalize_unicode,
    split_sentences,
    tokenize,
)


def _as_text(text: object) -> str:
    return text if isinstance(text, str) else ""


def text_statistics(text: str) -> dict[str, float | int | str]:
    """Basic statistics for one text.

    Lengths are counted in Unicode characters. In Bangla one visible letter can be
    several characters (e.g. a consonant + a vowel sign), so ``avg_word_length`` is a
    relative measure, not a count of visible letters.
    """
    normalized = normalize_unicode(_as_text(text), remove_zero_width=True)
    tokens = tokenize(normalized)
    sentences = split_sentences(normalized)
    no_space = "".join(normalized.split())

    n_words = len(tokens)
    n_sentences = len(sentences)
    bangla_chars = sum(1 for ch in no_space if "ঀ" <= ch <= "৿")

    return {
        "characters": len(normalized),
        "characters_no_spaces": len(no_space),
        "bangla_characters": bangla_chars,
        "bangla_ratio": round(bangla_chars / len(no_space), 3) if no_space else 0.0,
        "digits": sum(1 for ch in no_space if ch in BN_DIGITS + EN_DIGITS),
        "words": n_words,
        "unique_words": len(set(tokens)),
        "sentences": n_sentences,
        "avg_word_length": round(sum(map(len, tokens)) / n_words, 2) if n_words else 0.0,
        "avg_sentence_length": round(n_words / n_sentences, 2) if n_sentences else 0.0,
        "lexical_diversity": round(len(set(tokens)) / n_words, 3) if n_words else 0.0,
        "longest_word": max(tokens, key=len) if tokens else "",
    }


def document_statistics(texts: Iterable[str]) -> pd.DataFrame:
    """Per-document counts (characters, words, sentences) for a collection of texts."""
    rows = []
    for text in texts:
        text = _as_text(text)
        rows.append(
            {
                "characters": len(normalize_unicode(text, remove_zero_width=True)),
                "words": len(tokenize(text)),
                "sentences": len(split_sentences(text)),
            }
        )
    return pd.DataFrame(rows, columns=["characters", "words", "sentences"])


def word_frequencies(
    text: str | Iterable[str],
    *,
    top_n: int | None = None,
    remove_stopwords: bool = True,
    min_length: int = 2,
    stem: bool = False,
) -> pd.DataFrame:
    """Count how often each word appears.

    ``text`` can be one string or an iterable of strings (documents).
    Returns a DataFrame with columns ``word``, ``count`` and ``percent``,
    sorted by count (descending) and then alphabetically for a stable order.
    """
    texts = [text] if isinstance(text, str) else list(text)
    tokens: list[str] = []
    for t in texts:
        tokens.extend(
            tokenize(_as_text(t), remove_stopwords=remove_stopwords, min_length=min_length, stem=stem)
        )

    if not tokens:
        return pd.DataFrame(columns=["word", "count", "percent"])

    dist = FreqDist(tokens)
    total = sum(dist.values())
    rows = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        rows = rows[:top_n]
    return pd.DataFrame(
        {
            "word": [w for w, _ in rows],
            "count": [c for _, c in rows],
            "percent": [round(100 * c / total, 2) for _, c in rows],
        }
    )


def ngram_frequencies(
    text: str | Iterable[str],
    n: int = 2,
    *,
    top_n: int | None = 20,
    remove_stopwords: bool = True,
) -> pd.DataFrame:
    """Most common word n-grams (bigrams by default).

    With ``remove_stopwords=True`` an n-gram is dropped when its first or last word
    is a stopword (the usual way to keep phrases like "কৃত্রিম বুদ্ধিমত্তা").
    """
    if n < 2:
        raise ValueError("n must be >= 2 (use word_frequencies for single words)")
    texts = [text] if isinstance(text, str) else list(text)
    stop = load_stopwords() if remove_stopwords else frozenset()

    dist: FreqDist = FreqDist()
    for t in texts:
        # every sentence is processed separately so n-grams never cross sentence borders
        for sentence in split_sentences(_as_text(t)):
            tokens = tokenize(sentence)
            for gram in nltk_ngrams(tokens, n):
                if gram[0] in stop or gram[-1] in stop:
                    continue
                dist[" ".join(gram)] += 1

    rows = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        rows = rows[:top_n]
    return pd.DataFrame({"ngram": [g for g, _ in rows], "count": [c for _, c in rows]})
