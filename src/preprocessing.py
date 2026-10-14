"""Bangla text preprocessing: normalisation, cleaning, sentence splitting, tokenising.

Design notes
------------
* Everything is normalised to Unicode **NFC**. Note that NFC *decomposes* the
  precomposed letters য় ড় ঢ় into base + nukta (e.g. U+09DF -> U+09AF U+09BC).
  This is intentional: it makes the two ways of typing these letters identical.
* Python's ``\\w`` does **not** match Bangla vowel signs (া ি ী ু ে ...), because they
  are combining marks. So we never use ``\\w`` for Bangla; we use explicit ranges.
* Zero-width joiner / non-joiner characters are removed by default. This unifies
  spellings for matching (e.g. র‍্যাম vs র্যাম) but can change how a word is rendered.
  Pass ``remove_zero_width=False`` if you need to keep them.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .config import STOPWORDS_FILE

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
EN_DIGITS = "0123456789"
_BN2EN = str.maketrans(BN_DIGITS, EN_DIGITS)
_EN2BN = str.maketrans(EN_DIGITS, BN_DIGITS)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


#: Words that flip the meaning of a nearby word. Kept out of the stopword file so
#: that sentiment analysis can keep them.
NEGATION_WORDS = frozenset(_nfc(w) for w in ("না", "নয়", "নেই", "নাই", "নহে", "নি"))

# Bangla letters + signs (candrabindu .. vowel signs/virama/khanda-ta .. nukta-letters)
_BN_CHARS = "ঁ-ৣৰৱ"
BN_WORD_RE = re.compile(f"[{_BN_CHARS}]+")
BN_OR_EN_WORD_RE = re.compile(f"[{_BN_CHARS}]+|[A-Za-z]+")

_ZERO_WIDTH_RE = re.compile("[​‌‍‎‏⁠﻿]")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HTML_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(f"[@#][A-Za-z0-9_{_BN_CHARS}]+")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿︀-️←-⇿]",
)
_LATIN_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RE = re.compile(f"[{BN_DIGITS}{EN_DIGITS}]+")

# Split after । ॥ ! ? (unless another terminator follows), after ". " and at newlines.
_SENT_SPLIT_RE = re.compile(r"(?<=[।॥!?])(?![।॥!?])\s*|(?<=\.)\s+|\n+")


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def normalize_unicode(text: str, remove_zero_width: bool = False) -> str:
    """NFC-normalise ``text`` and fix the khanda-ta (ত্‍ -> ৎ) spelling variant."""
    if not isinstance(text, str):
        return ""
    text = _nfc(text)
    text = text.replace("ত্‍", "ৎ")
    if remove_zero_width:
        text = _ZERO_WIDTH_RE.sub("", text)
    return text


def bn_to_en_digits(text: str) -> str:
    """Convert Bangla digits (০-৯) to ASCII digits (0-9)."""
    return text.translate(_BN2EN)


def en_to_bn_digits(text: str) -> str:
    """Convert ASCII digits (0-9) to Bangla digits (০-৯)."""
    return text.translate(_EN2BN)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(
    text: str,
    *,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True,
    remove_mentions: bool = True,
    remove_emojis: bool = True,
    remove_punctuation: bool = True,
    remove_digits: bool = False,
    remove_english: bool = False,
    normalize_digits: str | None = None,
    remove_zero_width: bool = True,
) -> str:
    """Clean a Bangla string and return it with single spaces between words.

    Parameters
    ----------
    normalize_digits:
        ``None`` keeps digits as they are, ``"en"`` converts to 0-9, ``"bn"`` to ০-৯.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text, remove_zero_width=remove_zero_width)

    if remove_html:
        text = _HTML_RE.sub(" ", text)
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_emails:
        text = _EMAIL_RE.sub(" ", text)
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)
    if remove_emojis:
        text = _EMOJI_RE.sub(" ", text)

    if normalize_digits == "en":
        text = bn_to_en_digits(text)
    elif normalize_digits == "bn":
        text = en_to_bn_digits(text)
    elif normalize_digits is not None:
        raise ValueError("normalize_digits must be None, 'en' or 'bn'")

    if remove_digits:
        text = _DIGIT_RE.sub(" ", text)
    if remove_english:
        text = _LATIN_RE.sub(" ", text)

    if remove_punctuation:
        # Unicode categories starting with P (punctuation) or S (symbols),
        # which includes the Bangla danda '।' and the taka sign '৳'.
        text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)

    return " ".join(text.split())


def normalize_for_model(text: str) -> str:
    """Light cleaning used before feature extraction for machine-learning models.

    Keeps digits (converted to ASCII) and English words, drops URLs/HTML/emoji/punctuation.
    """
    return clean_text(text, normalize_digits="en").lower()


# --------------------------------------------------------------------------- #
# Stopwords
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def load_stopwords(path: str | Path | None = None, include_negation: bool = True) -> frozenset[str]:
    """Load the Bangla stopword list.

    Set ``include_negation=False`` to keep words like না / নয় / নেই in the text
    (recommended for sentiment analysis).
    """
    file = Path(path) if path else STOPWORDS_FILE
    words: set[str] = set()
    if file.exists():
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(_nfc(line))
    if include_negation:
        words |= NEGATION_WORDS
    return frozenset(words)


# --------------------------------------------------------------------------- #
# Sentences & tokens
# --------------------------------------------------------------------------- #
def _has_content(fragment: str) -> bool:
    return any(ch.isalnum() or "ঁ" <= ch <= "ৣ" for ch in fragment)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ।, ॥, !, ?, '. ' and newlines.

    Limitation: abbreviations such as "ড. রহমান" are split after the full stop.
    """
    text = normalize_unicode(text, remove_zero_width=True)
    parts = (p.strip() for p in _SENT_SPLIT_RE.split(text) if p is not None)
    return [p for p in parts if p and _has_content(p)]


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = False,
    stopwords: Iterable[str] | None = None,
    keep_negation: bool = False,
    keep_english: bool = False,
    min_length: int = 1,
    stem: bool = False,
) -> list[str]:
    """Split text into Bangla word tokens.

    Parameters
    ----------
    keep_negation:
        When removing stopwords, still keep না / নয় / নেই ...
    keep_english:
        Also return English words (lower-cased). Off by default.
    min_length:
        Minimum token length in Unicode characters (vowel signs count as characters).
    stem:
        Apply :func:`light_stem` (experimental heuristic).
    """
    cleaned = clean_text(text, remove_digits=True)
    regex = BN_OR_EN_WORD_RE if keep_english else BN_WORD_RE
    tokens = [t.lower() if t.isascii() else t for t in regex.findall(cleaned)]

    if remove_stopwords:
        sw = frozenset(_nfc(w) for w in stopwords) if stopwords is not None else load_stopwords(
            include_negation=not keep_negation
        )
        tokens = [t for t in tokens if t not in sw]

    if stem:
        tokens = [light_stem(t) for t in tokens]

    if min_length > 1:
        tokens = [t for t in tokens if len(t) >= min_length]
    return tokens


def word_tokenizer(text: str) -> list[str]:
    """Tokenizer used by the sentiment model (keeps stopwords, negation and English words)."""
    return tokenize(text, keep_english=True)


# --------------------------------------------------------------------------- #
# Experimental light stemmer
# --------------------------------------------------------------------------- #
_SUFFIXES: Sequence[str] = tuple(
    sorted(
        (
            _nfc(s)
            for s in (
                "গুলোকে", "গুলোর", "গুলোতে", "গুলো", "গুলি", "গুলা", "গুলার",
                "দেরকে", "দের", "টিকে", "টির", "টিতে", "টি", "টা", "খানা", "খানি",
            )
        ),
        key=len,
        reverse=True,
    )
)


def light_stem(token: str, min_stem_length: int = 3) -> str:
    """Strip one common plural/classifier suffix (গুলো, দের, টি ...).

    This is a *conservative heuristic*, not a real stemmer: it only removes
    a suffix when at least ``min_stem_length`` characters remain, so short words
    such as মাটি or রুটি are left untouched.
    """
    token = _nfc(token)
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem_length:
            return token[: -len(suffix)]
    return token
