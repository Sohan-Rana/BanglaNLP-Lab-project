"""Small helpers for loading text and labelled data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LABELS, SAMPLE_CORPUS, SAMPLE_SENTIMENT


def load_sample_corpus(path: str | Path | None = None) -> str:
    """Read the demo corpus (UTF-8). Pass ``path`` to use your own text file."""
    return Path(path or SAMPLE_CORPUS).read_text(encoding="utf-8-sig")


def decode_bytes(raw: bytes) -> str:
    """Decode uploaded bytes as UTF-8 (with or without BOM) and explain failures clearly."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The file is not valid UTF-8. Please re-save it as 'UTF-8' text and try again."
        ) from exc


def load_labelled_data(source: str | Path | pd.DataFrame | None = None) -> pd.DataFrame:
    """Load and validate a labelled sentiment dataset.

    Expected columns: ``text`` and ``label`` (positive / negative / neutral).
    Empty texts and exact duplicates are dropped; labels are lower-cased and stripped.
    """
    if source is None:
        source = SAMPLE_SENTIMENT
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, encoding="utf-8-sig")

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain the columns 'text' and 'label' (missing: {sorted(missing)})")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""].drop_duplicates().reset_index(drop=True)

    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels {unknown}. Allowed labels: {LABELS}")
    if df.empty:
        raise ValueError("The dataset is empty after cleaning.")
    return df
