"""Bangla sentiment analysis: a transparent lexicon baseline + a TF-IDF / logistic-regression model.

IMPORTANT - read before quoting any number
-----------------------------------------
The bundled dataset (data/sample_sentiment.csv) is a tiny, hand-written *demo* set.
Scores measured on it say almost nothing about real-world performance. Train and
evaluate on a properly sourced, human-labelled dataset before making any accuracy claim.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import LABELS, MODEL_PATH, RANDOM_STATE
from .preprocessing import NEGATION_WORDS, normalize_for_model, tokenize, word_tokenizer


def _nfc_set(words: Iterable[str]) -> frozenset[str]:
    return frozenset(unicodedata.normalize("NFC", w) for w in words)


# --------------------------------------------------------------------------- #
# 1) Lexicon baseline (no training needed, fully explainable)
# --------------------------------------------------------------------------- #
POSITIVE_WORDS = _nfc_set(
    """ভালো ভাল ভালোবাসা ভালবাসা ভালোবাসি সুন্দর চমৎকার দারুণ অসাধারণ দুর্দান্ত সেরা আনন্দ আনন্দিত
    খুশি সুখী সুখ প্রিয় মজা মজার মজাদার সফল সাফল্য প্রশংসা প্রশংসনীয় উপকারী সহজ নিরাপদ
    মনোমুগ্ধকর মনোরম ধন্যবাদ অভিনন্দন শান্তি শান্ত গর্ব গর্বিত কৃতজ্ঞ সন্তুষ্ট আন্তরিক উন্নত মধুর
    আরামদায়ক যত্নবান পরিষ্কার ভদ্র পরিশ্রমী সৎ উপভোগ নির্ভরযোগ্য বিনয়ী সুস্বাদু নিখুঁত দুর্দান্ত
    উৎকৃষ্ট চমকপ্রদ উচ্ছ্বসিত স্বস্তি""".split()
)

NEGATIVE_WORDS = _nfc_set(
    """খারাপ বাজে জঘন্য কষ্ট দুঃখ দুঃখিত হতাশ হতাশা ব্যর্থ ব্যর্থতা সমস্যা ক্ষতি ঠকবাজ প্রতারণা প্রতারক
    ভয় ভয়ের ভীতিকর রাগ ক্ষুব্ধ বিরক্ত বিরক্তিকর অপছন্দ দুর্নীতি নোংরা অপরিষ্কার অসুস্থ দুর্ভোগ
    ভোগান্তি লজ্জা লজ্জাজনক ঘৃণা মিথ্যা বিপজ্জনক নষ্ট দুর্বল দুর্দশা ক্লান্ত দেরি দেরিতে কঠিন
    ঝামেলা অভিযোগ হেরে ঠকিয়ে বিশ্রী কুৎসিত অন্যায় অবহেলা অসন্তুষ্ট দুশ্চিন্তা অসহ্য অপমান""".split()
)

# words that must match exactly (they are prefixes of unrelated words, e.g. ভাল -> ভালুক)
_EXACT_ONLY = _nfc_set(["ভাল", "ভয়"])
_NEGATOR_ENDINGS = tuple(unicodedata.normalize("NFC", s) for s in ("েনি", "য়নি", "ইনি", "িনি"))
_NOT_NEGATORS = _nfc_set(["চিনি"])  # চিনি = sugar


def _is_negator(token: str) -> bool:
    if token in NEGATION_WORDS:
        return True
    return len(token) >= 4 and token not in _NOT_NEGATORS and token.endswith(_NEGATOR_ENDINGS)


def _matches(token: str, lexicon: frozenset[str]) -> bool:
    """Exact match, or lexicon word + a short suffix (ভালোই, সুন্দরভাবে ...)."""
    if token in lexicon:
        return True
    for word in lexicon:
        if word in _EXACT_ONLY or len(word) < 3:
            continue
        if token.startswith(word) and len(token) - len(word) <= 3:
            return True
    return False


@dataclass
class LexiconSentiment:
    """Counts positive/negative words and flips a word's polarity when a negator follows it.

    Example: "খাবার ভালো না" -> ভালো is followed by না, so it counts as negative.
    """

    positive: frozenset[str] = POSITIVE_WORDS
    negative: frozenset[str] = NEGATIVE_WORDS
    negation_window: int = 2

    def explain(self, text: str) -> list[dict]:
        """Return every matched word with its polarity (+1 / -1) and whether it was negated."""
        tokens = tokenize(text)
        hits = []
        for i, tok in enumerate(tokens):
            if _matches(tok, self.positive):
                polarity = 1
            elif _matches(tok, self.negative):
                polarity = -1
            else:
                continue
            following = tokens[i + 1 : i + 1 + self.negation_window]
            negated = any(_is_negator(t) for t in following)
            hits.append(
                {"word": tok, "base_polarity": polarity, "negated": negated, "score": -polarity if negated else polarity}
            )
        return hits

    def score(self, text: str) -> int:
        return sum(h["score"] for h in self.explain(text))

    def predict(self, text: str) -> str:
        s = self.score(text)
        return "positive" if s > 0 else "negative" if s < 0 else "neutral"

    def predict_many(self, texts: Iterable[str]) -> list[str]:
        return [self.predict(t) for t in texts]


# --------------------------------------------------------------------------- #
# 2) Machine-learning model
# --------------------------------------------------------------------------- #
def build_pipeline(C: float = 3.0) -> Pipeline:
    """TF-IDF (word 1-2-grams + character 2-4-grams) -> logistic regression.

    Character n-grams help with Bangla's rich suffixes (ভালো / ভালোই / ভালোবাসা).
    """
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    tokenizer=word_tokenizer,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    preprocessor=normalize_for_model,
                    lowercase=False,
                    ngram_range=(2, 4),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("features", features), ("clf", clf)])


class SentimentModel:
    """Small wrapper around the sklearn pipeline with confidence handling and persistence."""

    def __init__(self, C: float = 3.0):
        self.C = C
        self.pipeline: Pipeline = build_pipeline(C)
        self.fitted = False

    # -- training / prediction ------------------------------------------------
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SentimentModel":
        self.pipeline.fit(list(texts), list(labels))
        self.fitted = True
        return self

    @property
    def classes_(self) -> list[str]:
        return [str(c) for c in self.pipeline.named_steps["clf"].classes_]

    def predict(self, texts: Sequence[str]) -> list[str]:
        self._check_fitted()
        return [str(label) for label in self.pipeline.predict(list(texts))]

    def predict_proba(self, texts: Sequence[str]) -> pd.DataFrame:
        self._check_fitted()
        proba = self.pipeline.predict_proba(list(texts))
        return pd.DataFrame(proba, columns=self.classes_)

    def predict_with_confidence(self, texts: Sequence[str], threshold: float = 0.5) -> pd.DataFrame:
        """Predict labels together with the model's confidence.

        Rows whose top probability is below ``threshold`` are flagged ``uncertain=True``.
        Treat those as "needs a human" instead of trusting the label.
        """
        texts = list(texts)
        proba = self.predict_proba(texts)
        out = pd.DataFrame({"text": texts})
        out["label"] = proba.idxmax(axis=1).to_numpy()
        out["confidence"] = proba.max(axis=1).round(3).to_numpy()
        out["uncertain"] = out["confidence"] < threshold
        for cls in proba.columns:
            out[f"p_{cls}"] = proba[cls].round(3).to_numpy()
        return out

    def top_features(self, n: int = 8) -> dict[str, list[tuple[str, float]]]:
        """Words / word pairs that push the model most towards each class (transparency aid)."""
        self._check_fitted()
        names = self.pipeline.named_steps["features"].get_feature_names_out()
        coefs = self.pipeline.named_steps["clf"].coef_
        word_idx = np.array([i for i, name in enumerate(names) if name.startswith("word__")])
        result: dict[str, list[tuple[str, float]]] = {}
        if len(self.classes_) == 2:  # binary case: coef_ has a single row
            coefs = np.vstack([-coefs[0], coefs[0]])
        for row, cls in zip(coefs, self.classes_):
            top = word_idx[np.argsort(row[word_idx])[::-1][:n]]
            result[cls] = [(names[i].removeprefix("word__"), float(row[i])) for i in top]
        return result

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path = MODEL_PATH) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path = MODEL_PATH) -> "SentimentModel":
        """Load a saved model. Only load files you created yourself (joblib uses pickle)."""
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model is not trained yet. Call .fit(texts, labels) first.")


# --------------------------------------------------------------------------- #
# 3) Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    n_total: int
    n_train: int
    n_test: int
    class_counts: dict[str, int]
    labels: list[str]
    accuracy: float
    macro_f1: float
    confusion: np.ndarray
    report: str
    cv_macro_f1_mean: float | None
    cv_macro_f1_std: float | None
    baseline_accuracy: float
    baseline_macro_f1: float
    baseline_confusion: np.ndarray
    model: SentimentModel = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "class_counts": self.class_counts,
            "model": {
                "accuracy": round(self.accuracy, 3),
                "macro_f1": round(self.macro_f1, 3),
                "cv_macro_f1_mean": None if self.cv_macro_f1_mean is None else round(self.cv_macro_f1_mean, 3),
                "cv_macro_f1_std": None if self.cv_macro_f1_std is None else round(self.cv_macro_f1_std, 3),
            },
            "lexicon_baseline": {
                "accuracy": round(self.baseline_accuracy, 3),
                "macro_f1": round(self.baseline_macro_f1, 3),
            },
            "warning": "Measured on a tiny demo dataset - not a real-world performance estimate.",
        }


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    cv_folds: int = 5,
    random_state: int = RANDOM_STATE,
) -> EvaluationResult:
    """Stratified hold-out evaluation + cross-validation, compared with the lexicon baseline.

    The returned ``model`` is trained on the *training split only*; use
    :func:`fit_final_model` to train on all data once you are done evaluating.
    """
    texts, labels = df["text"].tolist(), df["label"].tolist()
    class_counts = df["label"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise ValueError("Need at least two different labels to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    model = SentimentModel().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    present = [lab for lab in LABELS if lab in class_counts]

    base_pred = LexiconSentiment().predict_many(X_test)

    # cross-validation on the full set (folds limited by the smallest class)
    folds = min(cv_folds, min(class_counts.values()))
    cv_mean = cv_std = None
    if folds >= 2:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(clone(build_pipeline()), texts, labels, cv=skf, scoring="f1_macro")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    return EvaluationResult(
        n_total=len(df),
        n_train=len(X_train),
        n_test=len(X_test),
        class_counts={k: int(v) for k, v in class_counts.items()},
        labels=present,
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", labels=present, zero_division=0)),
        confusion=confusion_matrix(y_test, y_pred, labels=present),
        report=classification_report(y_test, y_pred, labels=present, zero_division=0),
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        baseline_accuracy=float(accuracy_score(y_test, base_pred)),
        baseline_macro_f1=float(f1_score(y_test, base_pred, average="macro", labels=present, zero_division=0)),
        baseline_confusion=confusion_matrix(y_test, base_pred, labels=present),
        model=model,
    )


def fit_final_model(df: pd.DataFrame) -> SentimentModel:
    """Train on every row (do this only after you have evaluated with held-out data)."""
    return SentimentModel().fit(df["text"].tolist(), df["label"].tolist())
