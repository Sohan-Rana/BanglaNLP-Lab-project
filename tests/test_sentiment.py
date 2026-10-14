import numpy as np
import pandas as pd
import pytest

from src.config import LABELS
from src.data import load_labelled_data
from src.sentiment import (
    LexiconSentiment,
    SentimentModel,
    fit_final_model,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def dataset():
    return load_labelled_data()


@pytest.fixture(scope="module")
def trained(dataset):
    return fit_final_model(dataset)


# --- data loading ---------------------------------------------------------
def test_sample_dataset_is_valid_and_balanced(dataset):
    assert len(dataset) >= 100
    assert set(dataset["label"]) == set(LABELS)
    assert dataset["label"].value_counts().nunique() == 1  # balanced demo set
    assert not dataset["text"].duplicated().any()


def test_load_labelled_data_validation():
    with pytest.raises(ValueError, match="text"):
        load_labelled_data(pd.DataFrame({"sentence": ["ক"], "label": ["positive"]}))
    with pytest.raises(ValueError, match="Unknown labels"):
        load_labelled_data(pd.DataFrame({"text": ["ক"], "label": ["happy"]}))
    cleaned = load_labelled_data(pd.DataFrame({"text": [" ভালো ", "ভালো", None, ""], "label": [" Positive", "positive", "neutral", "neutral"]}))
    assert cleaned["text"].tolist() == ["ভালো"] and cleaned["label"].tolist() == ["positive"]


# --- lexicon baseline -------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("খাবারটা খুব ভালো ছিল", "positive"),
        ("পরিষেবা বাজে", "negative"),
        ("খাবার ভালো না", "negative"),  # negation flips the word
        ("ভালো লাগেনি", "negative"),  # negation attached to the verb
        ("আজ বৃহস্পতিবার", "neutral"),
        ("", "neutral"),
    ],
)
def test_lexicon_predictions(text, expected):
    assert LexiconSentiment().predict(text) == expected


def test_lexicon_does_not_confuse_unrelated_words():
    assert LexiconSentiment().predict("ভালুক বনে থাকে") == "neutral"  # ভালুক is not ভাল
    assert LexiconSentiment().predict("চিনি কম দিন") == "neutral"  # চিনি is not a negation


def test_lexicon_explain_reports_negation():
    hits = LexiconSentiment().explain("খাবার ভালো না")
    assert hits == [{"word": "ভালো", "base_polarity": 1, "negated": True, "score": -1}]


# --- ML model ---------------------------------------------------------------
def test_model_predictions_and_probabilities(trained):
    texts = ["দারুণ সুন্দর একটি দিন", "খুব বাজে অভিজ্ঞতা", "ট্রেন ছয়টায় ছাড়বে"]
    assert set(trained.classes_) == set(LABELS)
    assert all(p in LABELS for p in trained.predict(texts))
    proba = trained.predict_proba(texts)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_fits_its_training_data(dataset, trained):
    accuracy = np.mean(np.array(trained.predict(dataset["text"].tolist())) == dataset["label"].to_numpy())
    assert accuracy > 0.9


def test_confidence_and_uncertain_flag(trained):
    out = trained.predict_with_confidence(["দারুণ সুন্দর", "ক খ গ ঘ"], threshold=0.99)
    assert {"text", "label", "confidence", "uncertain"} <= set(out.columns)
    assert out["uncertain"].all()  # nothing reaches 99% confidence on this tiny model
    assert (out["confidence"] <= 1).all()


def test_top_features_structure(trained):
    features = trained.top_features(n=5)
    assert set(features) == set(LABELS)
    assert all(len(v) == 5 for v in features.values())


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        SentimentModel().predict(["ভালো"])


def test_save_and_load_roundtrip(trained, tmp_path):
    path = trained.save(tmp_path / "model.joblib")
    loaded = SentimentModel.load(path)
    texts = ["সুন্দর দিন", "খারাপ খাবার"]
    assert loaded.predict(texts) == trained.predict(texts)


# --- evaluation ---------------------------------------------------------------
def test_train_and_evaluate(dataset):
    result = train_and_evaluate(dataset)
    assert result.n_train + result.n_test == len(dataset)
    assert 0 <= result.accuracy <= 1 and 0 <= result.macro_f1 <= 1
    assert result.confusion.shape == (3, 3) and result.confusion.sum() == result.n_test
    assert "tiny demo dataset" in result.summary()["warning"]


def test_train_and_evaluate_is_reproducible(dataset):
    a, b = train_and_evaluate(dataset), train_and_evaluate(dataset)
    assert a.accuracy == b.accuracy and (a.confusion == b.confusion).all()


def test_train_and_evaluate_needs_two_classes():
    df = pd.DataFrame({"text": ["ক", "খ", "গ", "ঘ"], "label": ["positive"] * 4})
    with pytest.raises(ValueError):
        train_and_evaluate(df)
