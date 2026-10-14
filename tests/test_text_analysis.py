import pandas as pd

from src.text_analysis import document_statistics, ngram_frequencies, text_statistics, word_frequencies


def test_text_statistics_basic():
    stats = text_statistics("আমি ভালো আছি। তুমি কেমন আছো?")
    assert stats["words"] == 6  # আমি ভালো আছি তুমি কেমন আছো
    assert stats["unique_words"] == 6
    assert stats["sentences"] == 2
    assert stats["bangla_ratio"] > 0.9
    assert stats["longest_word"]


def test_text_statistics_empty_and_invalid():
    for value in ("", None):
        stats = text_statistics(value)
        assert stats["words"] == 0 and stats["sentences"] == 0 and stats["avg_word_length"] == 0.0


def test_document_statistics_shape():
    df = document_statistics(["আমি ভালো আছি।", "তুমি কেমন আছো? ভালো তো?"])
    assert list(df.columns) == ["characters", "words", "sentences"]
    assert df["sentences"].tolist() == [1, 2]


def test_word_frequencies_counts_and_order():
    df = word_frequencies("কাজ ভালো ভালো মন্দ ভালো কাজ", remove_stopwords=False)
    assert df.iloc[0]["word"] == "ভালো" and df.iloc[0]["count"] == 3
    assert df.iloc[1]["word"] == "কাজ" and df.iloc[1]["count"] == 2
    assert abs(df["percent"].sum() - 100) < 0.1
    assert df["count"].is_monotonic_decreasing


def test_word_frequencies_filters_stopwords_and_accepts_many_documents():
    df = word_frequencies(["আমি ভালো আছি", "আমরা ভালো আছি"], remove_stopwords=True)
    assert df["word"].tolist() == ["ভালো"]
    assert df["count"].tolist() == [2]


def test_word_frequencies_empty_returns_empty_frame():
    df = word_frequencies("এবং ও আর")
    assert isinstance(df, pd.DataFrame) and df.empty


def test_ngram_frequencies():
    df = ngram_frequencies("ভালো কাজ। ভালো কাজ। মন্দ কাজ।", n=2, remove_stopwords=False)
    assert df.iloc[0]["ngram"] == "ভালো কাজ" and df.iloc[0]["count"] == 2


def test_ngrams_do_not_cross_sentences():
    df = ngram_frequencies("আজ বৃষ্টি। কাল রোদ।", n=2, remove_stopwords=False)
    assert "বৃষ্টি কাল" not in df["ngram"].tolist()
