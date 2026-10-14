import pytest

from src.preprocessing import (
    bn_to_en_digits,
    clean_text,
    en_to_bn_digits,
    light_stem,
    load_stopwords,
    normalize_unicode,
    split_sentences,
    tokenize,
)


def test_clean_text_removes_noise():
    raw = "আমি   বাংলা গান গাই! 😀 https://example.com <b>ভালো</b>। ২০২৬ সালে"
    assert clean_text(raw) == "আমি বাংলা গান গাই ভালো ২০২৬ সালে"


def test_clean_text_options():
    assert clean_text("সালে ২০২৬ এবং 2026", remove_digits=True) == "সালে এবং"
    assert clean_text("সাল ২০২৬", normalize_digits="en") == "সাল 2026"
    assert clean_text("hello বাংলা", remove_english=True) == "বাংলা"
    assert clean_text("বাংলা, ভাষা।", remove_punctuation=False) == "বাংলা, ভাষা।"
    with pytest.raises(ValueError):
        clean_text("বাংলা", normalize_digits="xx")


@pytest.mark.parametrize("bad", [None, 123, "", "   "])
def test_clean_text_handles_non_text(bad):
    assert clean_text(bad) == ""


def test_zero_width_characters_are_removed_by_default():
    assert "‍" not in clean_text("র‍্যাম")
    assert "‍" in clean_text("র‍্যাম", remove_zero_width=False)


def test_unicode_normalisation_makes_nukta_variants_equal():
    precomposed = "য়"  # য়
    decomposed = "য়"  # য + nukta
    assert normalize_unicode(precomposed) == normalize_unicode(decomposed)


def test_digit_conversion_roundtrip():
    assert bn_to_en_digits("২০২৬") == "2026"
    assert en_to_bn_digits("2026") == "২০২৬"


def test_split_sentences():
    sentences = split_sentences("আমি ভালো আছি। তুমি কেমন আছো? চমৎকার!!")
    assert sentences == ["আমি ভালো আছি।", "তুমি কেমন আছো?", "চমৎকার!!"]


def test_split_sentences_ignores_empty_fragments():
    assert split_sentences("   \n\n ।।। ") == []


def test_tokenize_keeps_bangla_words_whole():
    # vowel signs must not split words (this is the classic \w pitfall)
    assert tokenize("বাংলাদেশের মানুষ") == ["বাংলাদেশের", "মানুষ"]


def test_tokenize_stopwords_and_negation():
    text = "আমি ভালো আছি। খাবার ভালো না।"
    assert tokenize(text, remove_stopwords=True) == ["ভালো", "খাবার", "ভালো"]
    assert tokenize(text, remove_stopwords=True, keep_negation=True) == ["ভালো", "খাবার", "ভালো", "না"]


def test_tokenize_english_and_min_length():
    assert tokenize("Python শেখা") == ["শেখা"]
    assert tokenize("Python শেখা", keep_english=True) == ["python", "শেখা"]
    assert tokenize("এক দুই তিনজন", min_length=4) == ["তিনজন"]


def test_stopword_file_loads_and_negation_toggle():
    with_neg = load_stopwords(include_negation=True)
    without_neg = load_stopwords(include_negation=False)
    assert "এবং" in with_neg and "এবং" in without_neg
    assert "না" in with_neg and "না" not in without_neg


def test_light_stem():
    assert light_stem("শিক্ষকদের") == "শিক্ষক"
    assert light_stem("ছবিগুলো") == "ছবি"
    assert light_stem("মাটি") == "মাটি"  # too short to strip safely
