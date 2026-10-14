"""BanglaNLP Lab - interactive Streamlit dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Make the `src` package importable no matter where the app is started from
# (needed on Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import LABELS, LABELS_BN, MODEL_PATH
from src.data import decode_bytes, load_labelled_data, load_sample_corpus
from src.preprocessing import clean_text, split_sentences, tokenize
from src.sentiment import LexiconSentiment, SentimentModel, fit_final_model, train_and_evaluate
from src.text_analysis import ngram_frequencies, text_statistics, word_frequencies
from src.visualization import (
    figure_to_png_bytes,
    image_to_png_bytes,
    make_wordcloud,
    plot_confusion_matrix,
    plot_sentiment_distribution,
    plot_top_features,
    plot_word_frequency,
)

st.set_page_config(page_title="BanglaNLP Lab", page_icon="🇧🇩", layout="wide")

EXAMPLE_SENTENCES = "খাবারটা খুব ভালো ছিল, আবার আসব।\nডেলিভারি অনেক দেরিতে এসেছে, খুবই বিরক্তিকর।\nসভাটি দুপুর দুইটায় শুরু হবে।"
DEMO_WARNING = (
    "⚠️ The built-in model is trained on a tiny, hand-written demo dataset (108 sentences). "
    "Its predictions are for learning and demonstration only - do not use them for real decisions."
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def show_image(image) -> None:
    """st.image across Streamlit versions (``width='stretch'`` is the newer spelling)."""
    try:
        st.image(image, width="stretch")
    except Exception:  # older Streamlit
        st.image(image, use_container_width=True)


def show_figure(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner="Preparing the demo sentiment model ...")
def get_demo_model() -> SentimentModel:
    """Load the saved demo model, or train it from the bundled dataset if it is missing/unreadable."""
    if MODEL_PATH.exists():
        try:
            return SentimentModel.load(MODEL_PATH)
        except Exception:
            pass  # e.g. saved with another scikit-learn version -> just retrain
    return fit_final_model(load_labelled_data())


@st.cache_data(show_spinner=False)
def cached_frequencies(text: str, remove_stop: bool, min_len: int, stem: bool) -> pd.DataFrame:
    return word_frequencies(text, remove_stopwords=remove_stop, min_length=min_len, stem=stem)


@st.cache_data(show_spinner="Drawing the word cloud ...")
def cached_wordcloud_png(words: tuple, counts: tuple, max_words: int) -> bytes:
    image = make_wordcloud(dict(zip(words, counts)), max_words=max_words)
    return image_to_png_bytes(image)


# --------------------------------------------------------------------------- #
# Sidebar: input text + options
# --------------------------------------------------------------------------- #
st.title("BanglaNLP Lab 🇧🇩")
st.caption("A Python toolkit for Bangla text: cleaning · statistics · word frequency · word cloud · sentiment")

with st.sidebar:
    st.header("1. Your text")
    source = st.radio("Where does the text come from?", ["Sample text", "Paste text", "Upload file"])

    text = ""
    if source == "Sample text":
        text = load_sample_corpus()
    elif source == "Paste text":
        text = st.text_area("Paste Bangla text here", height=220, placeholder="এখানে বাংলা লেখা পেস্ট করুন ...")
    else:
        uploaded = st.file_uploader("Upload a .txt or .csv file (UTF-8)", type=["txt", "csv"], key="text_file")
        if uploaded is not None:
            try:
                raw = decode_bytes(uploaded.getvalue())
                if uploaded.name.lower().endswith(".csv"):
                    table = pd.read_csv(io.StringIO(raw))
                    column = st.selectbox("Column that contains the text", list(table.columns))
                    text = "\n".join(table[column].dropna().astype(str))
                else:
                    text = raw
            except Exception as exc:  # show a friendly message instead of a traceback
                st.error(f"Could not read the file: {exc}")

    st.header("2. Options")
    top_n = st.slider("Words to show", 5, 40, 15)
    remove_stop = st.checkbox("Remove stopwords", value=True)
    min_len = st.slider("Minimum word length (characters)", 1, 5, 2)
    stem = st.checkbox("Light suffix stripping (experimental)", value=False)
    cloud_words = st.slider("Words in word cloud", 20, 150, 80, step=10)

if not text.strip():
    st.info("👈 Choose *Sample text*, paste some Bangla text, or upload a file to start.")
    st.stop()

tab_clean, tab_stats, tab_freq, tab_cloud, tab_sent, tab_about = st.tabs(
    ["🧹 Cleaning", "📊 Statistics", "🔤 Word frequency", "☁️ Word cloud", "🙂 Sentiment", "ℹ️ About & ethics"]
)

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
with tab_clean:
    st.subheader("Text cleaning")
    c1, c2, c3 = st.columns(3)
    opt_urls = c1.checkbox("Remove URLs, e-mails, HTML", value=True)
    opt_emoji = c1.checkbox("Remove emojis", value=True)
    opt_punct = c2.checkbox("Remove punctuation (incl. ।)", value=True)
    opt_digits = c2.checkbox("Remove digits", value=False)
    opt_english = c3.checkbox("Remove English letters", value=False)
    opt_digit_style = c3.selectbox("Digit style", ["keep as is", "Bangla (০-৯)", "English (0-9)"])

    cleaned = clean_text(
        text,
        remove_urls=opt_urls,
        remove_emails=opt_urls,
        remove_html=opt_urls,
        remove_emojis=opt_emoji,
        remove_punctuation=opt_punct,
        remove_digits=opt_digits,
        remove_english=opt_english,
        normalize_digits={"keep as is": None, "Bangla (০-৯)": "bn", "English (0-9)": "en"}[opt_digit_style],
    )
    left, right = st.columns(2)
    left.text_area("Original", value=text, height=260, disabled=True)
    right.text_area("Cleaned", value=cleaned, height=260, disabled=True)
    st.caption(f"{len(text):,} → {len(cleaned):,} characters. Unicode is normalised to NFC and zero-width characters are removed.")
    st.download_button("⬇️ Download cleaned text", cleaned, file_name="cleaned_text.txt", mime="text/plain")

# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
with tab_stats:
    st.subheader("Text statistics")
    stats = text_statistics(text)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words", f"{stats['words']:,}")
    m2.metric("Unique words", f"{stats['unique_words']:,}")
    m3.metric("Sentences", f"{stats['sentences']:,}")
    m4.metric("Characters", f"{stats['characters']:,}")

    st.dataframe(
        pd.DataFrame({"measure": list(stats.keys()), "value": [str(v) for v in stats.values()]}),
        hide_index=True,
    )
    st.caption(
        "Lengths are counted in Unicode characters; one visible Bangla letter can be several characters "
        "(consonant + vowel sign), so treat word length as a relative measure."
    )
    with st.expander("Sentences"):
        sentences = split_sentences(text)
        st.dataframe(
            pd.DataFrame({"sentence": sentences, "words": [len(tokenize(s)) for s in sentences]}),
            hide_index=True,
        )

# --------------------------------------------------------------------------- #
# Word frequency
# --------------------------------------------------------------------------- #
freq = cached_frequencies(text, remove_stop, min_len, stem)

with tab_freq:
    st.subheader("Word frequency")
    if freq.empty:
        st.warning("No words left after filtering. Try lowering the minimum word length or keeping stopwords.")
    else:
        left, right = st.columns([3, 2])
        with left:
            show_figure(plot_word_frequency(freq, top_n=top_n))
        with right:
            st.dataframe(freq.head(top_n), hide_index=True)
            st.download_button(
                "⬇️ Download all frequencies (CSV)",
                freq.to_csv(index=False, encoding="utf-8-sig"),
                file_name="word_frequencies.csv",
                mime="text/csv",
            )
        st.markdown("**Most common word pairs**")
        st.dataframe(ngram_frequencies(text, n=2, top_n=15, remove_stopwords=remove_stop), hide_index=True)

# --------------------------------------------------------------------------- #
# Word cloud
# --------------------------------------------------------------------------- #
with tab_cloud:
    st.subheader("Word cloud")
    if freq.empty:
        st.warning("Nothing to draw - no words left after filtering.")
    else:
        png = cached_wordcloud_png(tuple(freq["word"]), tuple(int(c) for c in freq["count"]), cloud_words)
        show_image(png)
        st.download_button("⬇️ Download word cloud (PNG)", png, file_name="wordcloud.png", mime="image/png")

# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
with tab_sent:
    st.subheader("Sentiment analysis")
    st.warning(DEMO_WARNING)

    custom_model = st.session_state.get("custom_model")
    model = custom_model if custom_model is not None else get_demo_model()
    st.caption("Model in use: **your uploaded dataset**" if custom_model is not None else "Model in use: **demo model**")

    st.markdown("**Analyse sentences** (one per line)")
    lines = st.text_area("Sentences", value=EXAMPLE_SENTENCES, height=140, label_visibility="collapsed")
    threshold = st.slider("Mark a prediction as 'uncertain' below this confidence", 0.34, 0.90, 0.50, step=0.01)
    sentences = [s.strip() for s in lines.splitlines() if s.strip()]

    if sentences:
        result = model.predict_with_confidence(sentences, threshold=threshold)
        lexicon = LexiconSentiment()
        table = pd.DataFrame(
            {
                "text": result["text"],
                "model prediction": [f"{LABELS_BN.get(l, l)} ({l})" for l in result["label"]],
                "confidence": result["confidence"],
                "uncertain?": result["uncertain"].map({True: "⚠️ yes - needs a human", False: ""}),
                "lexicon baseline": [LABELS_BN[lexicon.predict(s)] for s in sentences],
            }
        )
        st.dataframe(table, hide_index=True)
        with st.expander("Why did the lexicon baseline decide that?"):
            for s in sentences[:20]:
                hits = lexicon.explain(s)
                st.markdown(f"**{s}**")
                if hits:
                    st.write(
                        ", ".join(
                            f"{h['word']} ({'+' if h['score'] > 0 else '−'}{'; negated' if h['negated'] else ''})"
                            for h in hits
                        )
                    )
                else:
                    st.write("No sentiment words found → neutral.")

    st.divider()
    st.markdown("### Train on your own labelled dataset")
    st.write(
        "Upload a UTF-8 CSV with two columns: **text** and **label** (`positive`, `negative` or `neutral`). "
        "The app holds out 25% of it for testing and reports honest hold-out scores."
    )
    labelled_file = st.file_uploader("Labelled CSV", type=["csv"], key="labelled_csv")
    if labelled_file is not None:
        try:
            labelled = load_labelled_data(pd.read_csv(io.StringIO(decode_bytes(labelled_file.getvalue()))))
            with st.spinner("Training and evaluating ..."):
                evaluation = train_and_evaluate(labelled)
                trained = fit_final_model(labelled)
            st.session_state["custom_model"] = trained

            e1, e2, e3 = st.columns(3)
            e1.metric("Hold-out accuracy", f"{evaluation.accuracy:.2f}")
            e2.metric("Hold-out macro-F1", f"{evaluation.macro_f1:.2f}")
            if evaluation.cv_macro_f1_mean is not None:
                e3.metric("Cross-validated macro-F1", f"{evaluation.cv_macro_f1_mean:.2f} ± {evaluation.cv_macro_f1_std:.2f}")
            st.caption(
                f"{evaluation.n_train} training and {evaluation.n_test} test texts. "
                "Small or noisy datasets give unreliable scores - always read the per-class report below."
            )
            c_left, c_right = st.columns(2)
            with c_left:
                show_figure(plot_sentiment_distribution(evaluation.class_counts, title="Your dataset"))
            with c_right:
                show_figure(plot_confusion_matrix(evaluation.confusion, evaluation.labels, title="Confusion matrix"))
            st.code(evaluation.report)
            show_figure(plot_top_features(trained.top_features(n=8), footnote="BanglaNLP Lab · your dataset"))
            st.success("The analysis above now uses the model trained on your dataset.")
        except Exception as exc:
            st.error(f"Could not use this dataset: {exc}")

    if custom_model is not None and st.button("Reset to the demo model"):
        del st.session_state["custom_model"]
        st.rerun()

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About & responsible use")
    st.markdown(
        """
**BanglaNLP Lab** is a learning-oriented toolkit. Please keep these points in mind:

- **Demo data only.** The bundled sentiment dataset is small and hand-written by one person. Scores on it are *not*
  an estimate of real-world accuracy.
- **Models learn shortcuts.** On small data a model may rely on style cues (sentence length, common function words)
  instead of real sentiment. Check *Top features* before trusting a model.
- **Bangla is diverse.** Spelling, dialects, code-mixing (Banglish) and sarcasm are not handled well.
  Test on text that looks like your real use case.
- **Keep a human in the loop** for anything that affects people (moderation, hiring, grading, credit ...).
  Use the *uncertain* flag to route low-confidence cases to a person.
- **Privacy.** Don't upload personal data you have no right to process. Check dataset licences before reuse.
        """
    )
    st.markdown("**Class names used in this app**")
    st.dataframe(pd.DataFrame({"label": LABELS, "বাংলা": [LABELS_BN[l] for l in LABELS]}), hide_index=True)
    st.caption("Font: Noto Sans Bengali (SIL Open Font License 1.1).")
