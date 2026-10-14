<p align="center">
  <img src="assets/banner.png" alt="BanglaNLP Lab banner" width="100%">
</p>

# BanglaNLP Lab 🇧🇩

A small, honest **NLP toolkit for Bangla (Bengali) text** written in Python: text cleaning, statistics, word-frequency analysis, word clouds, sentiment classification and an interactive Streamlit dashboard.

It is built as a learning and portfolio project with a **Responsible AI** mindset: every score is labelled with its limits and the sentiment model can say "I am not sure".

## Features
Live : https://banglanlp-lab-project-f4vvtsvvwytpsogb22cvfy.streamlit.app/
| Feature | What it does | Code |
|---|---|---|
| **Text cleaning** | Unicode (NFC) normalisation, removes URLs / HTML / e-mails / emojis / punctuation (incl. `।`), digit conversion (০-৯ ⇄ 0-9), zero-width character handling | `src/preprocessing.py` |
| **Tokenising & sentences** | Bangla-aware word tokenizer (does not split at vowel signs), sentence splitter for `।` `?` `!`, stopword list, optional light suffix stripping | `src/preprocessing.py` |
| **Text statistics** | Words, unique words, sentences, characters, Bangla-character ratio, lexical diversity | `src/text_analysis.py` |
| **Word frequency & bigrams** | Frequency tables (NLTK `FreqDist`) and common word pairs | `src/text_analysis.py` |
| **Word cloud** | Correctly shaped Bangla glyphs using Noto Sans Bengali (downloaded automatically on first use) | `src/visualization.py` |
| **Sentiment analysis** | Explainable lexicon baseline **and** TF-IDF + Logistic Regression, with confidence and an "uncertain" flag | `src/sentiment.py` |
| **Dashboard** | Streamlit app: paste/upload text, explore every feature, train on your own labelled CSV | `app.py` |

## Results gallery

These images are created by `python scripts/generate_figures.py` (or by the **figures** GitHub Action).

| Word frequency | Word cloud |
|:--:|:--:|
| <img src="outputs/word_frequency.png" width="420"> | <img src="outputs/wordcloud.png" width="420"> |

| Sentiment distribution | Text length by class |
|:--:|:--:|
| <img src="outputs/sentiment_distribution.png" width="420"> | <img src="outputs/text_length_by_class.png" width="420"> |

| Confusion matrix (model) | Confusion matrix (lexicon) |
|:--:|:--:|
| <img src="outputs/confusion_matrix_model.png" width="420"> | <img src="outputs/confusion_matrix_lexicon.png" width="420"> |

| Model comparison | What the model learned |
|:--:|:--:|
| <img src="outputs/model_comparison.png" width="420"> | <img src="outputs/top_features.png" width="420"> |

## ⚠️ Read this before quoting any score

The bundled sentiment dataset (`data/sample_sentiment.csv`) is **108 short sentences written by the project author** - a *demo*, not a benchmark.

| Measured on the demo data (seed 42) | Lexicon baseline | TF-IDF + LogReg |
|---|---|---|
| Hold-out test (27 texts) - accuracy | 0.85 | 0.78 |
| Hold-out test (27 texts) - macro-F1 | 0.86 | 0.78 |
| 5-fold cross-validation - macro-F1 | - | 0.67 ± 0.07 |

With only 27 test texts one extra mistake changes accuracy by ~4 points, and the author wrote both the data and the lexicon. **These numbers show that the pipeline works, not how good Bangla sentiment analysis is.** For a real accuracy claim, train and evaluate on a properly sourced, human-labelled dataset (see [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md)).

## Quick start

```bash
git clone https://github.com/<Sohan-Rana>/bangla-nlp-lab.git
cd bangla-nlp-lab

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # open the dashboard
```

Other commands:

```bash
python scripts/generate_figures.py          # rebuild every chart in outputs/
python scripts/train_model.py               # train + evaluate on the demo data
python scripts/train_model.py --data data/my_dataset.csv   # your own labelled CSV
pip install -r requirements-dev.txt && python -m pytest    # run the tests
```

## Use it as a library

```python
from src.preprocessing import clean_text, tokenize
from src.text_analysis import text_statistics, word_frequencies
from src.sentiment import LexiconSentiment, fit_final_model
from src.data import load_labelled_data

text = "আমি বাংলায় গান গাই! 😀 https://example.com"
clean_text(text)                       # 'আমি বাংলায় গান গাই'
tokenize(text, remove_stopwords=True)  # ['বাংলায়', 'গান', 'গাই']

LexiconSentiment().predict("খাবার ভালো না")      # 'negative' (negation handled)

model = fit_final_model(load_labelled_data())
model.predict_with_confidence(["দারুণ সুন্দর একটি দিন"], threshold=0.5)
```

## Project structure

```
bangla-nlp-lab/
├── app.py                      # Streamlit dashboard
├── src/                        # config, data, fonts, preprocessing, text_analysis, sentiment, visualization
├── scripts/                    # generate_figures.py, train_model.py
├── data/                       # demo corpus, demo sentiment CSV, stopwords
├── outputs/                    # generated charts, tables, metrics
├── tests/                      # pytest suite (49 tests)
├── docs/                       # MODEL_CARD.md, DATASET_GUIDE.md
├── .github/workflows/          # tests.yml, figures.yml
├── requirements.txt
└── LICENSE
```

## Responsible AI notes

- **Uncertainty is visible.** `predict_with_confidence` flags low-confidence predictions for human review.
- **Transparency.** The lexicon baseline explains every decision; the ML model exposes its top word weights.
- **Honest evaluation.** Stratified hold-out split, cross-validation, per-class report and macro-F1.
- **Human in the loop.** Do not use these models for decisions about people without human review.

See also the [model card](docs/MODEL_CARD.md).

## Known limitations

- The sentence splitter breaks after abbreviations such as "ড. রহমান".
- Negation handling is a simple window rule; sarcasm, dialects and Banglish are not handled.
- The suffix stripper is an experimental heuristic, off by default.
- The font is downloaded on first use; without internet, charts fall back to a font that cannot draw Bangla.

## License & credits

Code: MIT © 2026 Sohan Rana.
Font: [Noto Sans Bengali](https://github.com/notofonts/bengali) by the Noto Project Authors, SIL Open Font License 1.1 (downloaded at run time, not stored in this repository).
