# Model card - BanglaNLP Lab sentiment models

A short, honest description of what the two sentiment components do and where they fail.

## Components

| | Lexicon baseline | TF-IDF + Logistic Regression |
|---|---|---|
| Code | `LexiconSentiment` in `src/sentiment.py` | `SentimentModel` in `src/sentiment.py` |
| Needs training data | No | Yes (`text`, `label` CSV) |
| Output | positive / negative / neutral | positive / negative / neutral + probabilities |
| Explainable | Yes - lists matched words and negations | Partly - top word weights per class |

**Features of the ML model:** word 1-2-grams and character 2-4-grams (TF-IDF), logistic regression with balanced class weights.

## Intended use

- Learning NLP for Bangla, portfolio demos, quick experiments, a starting point for research.

## Not intended for

- Moderation, hiring, grading, credit, medical or legal decisions, or anything else that affects people without a human reviewing the result.
- Claims about "the accuracy of Bangla sentiment analysis".

## Data

- `data/sample_sentiment.csv`: **108 short sentences written by the project author** (36 per class). It is balanced, clean and tiny. It contains no personal data.
- Because one person wrote it, vocabulary is more "typical" than real social-media text, and the lexicon was written by the same person - this **favours the lexicon baseline**.

## Measured results on the demo data (seed 42)

| Evaluation | Lexicon baseline | TF-IDF + LogReg |
|---|---|---|
| Hold-out test (27 texts) - accuracy | 0.85 | 0.78 |
| Hold-out test (27 texts) - macro-F1 | 0.86 | 0.78 |
| 5-fold cross-validation - macro-F1 | - | 0.67 ± 0.07 |

With 27 test texts, one extra mistake moves accuracy by ~4 points. **Do not treat these numbers as real-world performance.** They only show that the pipeline runs end to end. The ML model does not beat the lexicon here because 81 training sentences are too few for a learned model - which is exactly why a real, larger dataset is needed.

## Known limitations and risks

1. **Tiny, single-author data** - results are noisy and optimistic.
2. **Shortcut learning** - in `outputs/top_features.png` the neutral class is driven partly by function words (থেকে, একটি, হবে) and the neutral sentences are shorter on average (median 6 vs 8 words, `outputs/text_length_by_class.png`). A model can learn *style*, not sentiment.
3. **Negation and sarcasm** - negation is handled by a simple window rule; sarcasm, irony and long-range negation are not handled.
4. **Language variety** - standard written Bangla only. Dialects, Banglish (Bangla in Latin letters), slang and emoji-only text are not covered.
5. **Topic and group bias** - a model trained on reviews may misread news or political text, and may behave differently for text about different communities. The demo data cannot reveal such bias; a real dataset should be audited per topic and group.
6. **Neutral is hard** - "neutral" mixes factual text, mixed feelings and "no opinion"; annotators often disagree on it.

## Recommendations

- Train on a properly sourced, human-labelled dataset with documented annotation guidelines (see [DATASET_GUIDE.md](DATASET_GUIDE.md)).
- Report macro-F1 and the per-class report, not only accuracy; keep a test set that is never used for tuning.
- Use the *uncertain* flag (`predict_with_confidence`) to send low-confidence cases to a human.
- Read the top features and a sample of errors before trusting any model.
- Re-evaluate whenever the type of text changes.
