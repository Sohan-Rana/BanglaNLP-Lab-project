# Dataset guide - how to bring real data into BanglaNLP Lab

The toolkit expects a UTF-8 CSV with two columns:

```csv
text,label
"খাবারটা খুব ভালো ছিল",positive
"ডেলিভারি অনেক দেরিতে এসেছে",negative
"সভাটি দুপুর দুইটায় শুরু হবে",neutral
```

Allowed labels: `positive`, `negative`, `neutral`. Then run:

```bash
python scripts/train_model.py --data data/my_dataset.csv
```

or upload the file in the Streamlit app (*Sentiment* tab → *Train on your own labelled dataset*).

## Where to find data

- Look for published, human-annotated Bangla sentiment corpora (for example datasets released with academic papers on noisy social-media text or book/product reviews). **Check that the dataset still exists, read its licence, and cite the paper** - this repository does not redistribute third-party data.
- Or build your own (below).

## Building your own dataset responsibly

1. **Define the labels first.** Write 1-2 sentences and 3 examples per label. Decide how to treat sarcasm, mixed feelings and questions.
2. **Collect lawfully.** Respect each platform's terms of service and local law. Prefer text you own, text with an open licence, or text people explicitly agreed to share.
3. **Protect privacy.** Remove names, phone numbers, e-mail addresses, usernames, URLs and anything that can identify a person before saving the file. Do not publish raw private messages.
4. **Use more than one annotator** for at least part of the data and measure agreement (for example Cohen's kappa). Low agreement means the labels or guidelines are unclear.
5. **Balance and diversify.** Cover several topics, writing styles and communities so the model does not learn one narrow style.
6. **Keep a test set untouched.** Never tune on it.
7. **Document it** - source, collection dates, annotation guideline, annotator count, known gaps (a short "datasheet").

## Quick sanity checks (do these before trusting a score)

- Class counts (`df['label'].value_counts()`) - is any class very small?
- Duplicates and near-duplicates between train and test (they inflate scores).
- Are labels linked to length, topic or a few keywords? (see `outputs/text_length_by_class.png` and `outputs/top_features.png` for how to check)
- Read 50 random rows and 50 model errors by eye.
