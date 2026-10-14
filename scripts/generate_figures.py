"""Regenerate every figure, table and metric in outputs/ plus the README banner.

Usage (from the project root):
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS_DIR, MODEL_PATH, OUTPUT_DIR  # noqa: E402
from src.data import load_labelled_data, load_sample_corpus  # noqa: E402
from src.sentiment import fit_final_model, train_and_evaluate  # noqa: E402
from src.text_analysis import ngram_frequencies, text_statistics, word_frequencies  # noqa: E402
from src.visualization import (  # noqa: E402
    DEMO_NOTE,
    make_banner,
    make_wordcloud,
    plot_confusion_matrix,
    plot_length_distribution,
    plot_model_comparison,
    plot_sentiment_distribution,
    plot_top_features,
    plot_word_frequency,
)


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, dpi=200)
    plt.close(fig)
    print(f"  saved outputs/{name}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("1/3  Corpus analysis")
    corpus = load_sample_corpus()
    (OUTPUT_DIR / "corpus_statistics.json").write_text(
        json.dumps(text_statistics(corpus), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    freq = word_frequencies(corpus)
    freq.to_csv(OUTPUT_DIR / "word_frequencies.csv", index=False, encoding="utf-8-sig")
    ngram_frequencies(corpus, n=2, top_n=20).to_csv(OUTPUT_DIR / "bigrams.csv", index=False, encoding="utf-8-sig")
    save(plot_word_frequency(freq, top_n=15), "word_frequency.png")
    make_wordcloud(freq, save_path=OUTPUT_DIR / "wordcloud.png")
    print("  saved outputs/wordcloud.png")

    print("2/3  Sentiment dataset and models")
    df = load_labelled_data()
    save(plot_sentiment_distribution(df["label"].value_counts().to_dict(), footnote=DEMO_NOTE), "sentiment_distribution.png")
    save(plot_length_distribution(df["text"], df["label"], footnote=DEMO_NOTE), "text_length_by_class.png")

    result = train_and_evaluate(df)
    summary = result.summary()
    save(
        plot_confusion_matrix(
            result.confusion, result.labels, title="Confusion matrix - TF-IDF + Logistic Regression",
            subtitle=f"Hold-out test set ({result.n_test} texts) · rows = actual, columns = predicted", footnote=DEMO_NOTE,
        ),
        "confusion_matrix_model.png",
    )
    save(
        plot_confusion_matrix(
            result.baseline_confusion, result.labels, title="Confusion matrix - lexicon baseline",
            subtitle=f"Same hold-out test set ({result.n_test} texts) · no training involved", footnote=DEMO_NOTE,
        ),
        "confusion_matrix_lexicon.png",
    )
    save(plot_model_comparison(summary), "model_comparison.png")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "classification_report.txt").write_text(
        "TF-IDF + Logistic Regression - hold-out test set\n\n" + result.report, encoding="utf-8"
    )

    final_model = fit_final_model(df)
    final_model.save(MODEL_PATH)
    save(plot_top_features(final_model.top_features(n=8)), "top_features.png")

    print("3/3  README banner")
    make_banner(ASSETS_DIR / "banner.png")
    print("  saved assets/banner.png")

    print("\nDone. Metrics:\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
