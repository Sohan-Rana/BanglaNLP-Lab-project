"""Train, evaluate and save the sentiment model on your own labelled CSV.

The CSV needs two columns: ``text`` and ``label`` (positive / negative / neutral).

Usage:
    python scripts/train_model.py                              # uses the tiny demo dataset
    python scripts/train_model.py --data data/my_dataset.csv   # your own data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH, SAMPLE_SENTIMENT  # noqa: E402
from src.data import load_labelled_data  # noqa: E402
from src.sentiment import fit_final_model, train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(SAMPLE_SENTIMENT), help="CSV with 'text' and 'label' columns")
    parser.add_argument("--output", default=str(MODEL_PATH), help="where to save the trained model (.joblib)")
    parser.add_argument("--test-size", type=float, default=0.25, help="share of data held out for testing")
    args = parser.parse_args()

    df = load_labelled_data(args.data)
    print(f"Loaded {len(df)} rows: {df['label'].value_counts().to_dict()}")

    result = train_and_evaluate(df, test_size=args.test_size)
    print("\nHold-out evaluation (TF-IDF + Logistic Regression)\n" + result.report)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))

    model = fit_final_model(df)  # final model uses all data
    path = model.save(args.output)
    print(f"\nSaved model trained on all {len(df)} rows to {path}")
    if Path(args.data).resolve() == Path(SAMPLE_SENTIMENT).resolve():
        print("NOTE: this is the tiny demo dataset - do not quote these scores as real accuracy.")


if __name__ == "__main__":
    main()
