"""Project-wide paths, labels and colours (single source of truth)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

SAMPLE_CORPUS = DATA_DIR / "sample_corpus.txt"
SAMPLE_SENTIMENT = DATA_DIR / "sample_sentiment.csv"
STOPWORDS_FILE = DATA_DIR / "bangla_stopwords.txt"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"

# Noto Sans Bengali (SIL Open Font License 1.1) - downloaded on first use by src/fonts.py
FONT_REGULAR = ASSETS_DIR / "fonts" / "NotoSansBengali-Regular.ttf"
FONT_BOLD = ASSETS_DIR / "fonts" / "NotoSansBengali-Bold.ttf"

RANDOM_STATE = 42

# Order matters: it also fixes the colour slot of each class in every chart.
LABELS = ["positive", "negative", "neutral"]
LABELS_BN = {"positive": "ইতিবাচক", "negative": "নেতিবাচক", "neutral": "নিরপেক্ষ"}

# Chart colours (validated colour-blind-safe categorical slots 1-3, light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CLASS_COLORS = {"positive": "#2a78d6", "negative": "#eb6834", "neutral": "#1baf7a"}

# One-hue sequential ramp (light -> dark), used for magnitude (word cloud, heatmaps)
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
