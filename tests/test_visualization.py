import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src.text_analysis import word_frequencies
from src.visualization import (
    make_banner,
    make_wordcloud,
    plot_confusion_matrix,
    plot_length_distribution,
    plot_sentiment_distribution,
    plot_top_features,
    plot_word_frequency,
)

FREQ = word_frequencies("বাংলা ভাষা বাংলা শব্দ ভাষা বাংলা গবেষণা", remove_stopwords=False)


def test_word_frequency_chart_saves_png(tmp_path):
    out = tmp_path / "freq.png"
    fig = plot_word_frequency(FREQ, top_n=5, save_path=out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 5_000


def test_word_frequency_chart_rejects_empty():
    with pytest.raises(ValueError):
        plot_word_frequency(FREQ.head(0))


def test_wordcloud_returns_image(tmp_path):
    out = tmp_path / "cloud.png"
    image = make_wordcloud(FREQ, width=600, height=300, save_path=out)
    assert isinstance(image, Image.Image) and image.size == (600, 300)
    assert out.exists()


def test_wordcloud_rejects_empty():
    with pytest.raises(ValueError):
        make_wordcloud({})


def test_distribution_and_confusion_charts(tmp_path):
    for fig in (
        plot_sentiment_distribution({"positive": 3, "negative": 2, "neutral": 1}, save_path=tmp_path / "d.png"),
        plot_confusion_matrix(np.array([[3, 1, 0], [0, 2, 1], [1, 0, 4]]), ["positive", "negative", "neutral"], save_path=tmp_path / "c.png"),
        plot_length_distribution(["ক খ গ", "ক খ", "ক খ গ ঘ ঙ"], ["positive", "negative", "neutral"], save_path=tmp_path / "l.png"),
        plot_top_features({k: [("ভালো", 1.0), ("খারাপ", 0.5)] for k in ("positive", "negative", "neutral")}, n=2, save_path=tmp_path / "t.png"),
    ):
        plt.close(fig)
    assert {p.name for p in tmp_path.iterdir()} == {"d.png", "c.png", "l.png", "t.png"}


def test_banner(tmp_path):
    from src.config import FONT_BOLD

    if not FONT_BOLD.exists():
        pytest.skip("Noto Sans Bengali font could not be downloaded (offline)")
    image = make_banner(tmp_path / "banner.png", width=800, height=210)
    assert image.size == (800, 210) and (tmp_path / "banner.png").exists()
