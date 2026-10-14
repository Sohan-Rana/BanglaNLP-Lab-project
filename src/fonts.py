"""Download the Noto Sans Bengali font on first use (SIL OFL 1.1) if it is not in assets/fonts/."""

from __future__ import annotations

import urllib.request

from .config import FONT_BOLD, FONT_REGULAR

_BASE = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansBengali/hinted/ttf/"


def ensure_fonts() -> bool:
    """Return True if both font files exist (downloading them if needed)."""
    for path in (FONT_REGULAR, FONT_BOLD):
        if path.exists():
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_BASE + path.name, path)
        except Exception as exc:  # no internet, URL moved, etc.
            print(f"Could not download {path.name}: {exc}")
            return False
    return True
