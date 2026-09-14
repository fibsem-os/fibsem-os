"""The correlation widgets style themselves from the text roles in
``fibsem.ui.tokens``, not from inline literals (FIB-978).

Before this guard the six widget files carried 71 ``setStyleSheet`` calls with
33 labels at 11 px, 17 at 12 px, and five different greys for "muted". A role
is named once in tokens; a literal here is a regression.
"""

import re
from pathlib import Path

WIDGETS = (
    Path(__file__).resolve().parents[2] / "fibsem" / "ui" / "correlation" / "widgets"
)
CHECKED = [
    "correlation_tab_widget.py",
    "coordinate_list_widget.py",
    "correlation_setup_section.py",
    "refractive_index_widget.py",
    "fit_confirmation_dialog.py",
    "fm_interpolate_dialog.py",
]
# Point-type colours live in the overlay on purpose; the canvases are matplotlib.
HEX = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")
FONT_SIZE = re.compile(r"font-size:\s*\d+px")


def _code_lines(path: Path):
    """Source lines with comments stripped, so a hex in a comment does not count."""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#", 1)[0] if not HEX.search(line.split("#", 1)[0]) else line
        yield n, code


def test_no_hex_colour_literals_in_the_correlation_widgets():
    hits = []
    for name in CHECKED:
        for n, line in _code_lines(WIDGETS / name):
            if HEX.search(line) and "tokens" not in line:
                hits.append(f"{name}:{n}: {line.strip()}")
    assert not hits, "hex colour literals; use fibsem.ui.tokens:\n" + "\n".join(hits)


def test_no_inline_font_sizes_in_the_correlation_widgets():
    hits = []
    for name in CHECKED:
        for n, line in (
            (n, l)
            for n, l in enumerate(
                (WIDGETS / name).read_text(encoding="utf-8").splitlines(), 1
            )
        ):
            if FONT_SIZE.search(line):
                hits.append(f"{name}:{n}: {line.strip()}")
    assert not hits, (
        "inline font sizes; use a text role from fibsem.ui.tokens:\n" + "\n".join(hits)
    )
