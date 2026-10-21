"""Write web/src/theme.css from fibsem/theme/tokens.py, so the web app and the Qt
app share one palette. CI runs it and fails on a diff; run it after changing a
token."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fibsem.theme import tokens  # noqa: E402

OUT = ROOT / "web" / "src" / "theme.css"
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def css() -> str:
    lines = [
        "/* Written by scripts/web_theme.py from fibsem/theme/tokens.py; do not edit. */",
        ":root {",
    ]
    for name in sorted(vars(tokens)):
        value = getattr(tokens, name)
        if name.isupper() and isinstance(value, str) and _HEX.match(value):
            lines.append(
                f"  --fibsem-{name.lower().replace('_', '-')}: {value.lower()};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(css(), encoding="utf-8", newline="\n")
