"""Check a pull request body before it becomes the squash commit (CONTRIBUTING.md,
"Commit messages"): the ``Release-Note:`` line is its last paragraph, and nothing in
it is an AI attribution that would ride into the commit.

Usage: ``python .github/scripts/check_pr_body.py`` with the body in ``PR_BODY``;
exits 1 with the reasons on stdout.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List

RELEASE_NOTE = re.compile(r"^Release-Note:\s*\S", re.MULTILINE)
# Trailers GitHub appends to, or keeps in, a squash commit; any of them after the
# Release-Note buries it, and all of them are attribution the repository refuses.
ATTRIBUTION = re.compile(
    r"^(Co-authored-by|Claude-Session):|Generated with \[Claude Code\]|claude\.ai/code",
    re.MULTILINE | re.IGNORECASE,
)


def problems(body: str) -> List[str]:
    """Why *body* would make a bad squash commit; empty when it is fine."""
    text = body.replace("\r\n", "\n").strip()
    found: List[str] = []
    if ATTRIBUTION.search(text):
        found.append(
            "the body carries an AI attribution (Co-authored-by, Claude-Session or a "
            "Claude Code footer); remove it, it would ride into the squash commit"
        )
    notes = RELEASE_NOTE.findall(text)
    if not notes:
        found.append(
            "no `Release-Note:` line; add one as the last paragraph "
            "(`Release-Note: none` for a change no user would notice)"
        )
        return found
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    last = paragraphs[-1].strip()
    if (
        not last.startswith("Release-Note:")
        or "\n" in last
        and not all(
            line.startswith(("Release-Note:", " ")) for line in last.splitlines()
        )
    ):
        found.append(
            "the `Release-Note:` line is not the last paragraph of the body, so git "
            "will not see it as a trailer; move it to the end"
        )
    return found


def main() -> int:
    body = os.environ.get("PR_BODY", "")
    found = problems(body)
    for line in found:
        print(f"- {line}")
    if found:
        print("\nSee CONTRIBUTING.md, 'Commit messages'.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
