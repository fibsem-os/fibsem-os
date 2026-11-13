"""Every repository path the Markdown docs name in backticks exists.

A rename that orphans a doc fails here rather than in a reader's hands.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Release history names files as they were then.
HISTORY = {"CHANGES.md"}
# Named on purpose but never committed (.gitignore).
UNTRACKED = {"tests/data"}

_SPAN = re.compile(r"`([^`\s]+)`")


def _git_ls_files(*args: str) -> list:
    if shutil.which("git") is None:
        pytest.skip("needs git")
    try:
        out = subprocess.run(
            ["git", "ls-files", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout")
    if not out:
        pytest.skip("not a git checkout")
    return out


def _named_paths(text: str, tops: set):
    for match in _SPAN.finditer(text):
        span = match.group(1).split("#")[0].split(":")[0].rstrip("/")
        if "/" not in span or span.split("/")[0] not in tops:
            continue
        if any(c in span for c in "*<>{}$"):
            continue
        yield text.count("\n", 0, match.start()) + 1, span


def test_every_path_named_in_the_docs_exists():
    tops = {path.split("/")[0] for path in _git_ls_files()}
    missing = []
    for doc in _git_ls_files("*.md"):
        if doc in HISTORY:
            continue
        text = (REPO / doc).read_text(encoding="utf-8")
        for line, span in _named_paths(text, tops):
            if span not in UNTRACKED and not (REPO / span).exists():
                missing.append(f"{doc}:{line}: {span}")
    assert not missing, "Docs name paths that do not exist:\n" + "\n".join(missing)
