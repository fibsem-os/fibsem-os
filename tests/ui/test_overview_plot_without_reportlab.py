"""The window and its overview plot do not need the `reporting` extra (reportlab).

It runs in a fresh interpreter: hiding reportlab only works before the window
module is first imported, and it must not leave it hidden for the rest of the run.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_plot_without_reportlab.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import fibsem

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

_WITHOUT_REPORTLAB = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.modules["reportlab"] = None  # `import reportlab` now raises ImportError
from fibsem.applications.autolamella.ui import window
"""


# The checkout this test is running against. The child must import the same one: tests
# run from a temporary directory, so left to itself it would import whichever fibsem is
# installed -- in a worktree, the main checkout's, which is not the code under test.
_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(fibsem.__file__)))


def _run(body: str) -> subprocess.CompletedProcess:
    script = _WITHOUT_REPORTLAB + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_SOURCE_ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )


def test_the_overview_plot_is_there_without_reportlab():
    result = _run(
        """
        assert callable(window.create_overview_image_widget)
        print("OK")
        """
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-2000:]
