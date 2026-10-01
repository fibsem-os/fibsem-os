"""The overview plot does not need the `reporting` extra; the PDF report does.

Both dialogs were imported in one ``try`` block in ``AutoLamellaUI.py``. The report
dialog imports reportlab, so without the extra the import failed, the overview plot --
matplotlib and Qt only -- went with it, and Generate Overview Plot said "Reporting
tools are not available". Generate Report, meanwhile, never checked the flag and
called a name the failed import had never defined.

Each case runs in a fresh interpreter: hiding reportlab only works before the window
module is first imported, and these must not leave it hidden for the rest of the run.

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
from fibsem.applications.autolamella.ui import AutoLamellaUI as window
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
        assert window.REPORTING_AVAILABLE is False, "reportlab was not hidden"
        assert callable(window.create_overview_image_widget)
        print("OK")
        """
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-2000:]


def test_generate_report_says_why_instead_of_raising():
    """Called unbound, on a stand-in for the window: the method reads one attribute."""
    result = _run(
        """
        from types import SimpleNamespace
        toasts = []
        window.notification_service.show_toast = (
            lambda message, kind="info", *a, **k: toasts.append((message, kind))
        )
        window.AutoLamellaUI.action_generate_report(SimpleNamespace(experiment=object()))
        assert toasts and toasts[0][1] == "warning", toasts
        assert "reporting" in toasts[0][0], toasts
        print("OK")
        """
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-2000:]
