"""A lookup table that is present but unreadable must fail like a missing one.

FIB-592's dead calculator: `lut_available` asked `_LUT_PATH.exists()`, which is
true for a truncated file or an HTML error page. Every control stayed enabled,
every lookup raised into a bare handler that did not log, and the factor simply
never moved — permanently, and indistinguishably from a working widget.

Shipping the table (FIB-637) removes the routine cause, but not the state: a
mangled install still gets there. So the gate is now whether the table *loads*.
"""
import gzip
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.refractive_index import (
    _LUT_PATH,
    SliceScalingFactorLUT,
    lut_load_error,
)


# ---------------------------------------------------------------------------
# Detection: the states that must be recognised as unusable
# ---------------------------------------------------------------------------


def test_a_healthy_table_reports_no_error():
    assert lut_load_error() is None


def test_a_truncated_table_is_not_loadable(tmp_path):
    """The classic partial-transfer artefact. It is still a valid *file*."""
    bad = tmp_path / "trunc.csv.gz"
    bad.write_bytes(_LUT_PATH.read_bytes()[:40_000])

    with pytest.raises(Exception):
        SliceScalingFactorLUT(bad)


def test_an_error_page_is_not_loadable(tmp_path):
    """A proxy or captive portal serves HTML with a 200. It has the right name
    and a plausible size, and nothing about it looks wrong on disk."""
    bad = tmp_path / "portal.csv.gz"
    bad.write_bytes(b"<!DOCTYPE html><html><body>403 Forbidden</body></html>")

    with pytest.raises(Exception):
        SliceScalingFactorLUT(bad)


def test_a_wellformed_csv_with_the_wrong_columns_is_not_loadable(tmp_path):
    """Readable, parses fine, and still cannot answer a lookup — the failure
    that `exists()` and "does it parse" both miss."""
    bad = tmp_path / "wrong.csv.gz"
    bad.write_bytes(gzip.compress(b"a,b,c\n1,2,3\n"))

    with pytest.raises(Exception):
        SliceScalingFactorLUT(bad)


# ---------------------------------------------------------------------------
# Consequence: the widget must disable, warn, and say which state it is in
# ---------------------------------------------------------------------------


def _widget(qapp, error):
    """Build the widget with `lut_load_error` reporting *error*."""
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    original = riw.lut_load_error
    riw.lut_load_error = lambda: error
    try:
        return riw.RefractiveIndexWidget()
    finally:
        riw.lut_load_error = original


def test_an_unusable_table_disables_the_calculator(qapp):
    w = _widget(qapp, EOFError("Compressed file ended before the end-of-stream marker"))

    assert w._lut_available is False
    assert not w._spin_na.isEnabled(), "a dead calculator must not look live"
    assert not w._spin_n2.isEnabled()


def test_unreadable_and_missing_read_differently(qapp, tmp_path, monkeypatch):
    """Both disable, but they are different problems worth telling apart: a
    corrupted install versus a file that was never there. The wording is chosen
    by whether the path exists, so the missing case needs a path that doesn't."""
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    unreadable = _widget(qapp, EOFError("truncated"))
    assert "unreadable" in unreadable._lut_warning.text().lower()

    monkeypatch.setattr(riw, "_LUT_PATH", tmp_path / "absent.csv.gz")
    missing = _widget(qapp, FileNotFoundError("no such file"))
    assert "not found" in missing._lut_warning.text().lower()


def test_a_healthy_table_leaves_the_calculator_alone(qapp):
    w = _widget(qapp, None)

    assert w._lut_available is True
    assert w._spin_na.isEnabled()
    assert not hasattr(w, "_lut_warning")
