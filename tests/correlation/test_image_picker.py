"""Tests for the Images tab's one-row image picker (FIB-302).

The picker replaced a combo + path line-edit showing the same file twice. Its
contract is small but load-bearing:

* items display a basename but carry the full path, so lamella candidates and a
  file browsed from elsewhere live in one control;
* ``path_selected`` fires only for user choices — combo *activation* and browse —
  so the programmatic ``show_path`` used for pre-loads and error-reverts cannot
  loop back into a reload.

Headless PyQt5, offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.correlation.widgets.correlation_tab_widget import _ImagePicker

LAMELLA = ["/lam/ref_Trench_ib.tif", "/lam/ref_MillRough_ib.tif"]
EXTERNAL = "/elsewhere/acquired_on_another_system_ib.tif"


def _picker():
    return _ImagePicker("TIFF (*.tif *.tiff);;All Files (*)")


def _emissions(picker):
    seen = []
    picker.path_selected.connect(seen.append)
    return seen


def test_items_show_basenames_but_carry_paths(qapp):
    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])

    assert [p.combo.itemText(i) for i in range(p.combo.count())] == [
        "ref_Trench_ib.tif",
        "ref_MillRough_ib.tif",
    ]
    assert p.combo.currentText() == "ref_Trench_ib.tif"
    assert p.current_path() == LAMELLA[0]


def test_current_path_is_empty_without_options(qapp):
    assert _picker().current_path() == ""


def test_show_path_selects_without_emitting(qapp):
    """Pre-loads and error-reverts go through show_path; emitting there would
    re-trigger the load that called it."""
    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    seen = _emissions(p)

    p.show_path(LAMELLA[1])
    assert p.current_path() == LAMELLA[1]
    assert seen == []


def test_show_path_adds_a_file_not_in_the_list(qapp):
    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    seen = _emissions(p)

    p.show_path(EXTERNAL)
    assert p.current_path() == EXTERNAL
    assert p.combo.count() == 3
    assert p.combo.currentText() == "acquired_on_another_system_ib.tif"
    assert seen == []


def test_show_path_does_not_duplicate_a_known_file(qapp):
    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    p.show_path(LAMELLA[1])
    p.show_path(LAMELLA[1])
    assert p.combo.count() == len(LAMELLA)


def test_activation_emits_the_full_path(qapp):
    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    seen = _emissions(p)

    p.combo.setCurrentIndex(1)
    p._on_activated(1)  # what QComboBox.activated delivers on a user pick
    assert seen == [LAMELLA[1]]


def test_browse_adds_the_file_selects_it_and_emits(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    seen = _emissions(p)
    monkeypatch.setattr(
        ctw.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (EXTERNAL, "")),
    )

    p._browse()
    assert seen == [EXTERNAL]  # the load is driven by the emission
    assert p.current_path() == EXTERNAL
    assert p.combo.count() == 3  # and it joins the list for re-selection


def test_cancelled_browse_changes_nothing(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    seen = _emissions(p)
    monkeypatch.setattr(
        ctw.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", ""))
    )

    p._browse()
    assert seen == []
    assert p.current_path() == LAMELLA[0]
    assert p.combo.count() == len(LAMELLA)


def test_set_options_replaces_the_list(qapp):
    p = _picker()
    p.set_options(LAMELLA, LAMELLA[0])
    p.set_options(["/other/a_ib.tif"], "/other/a_ib.tif")
    assert p.combo.count() == 1
    assert p.current_path() == "/other/a_ib.tif"
