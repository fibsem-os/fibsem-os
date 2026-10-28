"""The Find button beside a vendor library's folder."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from fibsem import vendor_libraries as vl  # noqa: E402
from fibsem.drivers.setup import SetupLibrary  # noqa: E402
from fibsem.ui.widgets import library_folder_edit  # noqa: E402
from fibsem.ui.widgets.library_folder_edit import LibraryFolderEdit  # noqa: E402

LIBRARY = SetupLibrary(label="Test library", packages=("tvendor_client",))


def _found(folder, version):
    return vl.LibraryFolder(folder, vl.SOURCE_FOUND, ("tvendor_client",), version)


@pytest.fixture
def edit(qapp):
    widget = LibraryFolderEdit(LIBRARY)
    yield widget
    widget.deleteLater()


def test_one_folder_found_fills_the_field(edit, monkeypatch):
    monkeypatch.setattr(vl, "search", lambda library: [_found("/a", "4.9")])
    edit.find()
    assert edit.text() == "/a"


def test_several_found_are_chosen_between(edit, monkeypatch):
    monkeypatch.setattr(
        vl, "search", lambda library: [_found("/new", "4.10"), _found("/old", None)]
    )
    offered = []

    def get_item(parent, title, label, items, current, editable):
        offered.extend(items)
        return items[1], True

    monkeypatch.setattr(library_folder_edit.QInputDialog, "getItem", get_item)
    edit.find()
    assert offered == ["/new (4.10)", "/old"]
    assert edit.text() == "/old"


def test_nothing_found_says_so_and_keeps_the_field(edit, monkeypatch):
    monkeypatch.setattr(vl, "search", lambda library: [])
    told = []
    monkeypatch.setattr(
        library_folder_edit.QMessageBox,
        "information",
        lambda parent, title, text: told.append(text),
    )
    edit.setText("/kept")
    edit.find()
    assert edit.text() == "/kept"
    assert "No Python on PATH has the Test library" in told[0]
