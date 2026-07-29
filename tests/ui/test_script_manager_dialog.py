"""The Scripts manager dialog, driven without an application around it (FIB-338)."""

from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from fibsem.ui.widgets.script_manager_dialog import ScriptManagerDialog  # noqa: E402
from fibsem.ui.widgets.script_runner import ScriptRunner  # noqa: E402


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


class FakeContext:
    def __init__(self):
        self.saved = False

    def save(self):
        self.saved = True


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    return path


def _dialog(tmp_path, context=None, reason="", notes=None):
    runner = ScriptRunner(
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (context, reason),
        notify=lambda m, l: (notes if notes is not None else []).append((l, m)),
    )
    return ScriptManagerDialog(runner=runner)


def _row_of(dialog, name: str) -> int:
    return [s.name for s in dialog.scripts].index(name)


def _cell_text(dialog, row: int, col: int) -> str:
    """Read a cell whether it holds a plain item or a widget.

    Name and type cells are widgets, because a QTableWidgetItem cannot carry a
    two-line layout or a pill background.
    """
    from PyQt5.QtWidgets import QLabel

    widget = dialog.table.cellWidget(row, col)
    if widget is not None:
        return " ".join(label.text() for label in widget.findChildren(QLabel))
    item = dialog.table.item(row, col)
    return item.text() if item is not None else ""


def test_lists_runnable_and_failed_together(qapp, tmp_path):
    """The failed one is a row, not an omission -- a menu has nowhere to put the
    reason, which is the whole point of this dialog existing."""
    _write(tmp_path, "good.py", '"""Fine."""\ndef run(ctx):\n    pass\n')
    _write(tmp_path, "bad.py", "def main(ctx):\n    pass\n")

    dialog = _dialog(tmp_path, context=FakeContext())

    assert dialog.table.rowCount() == 2
    assert dialog.title_label.text() == "1 script, 1 failed to load"


def test_the_failure_reason_is_shown_in_the_row(qapp, tmp_path):
    _write(tmp_path, "bad.py", "def main(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    assert "run()" in _cell_text(dialog, 0, 0)
    assert "Error" in _cell_text(dialog, 0, 1)


def test_flags_appear_in_the_type_cell(qapp, tmp_path):
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    assert "writes" in _cell_text(dialog, 0, 1)


def test_a_writing_script_warns_in_the_detail_panel(qapp, tmp_path):
    """The consequence has to be stated before Run is pressed."""
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)
    assert "saves it when finished" in dialog.detail_label.text()


def test_source_and_hash_are_shown(qapp, tmp_path):
    """Users edit these in place, so 'which version ran' is otherwise unanswerable."""
    _write(tmp_path, "s.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)
    assert "s.py" in dialog.detail_label.text()
    assert dialog.scripts[0].content_hash in dialog.detail_label.text()


def test_run_is_disabled_for_a_broken_script(qapp, tmp_path):
    _write(tmp_path, "bad.py", "def main(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)
    assert not dialog.run_button.isEnabled()


def test_run_is_disabled_for_a_microscope_script(qapp, tmp_path):
    """The runner would refuse it (FIB-340); a button that only complains is worse
    than one that is visibly unavailable."""
    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)
    assert not dialog.run_button.isEnabled()


def test_run_is_disabled_and_explained_when_the_host_is_not_ready(qapp, tmp_path):
    _write(tmp_path, "s.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=None, reason="Load an experiment")
    dialog.table.selectRow(0)
    assert not dialog.run_button.isEnabled()
    assert dialog.hint_label.text() == "Load an experiment"


def test_running_records_the_outcome_in_the_last_run_column(qapp, tmp_path):
    _write(tmp_path, "s.py", "def run(ctx):\n    return 'done'\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    dialog.run_selected()

    assert "ok" in _cell_text(dialog, 0, 2)


def test_a_failing_run_is_recorded_as_failed(qapp, tmp_path):
    _write(tmp_path, "s.py", "def run(ctx):\n    raise RuntimeError('nope')\n")
    notes = []
    dialog = _dialog(tmp_path, context=FakeContext(), notes=notes)
    dialog.table.selectRow(0)

    dialog.run_selected()

    assert "failed" in _cell_text(dialog, 0, 2)
    assert notes and notes[-1][0] == "error"


def test_rescan_picks_up_a_new_file(qapp, tmp_path):
    _write(tmp_path, "one.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    assert dialog.table.rowCount() == 1

    _write(tmp_path, "two.py", "def run(ctx):\n    pass\n")
    dialog.refresh()

    assert dialog.table.rowCount() == 2


def test_empty_folder_renders_without_a_selection(qapp, tmp_path):
    dialog = _dialog(tmp_path, context=FakeContext())
    assert dialog.table.rowCount() == 0
    assert dialog.selected_script() is None
    assert not dialog.run_button.isEnabled()


def test_the_row_tooltip_carries_the_full_path(qapp, tmp_path):
    """The header shows the folder and the detail panel the filename, so the
    absolute path is the one thing you otherwise cannot read off the dialog --
    and it is what you need in order to go and edit the file."""
    path = _write(tmp_path, "s.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    tooltip = dialog.table.cellWidget(0, 0).toolTip()

    assert str(path) in tooltip
    assert dialog.scripts[0].content_hash in tooltip


def test_the_type_tooltip_spells_out_the_flags(qapp, tmp_path):
    """Chips are shorthand; the tooltip says what they actually mean."""
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    tooltip = dialog.table.cellWidget(0, 1).toolTip()

    assert "saves the experiment" in tooltip
