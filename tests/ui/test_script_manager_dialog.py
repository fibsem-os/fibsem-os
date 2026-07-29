"""The Scripts manager dialog, driven without an application around it (FIB-338)."""

import time
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
        self.stop_event = None  # the runner injects one for threaded scripts

    def save(self):
        self.saved = True

    def raise_if_cancelled(self):
        from fibsem.cancellation import raise_if_cancelled
        raise_if_cancelled(self.stop_event)


def _drain(qapp, predicate, timeout=5.0):
    """Pump the Qt event loop until predicate() is true, or give up.

    Threaded scripts report back through a queued signal, so nothing lands unless
    the loop runs.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    return predicate()


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    return path


def _dialog(tmp_path, context=None, reason="", notes=None, confirm=True):
    runner = ScriptRunner(
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (context, reason),
        notify=lambda m, l: (notes if notes is not None else []).append((l, m)),
        # the real one is a modal box; a stub keeps the tests from blocking on it
        confirm=lambda question, detail: confirm,
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
    # the heading says what the dialog is; the counts are meta beneath it
    assert dialog.title_label.text() == "User scripts"
    assert "1 script, 1 failed to load" in dialog.meta_label.text()


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
    # the consequence sits opposite the facts; the chip-length phrase is on
    # screen and the full sentence is in the tooltip
    assert "Modifies and saves" in dialog.consequence_label.text()
    assert "saves it when it finishes" in dialog.consequence_label.toolTip()


def test_a_read_only_script_says_so(qapp, tmp_path):
    """Silence would leave the user guessing whether it writes."""
    _write(tmp_path, "r.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)
    assert "Read-only" in dialog.consequence_label.text()


def test_an_empty_folder_says_how_to_start(qapp, tmp_path):
    dialog = _dialog(tmp_path, context=FakeContext())
    assert "New script" in dialog.detail_label.text()


def test_the_last_run_stamp_is_12_hour(qapp, tmp_path):
    _write(tmp_path, "s.py", "def run(ctx):\n    return 'ok'\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    dialog.run_selected()

    stamp = _cell_text(dialog, 0, 2)
    assert "am" in stamp or "pm" in stamp


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


def test_a_microscope_script_can_be_run_and_says_what_it_will_do(qapp, tmp_path):
    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    assert dialog.run_button.isEnabled()
    assert "Drives the microscope" in dialog.consequence_label.text()
    # the warning has to be specific about what is missing, not just "careful"
    assert "no limits, no interlocks" in dialog.consequence_label.toolTip()


def test_run_becomes_stop_while_a_script_is_running(qapp, tmp_path):
    """A microscope script runs for minutes. Without this the dialog looks idle
    and the only way to stop it is to kill the app."""
    _write(tmp_path, "slow.py",
           "uses_microscope = True\n"
           "def run(ctx):\n"
           "    ctx.stop_event.wait(3)\n"
           "    ctx.raise_if_cancelled()\n"
           "    return 'ran to completion'\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    dialog.run_selected()
    assert _drain(qapp, lambda: dialog.runner.is_running)
    assert dialog.run_button.text() == "Stop script"
    assert dialog.run_button.isEnabled()
    # nothing that would start a second run or move the ground under this one
    assert not dialog.table.isEnabled()
    assert not dialog.change_folder_button.isEnabled()

    dialog.run_selected()  # the same button, now Stop

    # not `not is_running` -- the thread dies before its result is delivered, so
    # that would pass while the dialog is still mid-teardown
    assert _drain(qapp, lambda: "slow" in dialog.last_run)
    assert dialog.run_button.text() == "Run script"
    assert dialog.table.isEnabled()
    assert "cancelled" in dialog.last_run["slow"]


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


def test_selection_survives_running_a_script(qapp, tmp_path):
    """Running one used to bounce the selection back to the top of the list."""
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path, name, "def run(ctx):\n    return 'ok'\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(_row_of(dialog, "b"))

    dialog.run_selected()

    assert dialog.selected_script().name == "b"


def test_cancelling_a_writes_confirmation_leaves_no_last_run(qapp, tmp_path):
    """A run that never happened must not read as one that did."""
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    context = FakeContext()
    dialog = _dialog(tmp_path, context=context, confirm=False)

    dialog.run_selected()

    assert dialog.last_run == {}
    assert _cell_text(dialog, 0, 2) == "—"
    assert not context.saved


def test_rebuilding_does_not_stack_cell_widgets(qapp, tmp_path):
    """setRowCount alone leaves the previous build's widgets parented to the
    viewport, so rows drew on top of each other."""
    from PyQt5.QtWidgets import QLabel

    _write(tmp_path, "a.py", '"""One."""\ndef run(ctx):\n    pass\n')
    dialog = _dialog(tmp_path, context=FakeContext())

    for _ in range(3):
        dialog.refresh()

    # name + description, not three builds' worth
    assert len(dialog.table.cellWidget(0, 0).findChildren(QLabel)) == 2


def test_change_folder_switches_the_listing(qapp, tmp_path):
    _write(tmp_path / "first", "one.py", "def run(ctx):\n    pass\n")
    _write(tmp_path / "second", "two.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path / "first", context=FakeContext())
    assert [s.name for s in dialog.scripts] == ["one"]

    dialog.runner.set_directory(tmp_path / "second")
    dialog.refresh(keep_selection=False)

    assert [s.name for s in dialog.scripts] == ["two"]


def test_new_script_creates_a_runnable_stub(qapp, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("fresh", True))
    monkeypatch.setattr(
        "fibsem.ui.widgets.script_manager_dialog.open_path_in_file_explorer",
        lambda path: None,
    )
    dialog = _dialog(tmp_path, context=FakeContext())

    dialog.new_script()

    assert (tmp_path / "fresh.py").exists()
    assert "fresh" in [s.name for s in dialog.scripts]
    # the stub must actually load, or the first thing a user sees is an error
    assert dialog.scripts[_row_of(dialog, "fresh")].is_runnable


def test_new_script_refuses_to_overwrite(qapp, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QInputDialog

    _write(tmp_path, "taken.py", "def run(ctx):\n    return 'original'\n")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("taken", True))
    notes = []
    dialog = _dialog(tmp_path, context=FakeContext(), notes=notes)

    dialog.new_script()

    assert "original" in (tmp_path / "taken.py").read_text()
    assert notes and notes[-1][0] == "warning"


def test_the_folder_is_elided_but_kept_whole_in_the_tooltip(qapp, tmp_path):
    """The path is far longer than anything else and would otherwise dictate the
    dialog's width."""
    deep = tmp_path / ("nested/" * 12).rstrip("/")
    _write(deep, "s.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(deep, context=FakeContext())
    dialog.show()

    assert "…" in dialog.meta_label.text()
    assert str(deep) == dialog.meta_label.toolTip()


def test_a_long_description_elides_instead_of_clipping(qapp, tmp_path):
    """It used to be cut off mid-glyph, which reads as a typo rather than as
    text continuing past the edge."""
    summary = "Archive the experiment folder and every image in it when a workflow finishes"
    _write(tmp_path, "a.py", f'"""{summary}"""\ndef run(ctx):\n    pass\n')
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.resize(520, 400)  # narrow enough that this description cannot fit
    dialog.grab()  # forces the paint pass that does the eliding

    from PyQt5.QtWidgets import QLabel
    shown = dialog.table.cellWidget(0, 0).findChildren(QLabel)[1].text()

    assert shown.endswith("…") and shown != summary
    assert summary in dialog.table.cellWidget(0, 0).toolTip()


def test_the_auto_flag_says_it_is_not_connected(qapp, tmp_path):
    """Nothing fires on_workflow_completed yet. An author who declared it has no
    other way to find that out, and a chip alone would read as a promise."""
    _write(tmp_path, "a.py", "on_workflow_completed = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    assert "off" in _cell_text(dialog, 0, 1)
    assert "not run automatically yet" in dialog.table.cellWidget(0, 1).toolTip()
    assert "not run automatically yet" in dialog.consequence_label.toolTip()


def test_the_menu_does_not_label_a_script_auto(qapp, tmp_path):
    """A menu has nowhere to explain the caveat, so it leaves the flag out."""
    from PyQt5.QtWidgets import QMenu

    from fibsem.ui.widgets.script_menu import ScriptMenuController

    _write(tmp_path, "a.py", "on_workflow_completed = True\ndef run(ctx):\n    pass\n")
    menu = QMenu()
    controller = ScriptMenuController(
        menu=menu,
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (FakeContext(), ""),
        notify=lambda m, l: None,
    )
    controller.rebuild()

    assert not any("auto" in a.text() for a in menu.actions())


def test_every_chip_carries_a_dot(qapp, tmp_path):
    """Type and flag chips looked like different kinds of thing for no readable
    reason. Colour separates them; shape should not."""
    _write(tmp_path, "s.py",
           "writes = True\non_workflow_completed = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    from PyQt5.QtWidgets import QLabel
    chips = dialog.table.cellWidget(0, 1).findChildren(QLabel)

    assert len(chips) == 3  # Data + writes + auto (off)
    assert all("9679" in chip.text() for chip in chips)


def test_the_type_column_fits_the_widest_row_of_chips(qapp, tmp_path):
    """A fixed width clips as soon as a script declares more than one flag, and
    ResizeToContents is no help -- it measures the item, not the cell widget."""
    _write(tmp_path, "plain.py", "def run(ctx):\n    pass\n")
    _write(tmp_path, "loaded.py",
           "writes = True\non_workflow_completed = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.show()

    widest = max(
        dialog.table.cellWidget(row, 1).sizeHint().width()
        for row in range(dialog.table.rowCount())
    )
    assert dialog.table.columnWidth(1) >= widest
