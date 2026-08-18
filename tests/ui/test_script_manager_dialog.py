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
    assert "Writes" in _cell_text(dialog, 0, 1)


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


def test_the_type_column_survives_a_folder_with_no_chips(qapp, tmp_path):
    """Dropping the Data chip meant an all-plain folder had nothing to measure, so
    the column collapsed to 28px and clipped its own header."""
    _write(tmp_path, "a.py", "def run(ctx):\n    pass\n")
    _write(tmp_path, "b.py", "def run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    header = dialog.table.horizontalHeader()
    assert dialog.table.columnWidth(1) >= header.fontMetrics().horizontalAdvance("Type")


def test_a_plain_data_script_gets_no_type_chip(qapp, tmp_path):
    """Data is the default, so a chip for it sat on nearly every row and separated
    nothing. An empty Type cell means "touches nothing unusual"; the word survives
    in the tooltip, where it costs no width."""
    _write(tmp_path, "plain.py", '"""Read some numbers."""\ndef run(ctx):\n    pass\n')
    dialog = _dialog(tmp_path, context=FakeContext())

    from PyQt5.QtWidgets import QLabel
    cell = dialog.table.cellWidget(0, 1)

    assert cell.findChildren(QLabel) == []
    assert "Type: Data" in cell.toolTip()


def test_a_microscope_script_that_also_writes_admits_to_both(qapp, tmp_path):
    """Only the worse of the two fits on the line, so the tooltip is the only place
    the second half can be stated -- and it was silently dropped."""
    _write(tmp_path, "both.py",
           "uses_microscope = True\nwrites = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    assert "Controls the microscope" in dialog.consequence_label.text()
    tip = dialog.consequence_label.toolTip()
    assert "no limits, no interlocks" in tip
    assert "saves it when it finishes" in tip


def test_the_consequence_line_matches_its_chip(qapp, tmp_path):
    """One script, one colour. The line under the table used to be red while the
    row's chip was amber, so the same fact arrived twice in two severities."""
    from fibsem.ui.widgets.script_manager_dialog import _MICROSCOPE, _WRITES

    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    pass\n")
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    dialog.table.selectRow(0)  # hw.py
    assert _MICROSCOPE in dialog.consequence_label.text()
    dialog.table.selectRow(1)  # w.py
    assert _WRITES in dialog.consequence_label.text()


def test_only_a_broken_script_is_red(qapp, tmp_path):
    """Colour in the list says what a script *is*; the confirmation dialog is what
    warns. Red left for the one thing that is actionable while browsing."""
    from fibsem.ui.widgets.script_manager_dialog import _ERROR

    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    pass\n")
    _write(tmp_path, "oops.py", "def main(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    dialog.table.selectRow(0)  # hw.py -- the most dangerous, still not red
    assert _ERROR not in dialog.consequence_label.text()
    dialog.table.selectRow(1)  # oops.py -- cannot load
    assert _ERROR in dialog.consequence_label.text()


def test_an_empty_folder_says_how_to_start_where_you_are_looking(qapp, tmp_path):
    """The message replaces the table rather than sitting under 450px of empty grid
    with the one useful sentence stranded in the detail panel at the bottom."""
    dialog = _dialog(tmp_path, context=FakeContext())

    from PyQt5.QtWidgets import QLabel
    shown = dialog.stack.currentWidget()
    text = " ".join(label.text() for label in shown.findChildren(QLabel))

    assert shown is not dialog.table
    assert "No scripts in this folder" in text
    assert "New script" in text and "Change folder" in text


def test_the_empty_state_points_at_the_worked_examples(qapp, tmp_path):
    """examples/scripts/ is not in the wheel, so a pip-installed user has no local
    copy and no way to know three of them exist. Saying "repository" is the point:
    without it the only advice is to write one from scratch."""
    from fibsem.ui.widgets.script_manager_dialog import EXAMPLES_PATH

    dialog = _dialog(tmp_path, context=FakeContext())

    from PyQt5.QtWidgets import QLabel
    text = " ".join(
        label.text() for label in dialog.stack.currentWidget().findChildren(QLabel)
    )

    assert EXAMPLES_PATH in text
    assert "repository" in text


def test_the_empty_message_is_not_also_repeated_in_the_detail_panel(qapp, tmp_path):
    """It used to be the only place it appeared; now it would be the second. The
    panel hides rather than sitting there as an empty bordered strip."""
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.show()

    assert "New script" not in dialog.detail_label.text()
    assert not dialog.detail_panel.isVisible()

    _write(tmp_path, "s.py", "def run(ctx):\n    pass\n")
    dialog.refresh()
    assert dialog.detail_panel.isVisible()


def test_the_table_comes_back_once_a_script_exists(qapp, tmp_path):
    dialog = _dialog(tmp_path, context=FakeContext())
    assert dialog.stack.currentWidget() is not dialog.table

    _write(tmp_path, "s.py", "def run(ctx):\n    pass\n")
    dialog.refresh()

    assert dialog.stack.currentWidget() is dialog.table


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
    assert "Controls the microscope" in dialog.consequence_label.text()
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

    assert "original" in (tmp_path / "taken.py").read_text(encoding="utf-8")
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
    text continuing past the edge.

    The description has to beat the Script column at the dialog's *minimum* width,
    not at the width passed to resize() -- a layout minimum of ~720px wins over a
    smaller request, so a merely long sentence stopped eliding the moment the Type
    column got narrower.
    """
    summary = (
        "Archive the experiment folder and every image in it when a workflow "
        "finishes, then upload the archive, verify its checksum, and email a "
        "summary to whoever started the run"
    )
    _write(tmp_path, "a.py", f'"""{summary}"""\ndef run(ctx):\n    pass\n')
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.resize(520, 400)  # narrow enough that this description cannot fit
    dialog.grab()  # forces the paint pass that does the eliding

    from PyQt5.QtWidgets import QLabel
    label = dialog.table.cellWidget(0, 0).findChildren(QLabel)[1]
    # `QLabel.text`, not `ElidedLabel.text`: the latter returns what was set, so that a
    # caller can tell whether a line is theirs to overwrite. What is *drawn* is the
    # elided string, and eliding rather than clipping is what this is about.
    shown = QLabel.text(label)

    assert shown.endswith("…") and shown != summary
    assert label.text() == summary, "the label should still know its full text"
    # A child's tooltip shadows its parent's, and the row's carries the path and hash
    # as well as the summary.
    assert label.toolTip() == ""
    assert summary in dialog.table.cellWidget(0, 0).toolTip()


def test_the_auto_flag_gets_no_chip_but_is_still_explained(qapp, tmp_path):
    """Nothing fires on_workflow_completed yet, so a chip for it was a badge for a
    feature that does not exist. The author who declared it still needs somewhere to
    learn that, and a tooltip costs no width."""
    _write(tmp_path, "a.py", "on_workflow_completed = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())
    dialog.table.selectRow(0)

    from PyQt5.QtWidgets import QLabel
    assert dialog.table.cellWidget(0, 1).findChildren(QLabel) == []
    assert "not run automatically yet" in dialog.table.cellWidget(0, 1).toolTip()
    assert "not run automatically yet" in dialog.consequence_label.toolTip()


def test_chips_are_text_only(qapp, tmp_path):
    """The dot separated nothing once every chip carried one, and it cost width in
    a column that has to fit more than one. Colour and the pill do the work."""
    _write(tmp_path, "s.py",
           "writes = True\non_workflow_completed = True\ndef run(ctx):\n    pass\n")
    dialog = _dialog(tmp_path, context=FakeContext())

    from PyQt5.QtWidgets import QLabel
    chips = dialog.table.cellWidget(0, 1).findChildren(QLabel)

    assert [chip.text() for chip in chips] == ["Writes"]


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


def test_no_widget_swallows_its_own_tooltip(qapp, tmp_path):
    """A selector-less stylesheet on a widget also styles the QToolTip over it.

    It beats the application sheet because it sits nearer, so
    ``background: transparent`` on a table cell leaves that cell's tooltip as
    floating text with no panel behind it -- unreadable against whatever is
    underneath. The four tooltips this dialog sets were all affected.

    Asserted over every widget rather than the four known ones: the trap is that
    a widget acquires a tooltip later, and nothing about ``setStyleSheet`` warns
    you. ``_style()`` restates the tooltip rule; rules that name their own type
    do not leak and are left alone.

    Not verifiable by rendering -- the offscreen platform never realises a
    tooltip window, so QTipLabel is absent from topLevelWidgets().
    """
    from PyQt5.QtWidgets import QWidget

    _write(tmp_path, "demo.py", '"""A demo script.\n\nReads the experiment.\n"""\ndef run(ctx):\n    pass\n')
    dialog = _dialog(tmp_path, context=FakeContext())

    offenders = [
        (type(w).__name__, w.styleSheet()[:60])
        for w in dialog.findChildren(QWidget) + [dialog]
        if w.styleSheet().strip() and "{" not in w.styleSheet()
    ]

    assert offenders == []
