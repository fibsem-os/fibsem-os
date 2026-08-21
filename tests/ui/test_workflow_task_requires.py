"""A task row states what it waits for, in a column of its own.

Every row used to carry a second line, and on most it read "No requirements" — an
absence restated once per task, which buried the two or three rows that actually
declare a dependency. Blanking that line was not enough on its own: the name sits
at the top of a two-line block, so a row without a sub-line reads as a name
floating above a gap.

One line each, and the dependency in a fixed-width right-aligned column, so the
entries end on the same edge and read downwards as a column rather than being
hunted for wherever each name happens to stop.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
)
from fibsem.applications.autolamella.ui.workflow_config_widget import (  # noqa: E402
    REQUIRES_MAX_WIDTH,
    WorkflowTaskRowWidget,
)


def _row(name: str, requires: list[str]) -> WorkflowTaskRowWidget:
    return WorkflowTaskRowWidget(
        AutoLamellaTaskDescription(
            name=name, supervise=False, required=True, requires=requires
        )
    )


def test_a_task_with_no_requirements_leaves_the_column_empty(qapp):
    row = _row("Mill Fiducial", [])

    assert row.name_label.text() == "Mill Fiducial"
    assert row.requires_label.text() == ""

    row.deleteLater()


def test_a_task_with_a_requirement_names_it(qapp):
    row = _row("Rough Milling", ["Mill Fiducial"])

    assert row.name_label.text() == "Rough Milling"
    assert row.requires_label.text() == "after Mill Fiducial"

    row.deleteLater()


def test_several_requirements_are_listed(qapp):
    row = _row("Polishing", ["Mill Fiducial", "Rough Milling"])

    assert row.requires_label.text() == "after Mill Fiducial, Rough Milling"

    row.deleteLater()


def test_the_column_is_the_same_width_and_edge_on_every_row(qapp):
    """The point of the column: entries end on one edge, so they read downwards.

    A suffix that simply followed the name would start and end wherever that name
    happened to stop, which is what makes a list of them hard to scan.
    """
    rows = [
        _row("Mill Fiducial", []),
        _row("Rough Milling", ["Mill Fiducial"]),
        _row("A Considerably Longer Task Name", ["Rough Milling"]),
    ]

    assert {r.requires_label.width() for r in rows} == {REQUIRES_MAX_WIDTH}
    assert all(r.requires_label.alignment() & Qt.AlignRight for r in rows)
    # And no row is taller than another for having something in the column.
    assert len({r.sizeHint().height() for r in rows}) == 1

    for r in rows:
        r.deleteLater()


def test_a_long_dependency_list_is_elided_not_clipped(qapp):
    """Four dependencies overflowed the row, and Qt cut the text off mid-word."""
    requires = [
        "Setup Lamella Position",
        "Spot Burn Fiducial",
        "Mill Fiducial",
        "Rough Milling",
    ]
    row = _row("Final Polishing Pass", requires)

    text = row.requires_label.text()
    assert "…" in text
    assert text.startswith("after Setup Lamella Position")
    # Nothing is lost: the whole list is one hover away.
    assert row.toolTip() == "Requires: " + ", ".join(requires)

    row.deleteLater()


def test_a_task_name_cannot_render_as_markup(qapp):
    """Names come from the protocol, and QLabel on AutoText would interpret them."""
    row = _row("Mill <b>Fiducial</b> & Co", [])

    assert row.name_label.textFormat() == Qt.PlainText
    assert row.name_label.text() == "Mill <b>Fiducial</b> & Co"

    row.deleteLater()
