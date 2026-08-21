"""A task row states what it waits for, on the same line as its name.

Every row used to carry a second line, and on most it read "No requirements" — an
absence restated once per task, which buried the two or three rows that actually
declare a dependency. Blanking that line was not enough on its own: the name sits
at the top of a two-line block, so a row without a sub-line reads as a name
floating above a gap. One line puts every name on the same baseline.

The dependency list is elided to a fixed budget, because a task waiting on four
others produced a label wider than the row and Qt cut it off mid-word. The row's
tooltip carries the full list.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
)
from fibsem.applications.autolamella.ui.workflow_config_widget import (  # noqa: E402
    REQUIRES_MAX_WIDTH,
    WorkflowTaskRowWidget,
)

TAGS = re.compile(r"<[^>]+>")


def _row(name: str, requires: list[str]) -> WorkflowTaskRowWidget:
    return WorkflowTaskRowWidget(
        AutoLamellaTaskDescription(
            name=name, supervise=False, required=True, requires=requires
        )
    )


def _rendered(row: WorkflowTaskRowWidget) -> str:
    """What the label actually reads, with the markup stripped."""
    return TAGS.sub("", row.name_label.text())


def test_a_task_with_no_requirements_says_only_its_name(qapp):
    row = _row("Mill Fiducial", [])

    assert _rendered(row) == "Mill Fiducial"

    row.deleteLater()


def test_a_task_with_a_requirement_names_it_after_the_task(qapp):
    row = _row("Rough Milling", ["Mill Fiducial"])

    assert _rendered(row) == "Rough Milling after Mill Fiducial"

    row.deleteLater()


def test_several_requirements_are_listed(qapp):
    row = _row("Polishing", ["Mill Fiducial", "Rough Milling"])

    assert _rendered(row) == "Polishing after Mill Fiducial, Rough Milling"

    row.deleteLater()


def test_rows_sit_on_one_baseline_either_way(qapp):
    """The point of the single line: no row is taller, and no name is higher.

    Two labels stacked would put the name at the top of the block on every row
    that has no dependency, which is most of them.
    """
    without = _row("Mill Fiducial", [])
    with_one = _row("Rough Milling", ["Mill Fiducial"])

    assert without.sizeHint().height() == with_one.sizeHint().height()
    # One label, so there is no second line for the name to sit above.
    assert not hasattr(without, "requires_label")

    without.deleteLater()
    with_one.deleteLater()


def test_a_long_dependency_list_is_elided_not_clipped(qapp):
    """Four dependencies made the label wider than the row, and Qt cut it mid-word."""
    requires = [
        "Setup Lamella Position",
        "Spot Burn Fiducial",
        "Mill Fiducial",
        "Rough Milling",
    ]
    row = _row("Final Polishing Pass", requires)

    rendered = _rendered(row)
    assert "…" in rendered
    assert rendered.startswith("Final Polishing Pass after Setup Lamella Position")
    # Bounded by the budget plus the name, rather than by however long the list is.
    name_only = _row("Final Polishing Pass", [])
    assert (
        row.name_label.sizeHint().width()
        <= name_only.sizeHint().width() + REQUIRES_MAX_WIDTH
    )
    # Nothing is lost: the whole list is one hover away.
    assert row.toolTip() == "Requires: " + ", ".join(requires)

    row.deleteLater()
    name_only.deleteLater()


def test_a_task_name_cannot_inject_markup(qapp):
    """Names come from the protocol, and the label is rich text."""
    row = _row("Mill <b>Fiducial</b> & Co", [])

    assert "&lt;b&gt;" in row.name_label.text()
    assert "<b>" not in row.name_label.text()

    row.deleteLater()
