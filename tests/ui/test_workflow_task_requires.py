"""A task row states its requirements, and says nothing when it has none.

Every row used to carry a second line, and on most of them it read "No
requirements" — an absence restated once per task, which is what buried the two or
three rows that actually declare a dependency. Blank leaves those standing alone.

The line is emptied rather than hidden, so every row keeps the same height and the
list does not go ragged wherever a dependency happens to fall. Both halves are
pinned here, because the obvious "tidy-up" is to hide the label.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
)
from fibsem.applications.autolamella.ui.workflow_config_widget import (  # noqa: E402
    WorkflowTaskRowWidget,
)


def _row(qapp, name: str, requires: list[str]) -> WorkflowTaskRowWidget:
    task = AutoLamellaTaskDescription(
        name=name, supervise=False, required=True, requires=requires
    )
    return WorkflowTaskRowWidget(task)


def test_a_task_with_no_requirements_says_nothing(qapp):
    row = _row(qapp, "Mill Fiducial", [])

    assert row.requires_label.text() == ""

    row.deleteLater()


def test_a_task_with_requirements_names_them(qapp):
    row = _row(qapp, "Rough Milling", ["Mill Fiducial"])

    assert row.requires_label.text() == "Mill Fiducial"

    row.deleteLater()


def test_several_requirements_are_listed(qapp):
    row = _row(qapp, "Polishing", ["Mill Fiducial", "Rough Milling"])

    assert row.requires_label.text() == "Mill Fiducial, Rough Milling"

    row.deleteLater()


def test_rows_keep_the_same_height_either_way(qapp):
    """Emptied, not hidden. Hiding the label shortens the row it is on.

    A list whose rows change height depending on whether a task declares a
    dependency is harder to read than the repeated text this removes.
    """
    without = _row(qapp, "Mill Fiducial", [])
    with_one = _row(qapp, "Rough Milling", ["Mill Fiducial"])

    assert without.requires_label.isVisibleTo(without)
    assert without.sizeHint().height() == with_one.sizeHint().height()

    without.deleteLater()
    with_one.deleteLater()
