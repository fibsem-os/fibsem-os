"""The workflow settings popover's run options (FIB-1376: Show summary after
completion). The checkboxes write the options, and show what is set when
the popover opens, since the agent can change an option while it is closed."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import AutoLamellaWorkflowOptions
from fibsem.applications.autolamella.ui.workflow.lamella_run import WorkflowInfoWidget


def test_the_summary_option_is_on_by_default_and_the_checkbox_writes_it(qapp):
    options = AutoLamellaWorkflowOptions()
    widget = WorkflowInfoWidget()
    try:
        widget.set_options(options)
        assert widget.show_run_summary_cb.isChecked()

        widget.show_run_summary_cb.setChecked(False)
        assert options.show_run_summary is False
        assert options.turn_beams_off is False  # untouched
    finally:
        widget.deleteLater()


def test_opening_shows_what_the_agent_set(qapp):
    options = AutoLamellaWorkflowOptions()
    widget = WorkflowInfoWidget()
    try:
        widget.set_options(options)
        options.show_run_summary = False  # set_workflow_options, while closed
        widget.show()
        assert not widget.show_run_summary_cb.isChecked()
    finally:
        widget.close()
        widget.deleteLater()


def test_an_old_protocol_without_the_option_shows_the_summary():
    options = AutoLamellaWorkflowOptions.from_dict({"turn_beams_off": True})
    assert options.show_run_summary is True
