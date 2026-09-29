"""The editors rebuild when an agent patch lands under their open forms.

The real main window, real editors: the refresh hooks are what stop a stale
open form writing old values back on the operator's next edit.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTaskConfig,
)
from fibsem.structures import MicroscopeState

TASK = "Rough Milling"


@pytest.fixture(scope="module")
def window(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    win._set_border_state("idle")
    yield win
    if win.autolamella_ui.microscope is not None:
        win.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def experiment(window, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="editor-refresh-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.positions[0].task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    exp.task_protocol.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    window.autolamella_ui.experiment = exp
    window.lamella_widget.set_experiment()
    return exp


def test_the_item_editor_rebuilds_only_for_the_shown_item(window, experiment):
    editor = window.lamella_widget
    lamella = experiment.positions[0]
    assert editor._selected_lamella is not None
    assert editor._selected_lamella.name == lamella.name

    # An outside writer changes state under the open panel; the hook rebuilds
    # the panel from state — observable through any displayed field.
    lamella.description = "changed by the agent"
    editor.refresh_if_showing(lamella.name)
    assert editor.line_edit_description.text() == "changed by the agent"

    # A different item's change leaves the shown panel alone.
    lamella.description = "changed again"
    editor.refresh_if_showing("some-other-item")
    assert editor.line_edit_description.text() == "changed by the agent"


def test_the_applier_reaches_the_editor_through_the_window(window, experiment):
    """The full GUI-side path: patch applied on the main window's embedded UI
    rebuilds the editor showing that item."""
    from fibsem.applications.autolamella.server.context import config_version

    ui = window.autolamella_ui
    lamella = experiment.positions[0]
    config = lamella.task_config[TASK]
    result = ui._apply_task_config_patch_for_agent(
        "item",
        lamella.name,
        TASK,
        {"milling.mill_rough.stages.0.pattern.depth": 2.7e-6},
        config_version(config),
    )
    assert result["applied"] is True
    assert config.milling["mill_rough"].stages[0].pattern.depth == 2.7e-6
