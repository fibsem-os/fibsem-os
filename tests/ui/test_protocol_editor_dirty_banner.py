"""The Protocol tab's status strip and its task tooltips.

Editing a task lit up a primary button under the task list that nobody looked at.
The strip below the columns says the same thing on the tab where the editing
happens. It is
always there at one height -- quiet while the lamellae carry the current settings,
lit with Apply once a task is edited -- so changing state moves and covers nothing.
The buttons stay for now as the second route to the same action.

The task rows stay plain text: what a task is and how the workflow runs it are
read in the tooltip and edited elsewhere.
"""

import copy
import os
import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402
from PyQt5.QtWidgets import QApplication, QSplitter, QWidget  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.autolamella_task_config_editor import (  # noqa: E402
    AutoLamellaProtocolTaskConfigEditor,
    _task_tooltips,
)
from fibsem.structures import MicroscopeState  # noqa: E402


@pytest.fixture
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def editor(qapp):
    import fibsem.applications.autolamella.protocol as protodir

    microscope, _ = utils.setup_session(manufacturer="Demo")
    exp = Experiment(path=pathlib.Path(tempfile.mkdtemp()), name="banner")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol.load(
        os.path.join(os.path.dirname(protodir.__file__), "task-protocol.yaml")
    )
    # lamellae created the way the app does it: carrying the protocol's configs
    for _ in range(2):
        exp.add_new_lamella(
            MicroscopeState(),
            EventedDict(copy.deepcopy(dict(exp.task_protocol.task_config))),
        )
    host = QWidget()
    host.experiment, host.microscope = exp, microscope
    widget = AutoLamellaProtocolTaskConfigEditor(parent=host)
    widget._host = host
    host.resize(1200, 700)
    host.show()
    qapp.processEvents()
    yield widget
    host.close()


def test_the_strip_is_always_there_and_changing_state_moves_nothing(editor, qapp):
    splitter = editor.findChild(QSplitter)
    strip = editor.dirty_banner
    assert not strip.isHidden()
    assert strip.apply_button.isHidden(), "nothing edited: no button"
    quiet_height, before = strip.height(), splitter.geometry()

    editor.task_list_widget.select("Rough Milling")
    editor._set_protocol_dirty(True)
    qapp.processEvents()

    assert not strip.apply_button.isHidden()
    assert strip.height() == quiet_height
    assert splitter.geometry() == before, "lighting the strip shifted the columns"
    assert strip.y() > splitter.y(), "the strip sits below the columns"

    editor._set_protocol_dirty(False)
    qapp.processEvents()
    assert strip.apply_button.isHidden()


def test_the_strip_says_only_what_the_edit_flag_knows(editor, qapp):
    """It says the task was edited and not applied. It does not claim which
    lamellae match, because nothing tracks that."""
    editor.task_list_widget.select("Rough Milling")
    editor._set_protocol_dirty(True)

    text = editor.dirty_banner.label.text()
    assert text == (
        "'Rough Milling' was edited but has not been applied to the 2 existing "
        "lamellae yet."
    )
    assert editor.dirty_banner.apply_button.text() == "Apply now"


def test_with_no_lamellae_the_strip_stays_quiet(editor, qapp):
    editor.experiment.positions.clear()
    editor._set_protocol_dirty(True)
    assert editor.dirty_banner.apply_button.isHidden()


def test_tooltips_say_what_a_task_is_and_how_the_workflow_runs_it(editor):
    protocol = editor.experiment.task_protocol
    tips = _task_tooltips(protocol)

    assert tips["Rough Milling"] == (
        "Rough Milling\nType: Mill Rough\nWorkflow: supervised, required, after Mill Fiducial"
    )
    assert editor.task_list_widget._list.item(2).toolTip() == tips["Rough Milling"]


def test_a_task_the_workflow_does_not_run_says_so():
    from fibsem.applications.autolamella.workflows.tasks.base import (
        AutoLamellaTaskConfig,
    )

    protocol = AutoLamellaTaskProtocol()
    protocol.task_config = {"Orphan": AutoLamellaTaskConfig(task_name="Orphan")}
    tips = _task_tooltips(protocol)
    assert tips["Orphan"].endswith("Workflow: not included")
