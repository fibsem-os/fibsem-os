"""The agent's start and requeue pass the same grid gate as the Run click and Add
to queue: a lamella whose grid is not on the stage is refused, by name, before
anything moves (FIB-71). See `workflows/grid_gate.py`."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.manager import (  # noqa: E402
    TaskManager,
)
from fibsem.structures import MicroscopeState  # noqa: E402

TASK_NAME = "Mill Fiducial"


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    import fibsem.config as fibsem_config

    arctis_config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    widget = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: arctis_config,
    )
    widget.system_widget.connect_to_microscope()

    experiment = Experiment(path=tmp_path, name="agent-grid-gate")
    (tmp_path / "agent-grid-gate").mkdir()
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name=TASK_NAME, required=True)
    )
    experiment.sync_grids_from_inventory(widget.microscope.sample_handling)
    grid = experiment.get_grid_by_name("Grid-02")
    experiment.add_new_lamella(
        microscope_state=MicroscopeState(),
        task_config=experiment.task_protocol.task_config,
        name="on-two",
        grid_id=grid.id,
    )
    widget.experiment = experiment

    yield widget
    widget._task_manager = None
    widget.experiment = None
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_the_agent_s_start_is_refused_while_the_grid_is_off_the_stage(ui, monkeypatch):
    started = []
    monkeypatch.setattr(
        ui, "_start_run_workflow_thread", lambda *args: started.append(args)
    )
    assert ui.microscope.sample_handling.loaded_grids == []

    refused = ui._start_workflow_for_agent([TASK_NAME], ["on-two"])
    assert refused["started"] is False
    assert "Grid-02" in refused["reason"] and "on-two" in refused["reason"]
    assert started == []

    # Every item, by omission, is gated the same way.
    assert ui._start_workflow_for_agent([TASK_NAME], None)["started"] is False
    assert started == []

    ui.microscope.sample_handling.ensure_loaded("Grid-02")
    ui._start_workflow_for_agent([TASK_NAME], ["on-two"])
    assert started == [([TASK_NAME], ["on-two"])]


def test_the_agent_s_requeue_is_refused_while_the_grid_is_off_the_stage(ui):
    manager = TaskManager(ui.microscope, ui.experiment, parent_ui=None)
    manager.queue.build_from_matrix([TASK_NAME], ["on-two"])
    ui._task_manager = manager
    ctx = AutoLamellaAppContext(ui)
    before = len(manager.queue.items)

    refused = ctx.requeue_task("on-two", TASK_NAME)
    assert refused["queued"] is False
    assert "Grid-02" in refused["reason"]
    assert len(manager.queue.items) == before

    ui.microscope.sample_handling.ensure_loaded("Grid-02")
    assert ctx.requeue_task("on-two", TASK_NAME)["queued"] is True
    assert len(manager.queue.items) == before + 1
