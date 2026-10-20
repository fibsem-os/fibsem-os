"""A grid renamed in the Sample view renames the experiment's record through
`rename_grid`, the Grids tab's and the App Server's rename (FIB-1446)."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow  # noqa: E402
from fibsem.structures import GridInventoryEntry, GridSlotState  # noqa: E402
from fibsem.workflows.grid.naming import GRIDS_CHANGED  # noqa: E402
from fibsem.workflows.grids import GridRecord  # noqa: E402


def _window(experiment, stage=None):
    refreshed = []
    window = SimpleNamespace(
        experiment=experiment,
        microscope=SimpleNamespace(sample_handling=stage),
        session=SimpleNamespace(
            record_event=lambda kind, payload: refreshed.append((kind, payload))
        ),
        grids_tab=SimpleNamespace(refresh=lambda: refreshed.append("grids")),
        workflow_tab=SimpleNamespace(
            grid_workflow_widget=SimpleNamespace(refresh=lambda: None)
        ),
        _refresh_grid_context=lambda: None,
    )
    return window, refreshed


def test_the_record_follows_the_slot(tmp_path):
    experiment = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    record = experiment.add_grid(GridRecord(name="grid-1"))
    window, refreshed = _window(experiment)

    AutoLamellaWindow._on_slot_grid_renamed(window, "grid-1", "grid-aspen")

    assert record.name == "grid-aspen"
    assert experiment.get_grid_by_name("grid-1") is None
    assert refreshed == [
        (GRIDS_CHANGED, {"renamed": "grid-1", "grid_name": "grid-aspen"}),
        "grids",
    ]


def test_a_grid_with_no_record_is_left_to_the_sync(tmp_path):
    experiment = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    window, refreshed = _window(experiment)

    AutoLamellaWindow._on_slot_grid_renamed(window, "grid-1", "grid-aspen")

    assert experiment.grids == [] and refreshed == []


def test_the_inventorys_grids_are_recorded_saved_and_on_the_record(tmp_path):
    """The Sample view's inventory read records the grids as the agent's
    add_grids does (``record_inventoried_grids``)."""
    experiment = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    entry = GridInventoryEntry(
        slot_name="Slot-01",
        index=0,
        source="magazine",
        name="G1",
        state=GridSlotState.OCCUPIED,
    )
    stage = SimpleNamespace(grid_inventory=lambda: [entry])
    window, recorded = _window(experiment, stage)

    AutoLamellaWindow._record_inventoried_grids(window)
    AutoLamellaWindow._record_inventoried_grids(window)

    assert [g.name for g in experiment.grids] == ["G1"]
    assert recorded == [(GRIDS_CHANGED, {"added": ["G1"]})]
    assert (tmp_path / "exp" / "experiment.yaml").exists()
