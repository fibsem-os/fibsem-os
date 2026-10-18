"""A grid renamed in the Sample view renames the experiment's record through
`rename_grid`, the Grids tab's and the App Server's rename (FIB-1446)."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    AutoLamellaSingleWindowUI,
)
from fibsem.workflows.grids import GridRecord  # noqa: E402


def _window(experiment):
    refreshed = []
    window = SimpleNamespace(
        autolamella_ui=SimpleNamespace(
            experiment=experiment, microscope=SimpleNamespace(sample_handling=None)
        ),
        grids_tab=SimpleNamespace(refresh=lambda: refreshed.append("grids")),
        grid_workflow_widget=SimpleNamespace(refresh=lambda: None),
        _refresh_grid_context=lambda: None,
    )
    return window, refreshed


def test_the_record_follows_the_slot(tmp_path):
    experiment = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    record = experiment.add_grid(GridRecord(name="grid-1"))
    window, refreshed = _window(experiment)

    AutoLamellaSingleWindowUI._on_slot_grid_renamed(window, "grid-1", "grid-aspen")

    assert record.name == "grid-aspen"
    assert experiment.get_grid_by_name("grid-1") is None
    assert refreshed == ["grids"]


def test_a_grid_with_no_record_is_left_to_the_sync(tmp_path):
    experiment = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    window, refreshed = _window(experiment)

    AutoLamellaSingleWindowUI._on_slot_grid_renamed(window, "grid-1", "grid-aspen")

    assert experiment.grids == [] and refreshed == []
