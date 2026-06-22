"""Tests for the grid workflow Phase 1: GridRecord, Experiment.grids, and the
GridTask lifecycle (pre_task -> _run -> post_task)."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from fibsem import utils
from fibsem.microscopes._stage import SampleGrid, _create_sample_stage
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    GridTaskDescription,
    GridWorkflowConfig,
    GridTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    GRID_TASK_REGISTRY,
    AcquireImageGridTaskConfig,
    AcquireImageTask,
    AcquireOverviewImageGridTask,
    AcquireOverviewImageGridTaskConfig,
    CryoCleaningGridTaskConfig,
    GridTask,
    GridTaskConfig,
    load_grid_task_config,
    run_grid_task,
)
from fibsem.applications.autolamella.workflows.tasks.grid_manager import (
    GridExchangeError,
    GridTaskManager,
    run_grid_workflow,
)


# --- helpers ---------------------------------------------------------------

@dataclass
class _NoOpGridConfig(GridTaskConfig):
    task_type: ClassVar[str] = "NOOP_GRID"
    display_name: ClassVar[str] = "No-op"


class _NoOpGridTask(GridTask):
    config_cls = _NoOpGridConfig

    def _run(self):
        pass


class _BoomGridTask(GridTask):
    config_cls = _NoOpGridConfig

    def _run(self):
        raise RuntimeError("boom")


@dataclass
class _MutatingGridConfig(GridTaskConfig):
    task_type: ClassVar[str] = "MUTATING_GRID"
    display_name: ClassVar[str] = "Mutating"
    marker: int = 0


class _MutatingGridTask(GridTask):
    config_cls = _MutatingGridConfig

    def _run(self):
        self.config.marker += 1  # mutate the config (simulates hfw/path writeback)


class _ResultGridTask(GridTask):
    config_cls = _NoOpGridConfig

    def _run(self):
        self.record_result(overview="/tmp/overview.tif", pixel_size=1.2e-8)


@pytest.fixture
def demo_microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    # build a deterministic 2-slot, non-compustage holder (no loader) so the grid
    # tests don't depend on the machine's default config / suite-order state.
    from fibsem.microscopes._stage import SampleHolder, Stage

    microscope.stage_is_compustage = False
    holder = SampleHolder(name="Test Holder", capacity=2)
    holder._ensure_slots()
    holder._parent = microscope
    microscope._stage = Stage(parent=microscope, holder=holder, loader=None)
    return microscope


@pytest.fixture
def experiment(tmp_path):
    return Experiment.create(path=str(tmp_path), name="exp-grid-test")


# --- GridRecord ------------------------------------------------------------

def test_grid_record_round_trip():
    g = GridRecord(name="grid-aspen")
    g.task_state.name = "NOOP_GRID"
    g.task_state.status = AutoLamellaTaskStatus.Completed
    g.task_history.append(g.task_state)

    g2 = GridRecord.from_dict(g.to_dict())

    assert g2.name == "grid-aspen"
    assert g2._id == g._id
    assert g2.task_state.status is AutoLamellaTaskStatus.Completed
    assert g2.completed_tasks == ["NOOP_GRID"]
    assert g2.has_completed_task("NOOP_GRID")
    assert not g2.is_failure


def test_grid_record_is_failure():
    g = GridRecord(name="grid-x")
    assert not g.is_failure
    g.task_state.status = AutoLamellaTaskStatus.Failed
    assert g.is_failure


# --- Experiment.grids ------------------------------------------------------

def test_add_grid_and_lookup(experiment):
    experiment.add_grid(GridRecord(name="grid-aspen"))
    assert experiment.get_grid_by_name("grid-aspen") is not None
    assert experiment.get_grid_by_name("missing") is None


def test_add_grid_rejects_duplicate(experiment):
    experiment.add_grid(GridRecord(name="grid-aspen"))
    with pytest.raises(ValueError):
        experiment.add_grid(GridRecord(name="grid-aspen"))


def test_add_grid_type_checked(experiment):
    with pytest.raises(TypeError):
        experiment.add_grid("not-a-grid")


def test_grids_persist_and_reload(experiment):
    experiment.add_grid(GridRecord(name="grid-aspen"))
    experiment.add_grid(GridRecord(name="grid-birch"))
    experiment.save()

    loaded = Experiment.load(f"{experiment.path}/experiment.yaml")
    assert [g.name for g in loaded.grids] == ["grid-aspen", "grid-birch"]


def test_from_dict_back_compat_without_grids_key(experiment):
    # an experiment dict predating the grids field must still load
    ddict = experiment.to_dict()
    del ddict["grids"]
    restored = Experiment.from_dict(ddict)
    assert list(restored.grids) == []


# --- sync_grids_from_holder ------------------------------------------------

def test_sync_grids_from_holder_is_idempotent(demo_microscope, experiment):
    slot_names = list(demo_microscope._stage.holder.slots.keys())[:2]
    demo_microscope._stage.holder.slots[slot_names[0]].loaded_grid = SampleGrid(name="grid-aspen")
    demo_microscope._stage.holder.slots[slot_names[1]].loaded_grid = SampleGrid(name="grid-birch")

    experiment.sync_grids_from_holder(demo_microscope)
    experiment.sync_grids_from_holder(demo_microscope)  # must not duplicate

    assert [g.name for g in experiment.grids] == ["grid-aspen", "grid-birch"]


# --- GridWorkflowConfig (run order + supervise) ----------------------------

def test_grid_workflow_config_sync_to_adds_drops_preserves():
    wf = GridWorkflowConfig(tasks=[
        GridTaskDescription(name="a", supervise=True),
        GridTaskDescription(name="b"),
    ])
    # 'b' dropped, 'c' appended; 'a' keeps its order + supervise flag
    wf.sync_to(["a", "c"])
    assert wf.order == ["a", "c"]
    assert wf.get("a").supervise is True
    assert wf.get("c").supervise is False


def test_grid_workflow_config_roundtrip():
    wf = GridWorkflowConfig(tasks=[
        GridTaskDescription(name="Overview", supervise=True),
        GridTaskDescription(name="Acquire Image"),
    ])
    restored = GridWorkflowConfig.from_dict(wf.to_dict())
    assert restored.order == ["Overview", "Acquire Image"]
    assert restored.get("Overview").supervise is True


def test_protocol_reconcile_mirrors_task_config():
    proto = GridTaskProtocol()
    proto.task_config["Overview"] = AcquireOverviewImageGridTaskConfig(task_name="Overview")
    proto.task_config["Clean"] = CryoCleaningGridTaskConfig(task_name="Clean")
    proto.reconcile_workflow()
    assert proto.workflow_config.order == ["Overview", "Clean"]


def test_protocol_workflow_persists_and_reloads():
    proto = GridTaskProtocol()
    proto.task_config["Overview"] = AcquireOverviewImageGridTaskConfig(task_name="Overview")
    proto.task_config["Clean"] = CryoCleaningGridTaskConfig(task_name="Clean")
    proto.reconcile_workflow()
    # reorder + supervise, then roundtrip
    proto.workflow_config.tasks = list(reversed(proto.workflow_config.tasks))
    proto.workflow_config.get("Overview").supervise = True
    restored = GridTaskProtocol.from_dict(proto.to_dict())
    assert restored.workflow_config.order == ["Clean", "Overview"]
    assert restored.workflow_config.get("Overview").supervise is True


def test_protocol_back_compat_without_workflow_config():
    # an old protocol dict (no workflow_config key) reconciles order from configs
    proto = GridTaskProtocol()
    proto.task_config["Overview"] = AcquireOverviewImageGridTaskConfig(task_name="Overview")
    ddict = proto.to_dict()
    del ddict["workflow_config"]
    restored = GridTaskProtocol.from_dict(ddict)
    assert restored.workflow_config.order == ["Overview"]


# --- task user-interaction primitives (ask_user / update_status_ui) --------

def test_grid_task_validate_reads_supervise(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="T"), record, experiment)
    # no workflow description → not supervised
    assert task.validate is False
    experiment.grid_protocol.workflow_config.tasks = [
        GridTaskDescription(name="T", supervise=True)
    ]
    assert task.validate is True
    experiment.grid_protocol.workflow_config.get("T").supervise = False
    assert task.validate is False


def test_grid_task_ask_user_gated_by_supervise(demo_microscope, experiment, monkeypatch):
    import fibsem.applications.autolamella.workflows.ui as wfui
    calls = []
    monkeypatch.setattr(wfui, "ask_user", lambda **kw: (calls.append(kw), "RESPONSE")[1])

    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="T"),
                         record, experiment, parent_ui=object())

    # not supervised → auto-continue (True), module ask_user not called
    assert task.ask_user("Go?") is True
    assert calls == []

    # supervised → delegates to the module ask_user with the message/buttons
    experiment.grid_protocol.workflow_config.tasks = [
        GridTaskDescription(name="T", supervise=True)
    ]
    assert task.ask_user("Go?", pos="Start", neg="Skip") == "RESPONSE"
    assert calls[-1]["msg"] == "Go?" and calls[-1]["pos"] == "Start" and calls[-1]["neg"] == "Skip"


def test_grid_task_update_status_ui_prefixes(demo_microscope, experiment, monkeypatch):
    import fibsem.applications.autolamella.workflows.ui as wfui
    seen = []
    monkeypatch.setattr(
        wfui, "update_status_ui",
        lambda parent_ui, msg, workflow_info=None: seen.append(msg),
    )
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="T"), record, experiment)
    task.update_status_ui("Working...")
    assert seen == ["grid-aspen [T] Working..."]


def test_grid_task_progress_helpers(demo_microscope, experiment, monkeypatch):
    import fibsem.applications.autolamella.workflows.ui as wfui
    calls = []
    monkeypatch.setattr(wfui, "update_progress_ui",
                        lambda parent_ui, **kw: calls.append(kw))
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="T"), record, experiment)

    task.progress_countdown(5, 10, "Working")
    task.progress_indeterminate("Busy")
    task.progress_done()

    assert calls[0] == {"remaining": 5, "total": 10,
                        "message": "grid-aspen [T] Working"}
    assert calls[1] == {"indeterminate": True, "message": "grid-aspen [T] Busy"}
    assert calls[2] == {"done": True}


# --- imaging-task supervise checkpoints (decline → skip cleanly) -----------

def _load_one_grid(microscope, name="grid-aspen"):
    slot_name = list(microscope._stage.holder.slots.keys())[0]
    microscope._stage.holder.slots[slot_name].loaded_grid = SampleGrid(name=name)


def _supervise(experiment, task_name):
    experiment.grid_protocol.workflow_config.tasks = [
        GridTaskDescription(name=task_name, supervise=True)
    ]


def test_acquire_image_skips_cleanly_when_declined(
    demo_microscope, experiment, monkeypatch
):
    import fibsem.applications.autolamella.workflows.ui as wfui

    _load_one_grid(demo_microscope)
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    _supervise(experiment, "ACQUIRE")

    monkeypatch.setattr(wfui, "ask_user", lambda **kw: False)  # decline at checkpoint
    monkeypatch.setattr(wfui, "update_status_ui", lambda **kw: None)
    monkeypatch.setattr(wfui, "update_progress_ui", lambda *a, **k: None)
    acquired = []
    monkeypatch.setattr(demo_microscope, "acquire_image",
                        lambda *a, **k: acquired.append(True))

    task = AcquireImageTask(demo_microscope,
                            AcquireImageGridTaskConfig(task_name="ACQUIRE"),
                            record, experiment, parent_ui=object())
    monkeypatch.setattr(task, "_move_to_grid_slot_position", lambda *a, **k: None)
    task.run()

    assert acquired == []  # declined before acquisition
    assert "ACQUIRE" not in record.results  # nothing recorded
    # skip is clean: the task itself still completes, workflow continues
    assert record.task_state.status is AutoLamellaTaskStatus.Completed


def test_acquire_overview_skips_cleanly_when_declined(
    demo_microscope, experiment, monkeypatch
):
    import fibsem.applications.autolamella.workflows.ui as wfui
    import fibsem.applications.autolamella.workflows.tasks.grid.imaging as imaging

    _load_one_grid(demo_microscope)
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    _supervise(experiment, "OVERVIEW")

    monkeypatch.setattr(wfui, "ask_user", lambda **kw: False)  # decline at checkpoint
    monkeypatch.setattr(wfui, "update_status_ui", lambda **kw: None)
    monkeypatch.setattr(wfui, "update_progress_ui", lambda *a, **k: None)
    stitched = []
    monkeypatch.setattr(imaging, "tiled_image_acquisition_and_stitch",
                        lambda **k: stitched.append(True))

    task = AcquireOverviewImageGridTask(
        demo_microscope, AcquireOverviewImageGridTaskConfig(task_name="OVERVIEW"),
        record, experiment, parent_ui=object())
    monkeypatch.setattr(task, "_move_to_grid_slot_position", lambda *a, **k: None)
    task.run()

    assert stitched == []  # declined before acquisition
    assert "OVERVIEW" not in record.results
    assert record.task_state.status is AutoLamellaTaskStatus.Completed


# --- stub tasks (logging-only _run, registered + runnable) -----------------

@pytest.mark.parametrize("task_type", [
    "PARALLEL_TRENCH_MILLING_GRID",
    "AUTOLAMELLA_TARGETING_GRID",
    "ACQUIRE_FLUORESCENCE_OVERVIEW_IMAGE_GRID",
])
def test_stub_task_registered_and_runs_cleanly(
    demo_microscope, experiment, task_type, monkeypatch
):
    # the stub tasks log only — running one completes without touching hardware
    # (targeting sleeps between log lines; neutralise it so the test is fast)
    import fibsem.applications.autolamella.workflows.tasks.grid.targeting as targeting
    monkeypatch.setattr(targeting.time, "sleep", lambda *a, **k: None)

    assert task_type in GRID_TASK_REGISTRY
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    run_grid_task(demo_microscope, task_type, experiment, record)

    assert record.task_state.status is AutoLamellaTaskStatus.Completed
    assert record.has_completed_task(task_type)
    assert task_type not in record.results  # stub records nothing


# --- GIS deposition task ---------------------------------------------------

def _gis_task(demo_microscope, experiment, monkeypatch, deposition_time=0.0,
              acquire_reference=False):
    import fibsem.applications.autolamella.workflows.ui as wfui

    monkeypatch.setattr(wfui, "update_status_ui", lambda **kw: None)
    monkeypatch.setattr(wfui, "update_progress_ui", lambda *a, **k: None)
    # the task now reaches GIS via the microscope facade (backend-agnostic)
    calls = []
    monkeypatch.setattr(
        demo_microscope, "run_gis_deposition",
        lambda duration, stop_event=None, on_progress=None: calls.append(
            {"duration": duration, "stop_event": stop_event, "on_progress": on_progress}),
    )

    _load_one_grid(demo_microscope)
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    from fibsem.applications.autolamella.workflows.tasks.grid import (
        CryoDepositionGridTask, CryoDepositionGridTaskConfig,
    )
    task = CryoDepositionGridTask(
        demo_microscope,
        CryoDepositionGridTaskConfig(task_name="GIS", deposition_time=deposition_time,
                                     acquire_reference=acquire_reference),
        record, experiment, parent_ui=object())
    monkeypatch.setattr(task, "_move_to_grid_slot_position", lambda *a, **k: None)
    monkeypatch.setattr(task, "acquire_grid_reference_image",
                        lambda *a, **k: ("/ref.tif", "/thumb.png"))
    return task, record, calls


def test_gis_deposition_delegates_to_microscope(
    demo_microscope, experiment, monkeypatch
):
    task, record, calls = _gis_task(demo_microscope, experiment, monkeypatch)
    task.run()

    # one deposition call, with the stop event threaded through for abort
    assert len(calls) == 1
    assert calls[0]["duration"] == 0.0
    assert calls[0]["stop_event"] is task._stop_event
    assert callable(calls[0]["on_progress"])
    assert record.results["GIS"]["deposition_time"] == 0.0
    assert "reference" not in record.results["GIS"]  # off by default in this fixture
    assert record.task_state.status is AutoLamellaTaskStatus.Completed


def test_gis_deposition_records_reference_when_enabled(
    demo_microscope, experiment, monkeypatch
):
    task, record, calls = _gis_task(demo_microscope, experiment, monkeypatch,
                                    acquire_reference=True)
    task.run()
    # the SEM reference is acquired and recorded alongside the deposition metadata
    assert record.results["GIS"]["deposition_time"] == 0.0
    assert record.results["GIS"]["reference"] == "/ref.tif"
    assert record.results["GIS"]["thumbnail"] == "/thumb.png"


def test_gis_deposition_skips_cleanly_when_declined(
    demo_microscope, experiment, monkeypatch
):
    import fibsem.applications.autolamella.workflows.ui as wfui

    task, record, calls = _gis_task(demo_microscope, experiment, monkeypatch)
    _supervise(experiment, "GIS")
    monkeypatch.setattr(wfui, "ask_user", lambda **kw: False)  # decline at checkpoint

    task.run()

    assert calls == []  # microscope deposition never invoked
    assert "GIS" not in record.results
    assert record.task_state.status is AutoLamellaTaskStatus.Completed


# --- sputter coating task --------------------------------------------------

def _sputter_task(demo_microscope, experiment, monkeypatch, acquire_reference=False):
    import fibsem.applications.autolamella.workflows.ui as wfui

    monkeypatch.setattr(wfui, "update_status_ui", lambda **kw: None)
    monkeypatch.setattr(wfui, "update_progress_ui", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(demo_microscope, "run_sputter_coater",
                        lambda t, current=None: calls.append((t, current)))

    _load_one_grid(demo_microscope)
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    from fibsem.applications.autolamella.workflows.tasks.grid import (
        CryoSputterGridTask, CryoSputterGridTaskConfig,
    )
    task = CryoSputterGridTask(
        demo_microscope,
        CryoSputterGridTaskConfig(task_name="SPUTTER", sputter_time=12.0,
                                  sputter_current=0.01,
                                  acquire_reference=acquire_reference),
        record, experiment, parent_ui=object())
    monkeypatch.setattr(task, "_move_to_grid_slot_position", lambda *a, **k: None)
    monkeypatch.setattr(task, "acquire_grid_reference_image",
                        lambda *a, **k: ("/ref.tif", "/thumb.png"))
    return task, record, calls


def test_sputter_runs_coater_with_int_time_and_current(
    demo_microscope, experiment, monkeypatch
):
    task, record, calls = _sputter_task(demo_microscope, experiment, monkeypatch)
    task.run()

    # time is passed as an int; current is threaded through
    assert calls == [(12, 0.01)]
    assert record.results["SPUTTER"]["sputter_time"] == 12.0
    assert record.results["SPUTTER"]["sputter_current"] == 0.01
    assert "reference" not in record.results["SPUTTER"]  # off by default in this fixture
    assert record.task_state.status is AutoLamellaTaskStatus.Completed


def test_sputter_records_reference_when_enabled(
    demo_microscope, experiment, monkeypatch
):
    task, record, calls = _sputter_task(demo_microscope, experiment, monkeypatch,
                                        acquire_reference=True)
    task.run()
    assert record.results["SPUTTER"]["sputter_time"] == 12.0
    assert record.results["SPUTTER"]["reference"] == "/ref.tif"
    assert record.results["SPUTTER"]["thumbnail"] == "/thumb.png"


def test_sputter_skips_cleanly_when_declined(
    demo_microscope, experiment, monkeypatch
):
    import fibsem.applications.autolamella.workflows.ui as wfui

    task, record, calls = _sputter_task(demo_microscope, experiment, monkeypatch)
    _supervise(experiment, "SPUTTER")
    monkeypatch.setattr(wfui, "ask_user", lambda **kw: False)  # decline

    task.run()

    assert calls == []  # coater never run
    assert "SPUTTER" not in record.results
    assert record.task_state.status is AutoLamellaTaskStatus.Completed


def test_acquire_grid_reference_image_saves_and_returns(
    demo_microscope, experiment, monkeypatch
):
    import os as _os
    import fibsem.applications.autolamella.workflows.ui as wfui

    monkeypatch.setattr(wfui, "update_status_ui", lambda **kw: None)
    _load_one_grid(demo_microscope)
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="T"),
                         record, experiment)
    monkeypatch.setattr(task, "_move_to_grid_slot_position", lambda *a, **k: None)

    img_path, thumb_path = task.acquire_grid_reference_image()

    # electron beam (default) → _eb suffix, matching the repo convention
    assert img_path.endswith("reference_eb.tif") and _os.path.exists(img_path)
    assert thumb_path.endswith("thumbnail.png") and _os.path.exists(thumb_path)


# --- GridTask lifecycle ----------------------------------------------------

def test_grid_task_lifecycle_success(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                  record, experiment).run()

    assert record.task_state.status is AutoLamellaTaskStatus.Completed
    assert record.has_completed_task("NOOP_GRID")
    assert record.task_state.duration >= 0


def test_grid_task_history_dataframe(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                  record, experiment).run()

    df = experiment.grid_task_history_dataframe()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["grid_name"] == "grid-aspen"
    assert row["grid_id"] == record._id
    assert row["task_name"] == "NOOP_GRID"
    assert row["task_status"] == "Completed"
    assert row["duration"] >= 0
    assert {"grid_name", "grid_id", "task_name", "task_id", "task_type",
            "task_status", "duration"} <= set(df.columns)


def test_grid_task_history_dataframe_empty(experiment):
    # no grids / history → empty frame, no error
    assert experiment.grid_task_history_dataframe().empty


def test_grid_task_records_result(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    _ResultGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                    record, experiment).run()

    assert record.results["NOOP_GRID"]["overview"] == "/tmp/overview.tif"
    assert record.results["NOOP_GRID"]["pixel_size"] == 1.2e-8


def test_grid_task_dirs_under_grids_subdir(demo_microscope, experiment):
    # outputs are separated from the top-level (per-lamella) layout under grids/
    import os

    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                         record, experiment)

    assert task.grid_dir() == os.path.join(experiment.path, "grids", "grid-aspen")
    tdir = task.task_dir()
    assert tdir == os.path.join(experiment.path, "grids", "grid-aspen", "NOOP_GRID")
    assert os.path.isdir(tdir)  # task_dir() creates it


def test_protocol_config_not_mutated_across_grids(demo_microscope, experiment):
    # the saved protocol config is a shared template; running a task on multiple
    # grids must NOT mutate it (regression: tiled acquisition wrote total_fov
    # back into the shared config's hfw, breaking subsequent grids).
    GRID_TASK_REGISTRY["MUTATING_GRID"] = _MutatingGridTask
    try:
        _load_two_grids(demo_microscope)
        experiment.sync_grids_from_holder(demo_microscope)
        cfg = _MutatingGridConfig(task_name="MUTATING_GRID", marker=0)
        experiment.grid_protocol.task_config["MUTATING_GRID"] = cfg

        run_grid_workflow(demo_microscope, experiment, ["MUTATING_GRID"],
                          grid_names=["A", "B"])

        # the shared template is untouched; each grid ran on its own copy
        assert experiment.grid_protocol.task_config["MUTATING_GRID"].marker == 0
    finally:
        GRID_TASK_REGISTRY.pop("MUTATING_GRID", None)


def test_grid_record_results_roundtrip():
    record = GridRecord(name="grid-aspen")
    record.results = {"OVERVIEW": {"overview": "/p/overview.tif", "pixel_size": 1e-8}}
    restored = GridRecord.from_dict(record.to_dict())
    assert restored.results == record.results


def test_grid_task_lifecycle_failure_records_state(demo_microscope, experiment):
    record = GridRecord(name="grid-birch")
    experiment.add_grid(record)

    with pytest.raises(RuntimeError):
        _BoomGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                      record, experiment).run()

    assert record.task_state.status is AutoLamellaTaskStatus.Failed
    assert record.is_failure
    assert record.task_state.status_message == "boom"


# --- Phase 2: config serialization & protocol -----------------------------

def test_grid_config_flat_round_trip_with_nested_settings():
    c = AcquireOverviewImageGridTaskConfig(task_name="overview")
    c.orientation = "FIB"
    c.settings.nrows, c.settings.ncols = 5, 4

    d = c.to_dict()
    assert d["task_type"] == "ACQUIRE_OVERVIEW_IMAGE_GRID"
    assert d["task_name"] == "overview"
    assert d["orientation"] == "FIB"
    assert "parameters" not in d  # flat, not a parameters subdict
    assert isinstance(d["settings"], dict) and d["settings"]["nrows"] == 5  # nested self-serialized

    c2 = load_grid_task_config(d)
    assert isinstance(c2, AcquireOverviewImageGridTaskConfig)
    assert c2.orientation == "FIB"
    assert c2.settings.nrows == 5 and c2.settings.ncols == 4
    assert c2.task_name == "overview"


def test_overview_config_default_factory_preserves_rich_default():
    # the default still carries the rich overview defaults (not clobbered)
    cfg = AcquireOverviewImageGridTaskConfig()
    assert cfg.settings.image_settings.resolution == (1024, 1024)


def test_grid_config_parameters_property_excludes_core():
    cc = CryoCleaningGridTaskConfig(task_name="clean")
    assert "task_name" not in cc.parameters
    assert {"orientation", "milling_angle", "field_of_view", "duration", "current"} <= set(cc.parameters)


def test_load_grid_task_config_unknown_returns_none():
    assert load_grid_task_config({"task_type": "NOT_A_REAL_TASK"}) is None


def test_grid_protocol_persist_and_reload(experiment):
    overview = AcquireOverviewImageGridTaskConfig(task_name="overview")
    overview.settings.nrows = 5
    experiment.grid_protocol.task_config["overview"] = overview
    experiment.grid_protocol.task_config["clean"] = CryoCleaningGridTaskConfig(task_name="clean")
    experiment.save()

    loaded = Experiment.load(f"{experiment.path}/experiment.yaml")
    assert set(loaded.grid_protocol.task_config.keys()) == {"overview", "clean"}
    ov = loaded.grid_protocol.task_config["overview"]
    assert isinstance(ov, AcquireOverviewImageGridTaskConfig)
    assert ov.settings.nrows == 5 and ov.task_name == "overview"


def test_experiment_back_compat_without_grid_protocol(experiment):
    ddict = experiment.to_dict()
    del ddict["grid_protocol"]
    restored = Experiment.from_dict(ddict)
    assert isinstance(restored.grid_protocol, GridTaskProtocol)
    assert len(restored.grid_protocol.task_config) == 0


# --- Phase 2: run_grid_task reads saved config ----------------------------

@dataclass
class _MarkedGridConfig(GridTaskConfig):
    task_type: ClassVar[str] = "MARKED_GRID"
    display_name: ClassVar[str] = "Marked"
    marker: str = "default"


class _RecordingGridTask(GridTask):
    config_cls = _MarkedGridConfig
    seen: ClassVar[list] = []

    def _run(self):
        type(self).seen.append(self.config.marker)


@pytest.fixture
def register_marked_task():
    GRID_TASK_REGISTRY["MARKED_GRID"] = _RecordingGridTask
    _RecordingGridTask.seen.clear()
    try:
        yield
    finally:
        GRID_TASK_REGISTRY.pop("MARKED_GRID", None)


def test_run_grid_task_uses_saved_config(demo_microscope, experiment, register_marked_task):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    experiment.grid_protocol.task_config["step"] = _MarkedGridConfig(
        task_name="step", marker="from-protocol")

    run_grid_task(demo_microscope, "step", experiment, record)

    assert _RecordingGridTask.seen == ["from-protocol"]


def test_run_grid_task_falls_back_to_default_config(demo_microscope, experiment, register_marked_task):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    # no saved config; task_name == task_type → default config
    run_grid_task(demo_microscope, "MARKED_GRID", experiment, record)

    assert _RecordingGridTask.seen == ["default"]


# --- Phase 3: orchestration (GridTaskManager) -----------------------------

@dataclass
class _OrderConfig(GridTaskConfig):
    task_type: ClassVar[str] = "ORDER_GRID"
    display_name: ClassVar[str] = "Order"
    fail_on: str = ""


class _OrderGridTask(GridTask):
    config_cls = _OrderConfig
    log: ClassVar[list] = []

    def _run(self):
        type(self).log.append((self.grid.name, self.task_name))
        if self.config.fail_on == self.grid.name:
            raise RuntimeError("boom")


@pytest.fixture
def register_order_task():
    GRID_TASK_REGISTRY["ORDER_GRID"] = _OrderGridTask
    _OrderGridTask.log = []
    try:
        yield
    finally:
        GRID_TASK_REGISTRY.pop("ORDER_GRID", None)


def _load_two_grids(microscope):
    names = list(microscope._stage.holder.slots.keys())[:2]
    microscope._stage.holder.slots[names[0]].loaded_grid = SampleGrid(name="A")
    microscope._stage.holder.slots[names[1]].loaded_grid = SampleGrid(name="B")


def test_workflow_runs_grid_outer(demo_microscope, experiment, register_order_task):
    _load_two_grids(demo_microscope)
    experiment.sync_grids_from_holder(demo_microscope)
    experiment.grid_protocol.task_config["t1"] = _OrderConfig(task_name="t1")
    experiment.grid_protocol.task_config["t2"] = _OrderConfig(task_name="t2")

    run_grid_workflow(demo_microscope, experiment, ["t1", "t2"], grid_names=["A", "B"])

    # all of A's tasks before any of B's
    assert _OrderGridTask.log == [("A", "t1"), ("A", "t2"), ("B", "t1"), ("B", "t2")]
    assert experiment.get_grid_by_name("A").has_completed_task("t2")


def test_task_failure_does_not_skip_grid(demo_microscope, experiment, register_order_task):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

    _load_two_grids(demo_microscope)
    experiment.sync_grids_from_holder(demo_microscope)
    experiment.grid_protocol.task_config["t1"] = _OrderConfig(task_name="t1", fail_on="A")
    experiment.grid_protocol.task_config["t2"] = _OrderConfig(task_name="t2")

    run_grid_workflow(demo_microscope, experiment, ["t1", "t2"], grid_names=["A", "B"])

    # A,t1 fails but A,t2 STILL runs — a failed task does not fail the grid
    assert _OrderGridTask.log == [("A", "t1"), ("A", "t2"), ("B", "t1"), ("B", "t2")]

    grid_a = experiment.get_grid_by_name("A")
    # the failure is recorded per-task in history, not as a grid-level failure
    assert any(ts.status is AutoLamellaTaskStatus.Failed for ts in grid_a.task_history)
    assert not grid_a.is_failure  # last task (t2) completed, so grid isn't failed


def test_workflow_fires_lifecycle_hooks(demo_microscope, experiment, register_order_task):
    from fibsem.applications.autolamella.workflows.tasks.grid_manager import GridTaskManager
    from fibsem.applications.autolamella.workflows.tasks.hooks import (
        FunctionHook,
        HookEvent,
        HookManager,
    )

    _load_two_grids(demo_microscope)
    experiment.sync_grids_from_holder(demo_microscope)
    experiment.grid_protocol.task_config["t1"] = _OrderConfig(task_name="t1", fail_on="A")

    fired = []
    hooks = HookManager()
    hooks.register(FunctionHook(
        name="capture",
        events=[HookEvent.TASK_STARTED, HookEvent.TASK_COMPLETED, HookEvent.TASK_FAILED],
        callback=lambda ctx: fired.append((ctx.item_name, ctx.task_name, ctx.event)),
    ))

    GridTaskManager(demo_microscope, experiment, hook_manager=hooks).run(
        ["t1"], grid_names=["A", "B"]
    )

    # A fails its task, B completes — hooks fire with the grid name as item_name
    assert ("A", "t1", "task_started") in fired
    assert ("A", "t1", "task_failed") in fired
    assert ("B", "t1", "task_completed") in fired


class _StubSignal:
    def __init__(self):
        self.events = []

    def emit(self, payload):
        self.events.append(payload)


class _StubUI:
    def __init__(self):
        self.grid_workflow_update_signal = _StubSignal()


def test_workflow_emits_status_updates(demo_microscope, experiment, register_order_task):
    _load_two_grids(demo_microscope)
    experiment.sync_grids_from_holder(demo_microscope)
    experiment.grid_protocol.task_config["t1"] = _OrderConfig(task_name="t1")

    ui = _StubUI()
    GridTaskManager(demo_microscope, experiment, parent_ui=ui).run(["t1"], grid_names=["A", "B"])

    triples = [
        (e["status"]["item_name"], e["status"]["task_name"], e["status"]["status"])
        for e in ui.grid_workflow_update_signal.events
    ]
    assert ("A", "t1", "InProgress") in triples
    assert ("A", "t1", "Completed") in triples
    assert ("B", "t1", "Completed") in triples
    assert triples[-1] == ("", "", "Completed")  # final "workflow complete"

    # progress indices are populated on the status dicts (neutral item_* keys)
    a_start = next(
        e["status"] for e in ui.grid_workflow_update_signal.events
        if e["status"]["item_name"] == "A" and e["status"]["status"] == "InProgress"
    )
    assert a_start["total_items"] == 2 and a_start["current_item_index"] == 0
    assert a_start["total_tasks"] == 1 and a_start["current_task_index"] == 0
    assert a_start["grid_name"] == "A"  # grid-specific alias still present


def test_ensure_loaded_noop_when_already_loaded(demo_microscope, experiment):
    name = list(demo_microscope._stage.holder.slots.keys())[0]
    demo_microscope._stage.holder.slots[name].loaded_grid = SampleGrid(name="X")
    record = GridRecord(name="X")
    experiment.add_grid(record)

    slot = GridTaskManager(demo_microscope, experiment).ensure_loaded(record)
    assert slot.loaded_grid.name == "X"


def test_ensure_loaded_halts_without_loader(demo_microscope, experiment):
    record = GridRecord(name="ghost")  # not in any slot, static holder (no loader)
    experiment.add_grid(record)

    with pytest.raises(GridExchangeError):
        GridTaskManager(demo_microscope, experiment).ensure_loaded(record)


def test_ensure_loaded_exchanges_on_autoloader(experiment):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)  # single slot + loader
    assert microscope._stage.loader is not None

    record = GridRecord(name="new-grid")
    experiment.add_grid(record)

    slot = GridTaskManager(microscope, experiment).ensure_loaded(record)
    assert slot.loaded_grid.name == "new-grid"
    assert microscope._stage.holder.find_slot_by_grid_name("new-grid") is not None


# --- Phase 3b: loader magazine --------------------------------------------

@pytest.fixture
def autoloader_microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)  # single working slot + loader
    return microscope


def test_loader_has_magazine(autoloader_microscope):
    loader = autoloader_microscope._stage.loader
    assert loader.capacity == 12
    assert len(loader.slots) == 12
    assert loader.run_inventory() == []  # empty until the operator loads it


def test_loader_assign_find_and_inventory(autoloader_microscope):
    loader = autoloader_microscope._stage.loader
    loader.assign_grid("Magazine-01", SampleGrid(name="A", radius=2e-3))
    loader.assign_grid("Magazine-02", SampleGrid(name="B"))

    assert {s.loaded_grid.name for s in loader.run_inventory()} == {"A", "B"}
    assert loader.find_grid("A").loaded_grid.radius == 2e-3
    assert loader.find_grid("missing") is None


def test_sync_grids_sources_loader_magazine(autoloader_microscope, experiment):
    loader = autoloader_microscope._stage.loader
    loader.assign_grid("Magazine-01", SampleGrid(name="A"))
    loader.assign_grid("Magazine-02", SampleGrid(name="B"))

    experiment.sync_grids_from_holder(autoloader_microscope)
    assert {g.name for g in experiment.grids} == {"A", "B"}


def test_ensure_loaded_pulls_real_grid_from_magazine(autoloader_microscope, experiment):
    loader = autoloader_microscope._stage.loader
    loader.assign_grid("Magazine-01", SampleGrid(name="A", radius=2e-3))
    record = GridRecord(name="A")
    experiment.add_grid(record)

    slot = GridTaskManager(autoloader_microscope, experiment).ensure_loaded(record)
    # the real magazine grid (with its geometry) is what gets inserted
    assert slot.loaded_grid.name == "A"
    assert slot.loaded_grid.radius == 2e-3


def test_ensure_loaded_exchanges_between_magazine_grids(autoloader_microscope, experiment):
    loader = autoloader_microscope._stage.loader
    loader.assign_grid("Magazine-01", SampleGrid(name="A"))
    loader.assign_grid("Magazine-02", SampleGrid(name="B"))
    experiment.add_grid(GridRecord(name="A"))
    experiment.add_grid(GridRecord(name="B"))
    mgr = GridTaskManager(autoloader_microscope, experiment)

    mgr.ensure_loaded(experiment.get_grid_by_name("A"))
    mgr.ensure_loaded(experiment.get_grid_by_name("B"))

    holder = autoloader_microscope._stage.holder
    assert holder.find_slot_by_grid_name("B") is not None
    assert holder.find_slot_by_grid_name("A") is None  # A retracted on exchange


# --- Lamella <-> Grid link -------------------------------------------------

def _make_lamella(experiment, name, grid_id=None):
    """A minimal Lamella under the experiment dir (avoids the empty task config
    path-stamping in add_new_lamella)."""
    from fibsem.applications.autolamella.structures import Lamella

    n = max((p.number for p in experiment.positions), default=0) + 1
    lam = Lamella(petname=name, path=f"{experiment.path}/{name}", number=n,
                  grid_id=grid_id)
    experiment.add_lamella(lam)
    return experiment.positions[-1]  # add_lamella deepcopies; return the stored copy


def test_lamella_grid_id_round_trip():
    from fibsem.applications.autolamella.structures import Lamella

    lam = Lamella(petname="lam-01", path="/tmp/lam-01", number=1, grid_id="grid-uuid")
    lam2 = Lamella.from_dict(lam.to_dict())
    assert lam2.grid_id == "grid-uuid"


def test_lamella_grid_id_back_compat():
    """A pre-grids lamella dict (no grid_id key) loads as unlinked."""
    from fibsem.applications.autolamella.structures import Lamella

    data = Lamella(petname="lam-01", path="/tmp/lam-01", number=1).to_dict()
    data.pop("grid_id", None)
    assert Lamella.from_dict(data).grid_id is None


def test_get_grid_by_id(experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    assert experiment.get_grid_by_id(record._id) is record
    assert experiment.get_grid_by_id("missing") is None
    assert experiment.get_grid_by_id(None) is None


def test_get_lamellae_for_grid(experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    a = _make_lamella(experiment, "lam-a", grid_id=record._id)
    b = _make_lamella(experiment, "lam-b", grid_id=record._id)
    _make_lamella(experiment, "lam-c", grid_id="other")  # different grid
    _make_lamella(experiment, "lam-d")  # unlinked

    linked = experiment.get_lamellae_for_grid(record)
    assert {lam.name for lam in linked} == {"lam-a", "lam-b"}
    # accepts an id directly too
    assert experiment.get_lamellae_for_grid(record._id) == linked
    assert experiment.get_lamellae_for_grid(GridRecord(name="empty")) == []


def test_get_grid_for_lamella(experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    linked = _make_lamella(experiment, "lam-a", grid_id=record._id)
    unlinked = _make_lamella(experiment, "lam-b")

    assert experiment.get_grid_for_lamella(linked) is record
    assert experiment.get_grid_for_lamella(unlinked) is None


def test_add_new_lamella_stamps_grid_id(experiment, demo_microscope):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

    experiment.task_protocol = AutoLamellaTaskProtocol()  # add_new_lamella reads defaults
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    state = demo_microscope.get_microscope_state()

    experiment.add_new_lamella(state, task_config={}, name="lam-01",
                               grid_id=record._id)

    assert experiment.positions[-1].grid_id == record._id
    assert experiment.get_lamellae_for_grid(record)[0].name == "lam-01"


def test_remove_grid_orphans_lamellae(experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    linked = _make_lamella(experiment, "lam-a", grid_id=record._id)
    other = _make_lamella(experiment, "lam-b", grid_id="other")

    experiment.remove_grid("grid-aspen")

    assert experiment.get_grid_by_name("grid-aspen") is None
    assert linked in experiment.positions  # lamella survives
    assert linked.grid_id is None          # link cleared
    assert other.grid_id == "other"        # untouched


def test_grid_task_create_lamella_stamps_grid_id(demo_microscope, experiment):
    """A grid task creating a lamella stamps it with the task's grid_id."""
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
    from fibsem.structures import FibsemStagePosition

    experiment.task_protocol = AutoLamellaTaskProtocol()
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)
    task = _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="T"),
                         record, experiment)

    lam = task.create_lamella(FibsemStagePosition(x=1e-3, y=2e-3, z=0),
                              name="target-01")

    assert lam.grid_id == record._id
    assert experiment.get_lamellae_for_grid(record) == [lam]
    assert lam.name == "target-01"
