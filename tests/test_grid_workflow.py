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
    GridTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (
    GRID_TASK_REGISTRY,
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


# --- GridTask lifecycle ----------------------------------------------------

def test_grid_task_lifecycle_success(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                  record, experiment).run()

    assert record.task_state.status is AutoLamellaTaskStatus.Completed
    assert record.has_completed_task("NOOP_GRID")
    assert record.task_state.duration >= 0


def test_grid_task_records_result(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    _ResultGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                    record, experiment).run()

    assert record.results["NOOP_GRID"]["overview"] == "/tmp/overview.tif"
    assert record.results["NOOP_GRID"]["pixel_size"] == 1.2e-8


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
