"""The App Server's config edits with no window: applied, saved and recorded
as the agent's edit, through the session's executor."""

from pathlib import Path

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.server.context import (
    config_version,
    item_fields_version,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTaskConfig,
)
from fibsem.structures import MicroscopeState, Point
from tests.run_state import stub_session

TASK = "Rough Milling"
KEY = "mill_rough"
DEPTH = f"milling.{KEY}.stages.0.pattern.depth"


class Scope:
    """A microscope's record, and nothing else."""

    def __init__(self):
        self.events = []

    def record_event(self, kind, payload):
        from fibsem.acting import current_actor

        self.events.append((kind, current_actor(), payload))


def _context(tmp_path):
    (tmp_path / "exp").mkdir()
    experiment = Experiment(path=tmp_path / "exp", name="no-window")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    experiment.positions[0].task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    session = stub_session()
    session.experiment = experiment
    session.microscope = Scope()
    return AutoLamellaAppContext(session), experiment, session.microscope


def test_an_item_task_config_patch_lands_and_is_the_agent_s_edit(tmp_path):
    context, experiment, scope = _context(tmp_path)
    lamella = experiment.positions[0]
    config = lamella.task_config[TASK]

    result = context.apply_item_task_config_patch(
        lamella.name, TASK, {DEPTH: 2.7e-6}, config_version(config)
    )

    assert result["available"] and result["applied"] and result["saved"], result
    assert config.milling[KEY].stages[0].pattern.depth == 2.7e-6
    ((kind, actor, payload),) = scope.events
    assert (kind, str(actor), payload["target"]) == ("edit", "agent", "task_config")
    reloaded = Experiment.load(Path(experiment.path) / "experiment.yaml")
    assert (
        reloaded.positions[0].task_config[TASK].milling[KEY].stages[0].pattern.depth
        == 2.7e-6
    )


def test_a_stale_version_is_refused(tmp_path):
    context, experiment, scope = _context(tmp_path)
    result = context.apply_item_task_config_patch(
        experiment.positions[0].name, TASK, {DEPTH: 2.7e-6}, "stale"
    )
    assert result == {"applied": False, "stale": True, "available": True}
    assert scope.events == []


def test_the_protocol_patch_reorder_and_apply_need_no_window(tmp_path):
    context, experiment, _scope = _context(tmp_path)
    lamella = experiment.positions[0]
    protocol_config = experiment.task_protocol.task_config[TASK]

    patched = context.apply_protocol_task_config_patch(
        TASK, {DEPTH: 3.3e-6}, config_version(protocol_config)
    )
    assert patched["applied"], patched

    stage = lamella.task_config[TASK].milling[KEY].stages[0]
    stage.pattern.point = Point(1e-6, 2e-6)
    applied = context.apply_protocol_to_item(lamella.name, [TASK])
    assert applied["applied"], applied
    stage = lamella.task_config[TASK].milling[KEY].stages[0]
    assert stage.pattern.depth == 3.3e-6
    # as the editors' apply: the item keeps where its patterns are
    assert (stage.pattern.point.x, stage.pattern.point.y) == (1e-6, 2e-6)

    config = lamella.task_config[TASK]
    order = list(reversed([s.name for s in config.milling[KEY].stages]))
    reordered = context.reorder_milling_stages(
        "item", lamella.name, TASK, KEY, order, config_version(config)
    )
    assert reordered["applied"], reordered
    assert [s.name for s in config.milling[KEY].stages] == order


def test_an_item_patch_needs_no_window(tmp_path):
    context, experiment, _scope = _context(tmp_path)
    lamella = experiment.positions[0]
    result = context.apply_item_patch(
        lamella.name, {"description": "checked"}, item_fields_version(lamella)
    )
    assert result["applied"], result
    assert lamella.description == "checked"


def test_the_editors_copy_and_global_edit_save_the_protocol(tmp_path):
    """The lamella editor's Apply to other lamellae, with the base protocol
    too, and the protocol editor's Global Edit: config_edits functions, the
    one the agent's apply_protocol calls among them, saved with the protocol."""
    from fibsem.applications.autolamella.config_edits import (
        copy_task_configs,
        set_reference_imaging,
    )

    _, experiment, _ = _context(tmp_path)
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    source, target = experiment.positions
    source.task_config[TASK].milling[KEY].stages[0].pattern.depth = 3.1e-6
    target.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)

    count, saved = copy_task_configs(
        experiment,
        [target.name],
        [TASK],
        source_name=source.name,
        update_base_protocol=True,
    )

    assert (count, saved) == (1, True)
    protocol = AutoLamellaTaskProtocol.load(Path(experiment.path) / "protocol.yaml")
    assert protocol.task_config[TASK].milling[KEY].stages[0].pattern.depth == 3.1e-6

    reference = experiment.task_protocol.task_config[TASK].reference_imaging
    reference.field_of_view1 = 222e-6
    target.task_state.name, target.task_state.status = TASK, _in_progress()
    milling, skipped = set_reference_imaging(
        experiment, [TASK], reference, 99e-6, experiment.positions
    )

    assert milling == 1 and skipped == [(target.name, TASK)]
    assert source.task_config[TASK].milling[KEY].field_of_view == 99e-6
    assert target.task_config[TASK].milling[KEY].field_of_view != 99e-6
    protocol = AutoLamellaTaskProtocol.load(Path(experiment.path) / "protocol.yaml")
    assert protocol.task_config[TASK].milling[KEY].field_of_view == 99e-6


def _in_progress():
    from fibsem.workflows.status import TaskStatus

    return TaskStatus.InProgress


def test_a_patch_naming_attention_or_requires_is_refused(tmp_path):
    """Both say how a task takes part in the workflow, and have their own
    verbs; a config patch naming either changes nothing."""
    from fibsem.workflows.attention import Attention

    context, experiment, scope = _context(tmp_path)
    protocol_config = experiment.task_protocol.task_config[TASK]
    lamella = experiment.positions[0]
    for field, verb, value in (
        ("attention", "protocol_set_supervision", "supervised"),
        ("requires", "protocol_set_task_requirements", ["Setup"]),
    ):
        refused = context.apply_protocol_task_config_patch(
            TASK, {field: value}, config_version(protocol_config)
        )
        assert refused["applied"] is False and verb in refused["invalid_patch"]
        assert refused["path"] == field
        refused = context.apply_item_task_config_patch(
            lamella.name,
            TASK,
            {field: value},
            config_version(lamella.task_config[TASK]),
        )
        assert refused["applied"] is False and verb in refused["invalid_patch"]
    assert experiment.task_protocol.get_attention(TASK) is Attention.automated
    assert experiment.task_protocol.requirements(TASK) == []
    assert scope.events == []


def test_the_editors_values_are_copied_touched_and_saved_on_one_path(tmp_path):
    """The editors' write: each value set whole and copied, touched for the
    record before it changes, and saved by the same save the patches make."""
    from fibsem.server.task_configs import save_edits, set_task_config_values
    from fibsem.workflows.edit_recording import PendingEdits

    _, experiment, scope = _context(tmp_path)
    lamella = experiment.positions[0]
    config = lamella.task_config[TASK]
    imaging = config.reference_imaging.__class__(field_of_view1=250e-6)
    edits = PendingEdits(lambda: scope, via="lamella editor")

    result = set_task_config_values(
        experiment,
        "item",
        lamella.name,
        TASK,
        {"reference_imaging": imaging, "sync_to_poi": config.sync_to_poi},
        edits,
    )

    assert result["applied"], result
    assert [c["path"] for c in result["changes"]] == ["reference_imaging"]
    assert config.reference_imaging.field_of_view1 == 250e-6
    assert config.reference_imaging is not imaging
    assert save_edits(experiment, "item")
    edits.flush()
    ((kind, _actor, payload),) = scope.events
    assert (kind, payload["target"], payload["via"]) == (
        "edit",
        "reference_imaging",
        "lamella editor",
    )
    assert payload["before"]["field_of_view1"] != 250e-6
    assert payload["after"]["field_of_view1"] == 250e-6
    reloaded = Experiment.load(Path(experiment.path) / "experiment.yaml")
    saved = reloaded.positions[0].task_config[TASK].reference_imaging
    assert saved.field_of_view1 == 250e-6


def test_the_editors_values_refuse_an_unknown_task(tmp_path):
    from fibsem.server.task_configs import set_task_config_values
    from fibsem.workflows.edit_recording import PendingEdits

    _, experiment, scope = _context(tmp_path)
    result = set_task_config_values(
        experiment,
        "protocol",
        "",
        "Nope",
        {"sync_to_poi": False},
        PendingEdits(lambda: scope, via="x"),
    )
    assert result["applied"] is False
    assert result["task_names"] == [TASK]


def test_the_running_task_s_config_is_refused_at_filing_and_at_apply(tmp_path):
    """Only the running task's own config waits (FIB-1505): a request to edit
    it is refused when it is filed, not after the operator approves it."""
    from types import SimpleNamespace

    from fibsem.workflows.queue import TaskQueue

    context, experiment, scope = _context(tmp_path)
    lamella = experiment.positions[0]
    queue = TaskQueue()
    queue.build_from_pairs([(lamella.name, TASK)])
    queue.next()
    context._session.runner.manager = SimpleNamespace(queue=queue)
    context._session.runner.is_running = True
    arguments = {"item_name": lamella.name, "task_name": TASK, "patch": {DEPTH: 2.5e-6}}

    why = context.check_action("item_update_task_config", arguments)
    assert why and "is running for" in why
    refused = context.apply_item_task_config_patch(
        lamella.name, TASK, {DEPTH: 2.5e-6}, config_version(lamella.task_config[TASK])
    )
    assert refused["applied"] is False and refused["error"] == why

    other = dict(arguments, task_name="Setup Lamella Position")
    assert context.check_action("item_update_task_config", other) is None
