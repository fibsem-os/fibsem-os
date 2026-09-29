"""The shipped example scripts, executed rather than eyeballed.

Every scripting example that was never run turned out to have a bug in it: one
addressed ``lamella.state``, which does not exist, and another relied on
``microscope.acquire_image`` honouring ``save=True``, which it does not -- so it
claimed to write files and wrote none. Both looked completely reasonable on the
page.

That is why ``examples/scripts/`` holds real ``.py`` files instead of markdown
snippets: they go through the same ``load_script`` / ``run_script`` path the GUI
uses, against a real Experiment and a simulated microscope. If an example stops
working, this fails.
"""

from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.utils as utils
from fibsem.applications.autolamella.scripting import ScriptContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.scripting import load_script, run_script
from fibsem.structures import FibsemStagePosition, MicroscopeState

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "scripts"

# Named individually rather than globbed: a typo in the folder name would make a
# glob quietly collect nothing and the suite would pass having tested nothing.
EXPECTED = ["describe_lamellae", "export_summary", "survey_positions"]


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path / "exp", name="exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    Path(exp.path).mkdir(parents=True, exist_ok=True)
    for i in range(2):
        exp.add_new_lamella(MicroscopeState(), EventedDict())
        # a fresh lamella's milling pose has no coordinates, and moving to None
        # is not something the examples should have to defend against
        exp.positions[i].stage_position = FibsemStagePosition(
            x=i * 100e-6, y=0.0, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
        )
    exp.save(save_protocol=True)
    return exp


@pytest.fixture(scope="module")
def microscope() -> DemoMicroscope:
    return DemoMicroscope(utils.load_microscope_configuration().system)


def _context(experiment, microscope=None) -> ScriptContext:
    return ScriptContext(experiment=experiment, microscope=microscope)


def _load(name: str):
    script = load_script(EXAMPLES / f"{name}.py")
    assert script.is_runnable, f"{name} failed to load: {script.error}"
    return script


# ── the folder itself ────────────────────────────────────────────────────────

def test_the_examples_folder_holds_exactly_the_documented_scripts():
    """SCRIPTING.md points at these by name, so an added or renamed file needs the
    docs updated in the same commit."""
    found = sorted(p.stem for p in EXAMPLES.glob("*.py"))
    assert found == sorted(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_every_example_loads_and_declares_a_docstring(name):
    """The first docstring line becomes the description in the manager dialog, so
    an example without one shows its own filename twice."""
    script = _load(name)
    assert script.description and script.description != script.name


# ── one test per example, asserting the effect it advertises ─────────────────

def test_export_summary_writes_a_csv_with_a_row_per_lamella(experiment):
    script = _load("export_summary")

    result = run_script(script, _context(experiment))

    assert result.ok, result.error
    assert isinstance(result.value, Path) and result.value.exists()
    assert result.value.parent == Path(experiment.path)
    # header + one row per lamella
    assert len(result.value.read_text(encoding="utf-8").strip().splitlines()) == len(experiment.positions) + 1


def test_export_summary_does_not_declare_itself_a_writer(experiment):
    """It is the read-only example. If it ever grows a `writes` flag the point of
    having three tiers is gone."""
    script = _load("export_summary")
    assert not script.writes
    assert not script.uses_microscope

    result = run_script(script, _context(experiment))

    assert result.saved is False


def test_describe_lamellae_sets_descriptions_and_asks_to_be_saved(experiment):
    script = _load("describe_lamellae")
    assert script.writes

    result = run_script(script, _context(experiment))

    assert result.ok, result.error
    assert result.saved is True  # run_script called ctx.save() because of the flag
    assert all(lamella.description for lamella in experiment.positions)
    assert "no tasks completed yet" in experiment.positions[0].description


def test_describe_lamellae_is_idempotent(experiment):
    """Second run changes nothing, so re-running it is not a way to dirty the
    experiment. The example short-circuits on an unchanged description."""
    script = _load("describe_lamellae")

    first = run_script(script, _context(experiment))
    second = run_script(script, _context(experiment))

    assert first.ok and second.ok
    assert "described 2 of 2" in first.value
    assert "described 0 of 2" in second.value


def test_describe_lamellae_fires_the_events_the_ui_listens_to(experiment):
    """The example claims in a comment that the GUI updates as it runs. That is
    only true if `description` actually emits, so pin it here rather than trust
    the comment."""
    seen = []
    for lamella in experiment.positions:
        lamella.events.description.connect(lambda *_: seen.append(1))

    run_script(_load("describe_lamellae"), _context(experiment))

    assert len(seen) == len(experiment.positions)


def test_survey_positions_acquires_an_image_per_lamella(experiment, microscope):
    script = _load("survey_positions")
    assert script.uses_microscope

    result = run_script(script, _context(experiment, microscope))

    assert result.ok, result.error
    for lamella in experiment.positions:
        written = list(Path(lamella.path).glob("survey*"))
        assert written, f"no survey image written for {lamella.name}"


def test_survey_positions_stops_when_asked(experiment, microscope):
    """Cancellation is cooperative, so this only passes because the example calls
    ctx.raise_if_cancelled(). Delete that line and this test is what notices."""
    import threading

    from fibsem.cancellation import OperationCancelledError

    context = _context(experiment, microscope)
    context.stop_event = threading.Event()
    context.stop_event.set()  # already cancelled before the first iteration

    result = run_script(_load("survey_positions"), context)

    assert isinstance(result.error, OperationCancelledError)
    for lamella in experiment.positions:
        assert not list(Path(lamella.path).glob("survey*"))
