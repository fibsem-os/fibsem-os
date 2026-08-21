"""Tier 1 user scripts: discovery, flags, and the runner (FIB-338)."""

import threading
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.scripting import (
    DEFAULT_SCRIPTS_DIR,
    SCRIPTS_DIR_ENV_VAR,
    ScriptContext,
    discover_scripts,
    get_scripts_directory,
    run_script,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.cancellation import OperationCancelledError
from fibsem.structures import MicroscopeState


def _experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path / "exp-root", name="exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    Path(exp.path).mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.save(save_protocol=True)
    return exp


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    return path


# ── directory resolution ─────────────────────────────────────────────────────


def test_scripts_directory_defaults_under_home(monkeypatch):
    monkeypatch.delenv(SCRIPTS_DIR_ENV_VAR, raising=False)
    assert get_scripts_directory() == DEFAULT_SCRIPTS_DIR


def test_scripts_directory_env_override(monkeypatch, tmp_path):
    """Shared microscope PCs need to point this elsewhere."""
    monkeypatch.setenv(SCRIPTS_DIR_ENV_VAR, str(tmp_path / "custom"))
    assert get_scripts_directory() == tmp_path / "custom"


def test_missing_directory_is_not_an_error(tmp_path):
    assert discover_scripts(tmp_path / "nope") == []


# ── discovery ────────────────────────────────────────────────────────────────


def test_docstring_becomes_the_description(tmp_path):
    _write(
        tmp_path,
        "export.py",
        '"""Export a summary."""\ndef run(ctx):\n    return "ok"\n',
    )
    script = discover_scripts(tmp_path)[0]
    assert script.name == "export"
    assert script.description == "Export a summary."
    assert script.is_runnable


def test_a_file_without_run_is_reported_not_hidden(tmp_path):
    """Silently omitting it is the failure this is meant to avoid — the author
    sees nothing and assumes the folder is wrong."""
    _write(tmp_path, "broken.py", "def main(ctx):\n    return 1\n")
    script = discover_scripts(tmp_path)[0]
    assert not script.is_runnable
    assert "run()" in script.error


def test_a_file_that_raises_on_import_is_reported_not_hidden(tmp_path):
    _write(tmp_path, "boom.py", "raise ValueError('bad import')\n")
    script = discover_scripts(tmp_path)[0]
    assert not script.is_runnable
    assert "bad import" in script.error


def test_flags_are_read_from_the_module(tmp_path):
    _write(
        tmp_path,
        "flags.py",
        "writes = True\nbackground = True\nuses_microscope = True\n"
        "on_workflow_completed = True\ndef run(ctx):\n    pass\n",
    )
    script = discover_scripts(tmp_path)[0]
    assert (script.writes, script.background) == (True, True)
    assert (script.uses_microscope, script.on_workflow_completed) == (True, True)


def test_flags_default_to_false(tmp_path):
    _write(tmp_path, "plain.py", "def run(ctx):\n    pass\n")
    script = discover_scripts(tmp_path)[0]
    assert not any(
        [
            script.writes,
            script.background,
            script.uses_microscope,
            script.on_workflow_completed,
        ]
    )


def test_same_filename_in_two_folders_does_not_collide(tmp_path):
    """Scripts are loaded under synthetic names and kept out of sys.modules."""
    _write(tmp_path / "a", "dup.py", 'def run(ctx):\n    return "A"\n')
    _write(tmp_path / "b", "dup.py", 'def run(ctx):\n    return "B"\n')
    a = discover_scripts(tmp_path / "a")[0]
    b = discover_scripts(tmp_path / "b")[0]
    assert a._module is not b._module
    assert (a._module.run(None), b._module.run(None)) == ("A", "B")


def test_underscore_files_are_skipped(tmp_path):
    _write(tmp_path, "_helper.py", "def run(ctx):\n    pass\n")
    _write(tmp_path, "real.py", "def run(ctx):\n    pass\n")
    assert [s.name for s in discover_scripts(tmp_path)] == ["real"]


def test_edits_are_picked_up_without_a_reload_step(tmp_path):
    """Scripts are not cached, so edit-then-run just works."""
    path = _write(tmp_path, "edit.py", 'def run(ctx):\n    return "before"\n')
    assert discover_scripts(tmp_path)[0]._module.run(None) == "before"
    path.write_text('def run(ctx):\n    return "after"\n')
    assert discover_scripts(tmp_path)[0]._module.run(None) == "after"


# ── running ──────────────────────────────────────────────────────────────────


def test_run_passes_the_experiment_and_returns_the_value(tmp_path):
    exp = _experiment(tmp_path)
    _write(
        tmp_path / "s",
        "names.py",
        "def run(ctx):\n    return [p.name for p in ctx.experiment.positions]\n",
    )
    script = discover_scripts(tmp_path / "s")[0]

    result = run_script(script, ScriptContext(experiment=exp))

    assert result.ok
    assert result.value == [p.name for p in exp.positions]


def test_a_failing_script_is_captured_not_raised(tmp_path):
    """This runs from a Qt slot, and PyQt5 aborts the process on any exception
    that escapes one (FIB-329)."""
    exp = _experiment(tmp_path)
    _write(tmp_path / "s", "bad.py", "def run(ctx):\n    raise RuntimeError('nope')\n")
    script = discover_scripts(tmp_path / "s")[0]

    result = run_script(script, ScriptContext(experiment=exp))

    assert not result.ok
    assert isinstance(result.error, RuntimeError)


def test_writes_flag_persists_the_change(tmp_path):
    exp = _experiment(tmp_path)
    _write(
        tmp_path / "s",
        "mark.py",
        "writes = True\n"
        "def run(ctx):\n    ctx.experiment.positions[0].defect.set_defect('bad')\n",
    )
    script = discover_scripts(tmp_path / "s")[0]

    result = run_script(script, ScriptContext(experiment=exp))

    assert result.saved
    reloaded = Experiment.load(str(Path(exp.path) / "experiment.yaml"))
    assert reloaded.positions[0].is_failure


def test_without_the_writes_flag_nothing_is_persisted(tmp_path):
    """A read-only script must not rewrite the experiment yaml by accident."""
    exp = _experiment(tmp_path)
    _write(
        tmp_path / "s",
        "sneaky.py",
        "def run(ctx):\n    ctx.experiment.positions[0].defect.set_defect('bad')\n",
    )
    script = discover_scripts(tmp_path / "s")[0]

    result = run_script(script, ScriptContext(experiment=exp))

    assert not result.saved
    reloaded = Experiment.load(str(Path(exp.path) / "experiment.yaml"))
    assert not reloaded.positions[0].is_failure


def test_an_unrunnable_script_returns_an_error_result(tmp_path):
    exp = _experiment(tmp_path)
    _write(tmp_path / "s", "broken.py", "def main(ctx):\n    pass\n")
    script = discover_scripts(tmp_path / "s")[0]

    result = run_script(script, ScriptContext(experiment=exp))

    assert not result.ok


def test_ctx_path_is_the_experiment_directory(tmp_path):
    exp = _experiment(tmp_path)
    assert ScriptContext(experiment=exp).path == Path(exp.path)


def test_ctx_cancellation_is_inert_without_a_stop_event(tmp_path):
    """A headless run has no one to press Stop, so the same script must still work."""
    ctx = ScriptContext(experiment=_experiment(tmp_path))

    assert ctx.cancelled is False
    ctx.raise_if_cancelled()  # must not raise


def test_ctx_raise_if_cancelled_aborts_once_the_event_is_set(tmp_path):
    ctx = ScriptContext(experiment=_experiment(tmp_path), stop_event=threading.Event())
    ctx.stop_event.set()

    assert ctx.cancelled is True
    with pytest.raises(OperationCancelledError):
        ctx.raise_if_cancelled()
