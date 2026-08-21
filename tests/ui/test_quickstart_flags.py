"""The --quickstart / --quickload developer flags.

Both do the same thing the first two clicks of every session do -- connect with
the default configuration, reopen the last experiment -- so the tests drive the
real widget against the real Demo microscope and a real Experiment on disk. The
only stand-in is the preferences file, redirected to a temp directory so a test
run never reads or rewrites the developer's own recents.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import _parse_args
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
)

TASKS = ["Trench", "Polishing"]


@pytest.fixture
def prefs_env(tmp_path, monkeypatch):
    """Point the preferences helpers at an isolated temp file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml")
    )
    return tmp_path


def make_experiment(path: Path, name: str, with_protocol: bool = True) -> Experiment:
    """Write a real experiment (and, by default, its protocol) to disk."""
    experiment = Experiment.create(path=path, name=name)
    if with_protocol:
        experiment.task_protocol = AutoLamellaTaskProtocol(
            workflow_config=AutoLamellaWorkflowConfig(
                tasks=[
                    AutoLamellaTaskDescription(name=n, supervise=False, required=False)
                    for n in TASKS
                ]
            ),
            task_config={n: AcquireReferenceImageConfig(task_name=n) for n in TASKS},
        )
        experiment.task_protocol.save(os.path.join(experiment.path, "protocol.yaml"))
    experiment.save()
    return experiment


def remember(experiment: Experiment) -> None:
    """Record an experiment the way the create and load dialogs do."""
    prefs = cfg.load_user_preferences()
    prefs.experiment.last_experiment_path = str(experiment.path)
    cfg.add_recent_experiment(prefs, os.path.join(experiment.path, "experiment.yaml"))
    cfg.save_user_preferences(prefs)


@pytest.fixture
def ui(qapp):
    """A real AutoLamellaUI, without the main window that normally hosts it."""
    widget = AutoLamellaUI(parent_ui=None)
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


# ── Flags ─────────────────────────────────────────────────────────────────────


def test_neither_flag_is_set_by_default():
    args = _parse_args([])

    assert args.quickstart is False
    assert args.quickload is False


@pytest.mark.parametrize("flag", ["--quickstart", "--quickload"])
def test_each_flag_sets_only_itself(flag):
    args = _parse_args([flag])

    assert getattr(args, flag.lstrip("-")) is True


def test_an_unknown_flag_is_rejected():
    """A typo should not silently start an ordinary, unattended session."""
    with pytest.raises(SystemExit):
        _parse_args(["--quickstrat"])


# ── Which experiment is "the last one" ────────────────────────────────────────


def test_last_experiment_is_resolved_to_its_yaml(prefs_env):
    experiment = make_experiment(prefs_env, "exp-a")
    remember(experiment)

    assert cfg.get_last_experiment_file() == os.path.join(
        experiment.path, "experiment.yaml"
    )


def test_a_deleted_last_experiment_falls_back_to_the_next_newest(prefs_env):
    kept = make_experiment(prefs_env, "exp-kept")
    remember(kept)
    deleted = make_experiment(prefs_env, "exp-deleted")
    remember(deleted)
    os.remove(os.path.join(deleted.path, "experiment.yaml"))

    assert cfg.get_last_experiment_file() == os.path.normpath(
        os.path.join(kept.path, "experiment.yaml")
    )


def test_nothing_recorded_resolves_to_nothing(prefs_env):
    assert cfg.get_last_experiment_file() is None


# ── Connecting and loading ────────────────────────────────────────────────────


def test_quickstart_connects_without_being_clicked(ui, prefs_env):
    ui.quickstart()

    assert ui.microscope is not None
    assert ui.system_widget.microscope is ui.microscope


def test_quickload_reopens_the_last_experiment(ui, prefs_env):
    experiment = make_experiment(prefs_env, "exp-a")
    remember(experiment)

    ui.quickstart(load_experiment=True)

    assert ui.microscope is not None
    assert ui.experiment is not None
    assert ui.experiment.name == "exp-a"
    assert ui.experiment.path == experiment.path


def test_quickstart_alone_leaves_the_experiment_unloaded(ui, prefs_env):
    remember(make_experiment(prefs_env, "exp-a"))

    ui.quickstart()

    assert ui.microscope is not None
    assert ui.experiment is None


def test_an_experiment_without_a_protocol_is_not_adopted(ui, prefs_env):
    """The bar the load dialog sets: there would be nothing to run."""
    remember(make_experiment(prefs_env, "exp-no-protocol", with_protocol=False))

    ui.quickstart(load_experiment=True)

    assert ui.microscope is not None
    assert ui.experiment is None


def test_no_recent_experiment_still_leaves_a_connected_window(ui, prefs_env):
    ui.quickstart(load_experiment=True)

    assert ui.microscope is not None
    assert ui.experiment is None


def test_quickstart_does_not_disconnect_an_existing_connection(ui, prefs_env):
    ui.quickstart()
    microscope = ui.microscope

    ui.quickstart()

    assert ui.microscope is microscope
