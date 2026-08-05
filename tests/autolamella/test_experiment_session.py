"""An experiment should record which session worked on it (FIB-451).

`Experiment` recorded what it contains and nothing about what made it, so "was
this before or after the upgrade", "which instrument was this on" and "which
build of my plugin ran" were all unanswerable from the record -- for the five
personas the project is built around, it is the one fact they all need.

Every part of it was already collected per-image. What was missing was a moment
to promote it, and `register_metadata` is that moment: the only place the
experiment meets a microscope.

A session is what `setup_session` establishes -- a connected instrument and the
configuration around it -- plus who is at the keyboard. It is the **latest** one,
not the one that created the experiment: `Experiment.create()` has no microscope
to ask.
"""

import os
from pathlib import Path

import pytest
import yaml

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.microscope import FibsemMicroscope


@pytest.fixture
def microscope() -> FibsemMicroscope:
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


def _stored(experiment: Experiment) -> dict:
    """What is actually on disk for this experiment."""
    with open(os.path.join(experiment.path, "experiment.yaml")) as f:
        return yaml.safe_load(f)


def _experiment(tmp_path: Path, metadata=None) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-experiment", metadata=metadata)
    exp.task_protocol = AutoLamellaTaskProtocol()
    # Constructing an Experiment deliberately does not write to disk (FIB-420).
    os.makedirs(exp.path, exist_ok=True)
    return exp


# ---------------------------------------------------------------------------
# What gets recorded
# ---------------------------------------------------------------------------


def test_an_experiment_no_session_has_touched_knows_nothing(tmp_path: Path) -> None:
    """None, rather than a half-filled record.

    Creating an experiment happens in a dialog with no microscope, so the
    instrument genuinely is unknown until a run adopts it.
    """
    assert _experiment(tmp_path).session is None


def test_registering_records_the_instrument(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    experiment = _experiment(tmp_path)

    experiment.register_metadata(microscope)

    system = experiment.session.system
    assert system.serial_number == microscope.system.info.serial_number
    assert system.manufacturer == microscope.system.info.manufacturer
    assert system.model == microscope.system.info.model


def test_it_records_the_software_that_ran(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """`fibsem_revision` is the point -- the version alone cannot pin a dev build."""
    import fibsem

    experiment = _experiment(tmp_path)

    experiment.register_metadata(microscope)

    assert experiment.session.system.fibsem_version == fibsem.__version__
    # Set by _register_metadata onto the very SystemInfo snapshotted here, so this
    # also pins the ordering: collect after registration, or it reads None.
    assert experiment.session.system.application == "autolamella"


def test_it_records_installed_plugin_versions(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Shape, not contents: which plugins exist depends on the environment."""
    experiment = _experiment(tmp_path)

    experiment.register_metadata(microscope)

    plugins = experiment.session.plugins
    assert isinstance(plugins, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in plugins.items())


def test_the_system_info_is_copied_not_referenced(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """A record that changes after the fact is not a record.

    `microscope.system.info` is live -- registration writes `application` to it,
    and a driver may update it -- so holding the same object would let the
    experiment's account of a finished run be rewritten later.
    """
    experiment = _experiment(tmp_path)
    experiment.register_metadata(microscope)

    microscope.system.info.model = "SomethingElse"

    assert experiment.session.system.model != "SomethingElse"


# ---------------------------------------------------------------------------
# Who ran it
# ---------------------------------------------------------------------------


def test_without_a_declared_operator_it_falls_back_to_the_account(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    experiment = _experiment(tmp_path)

    experiment.register_metadata(microscope)

    from fibsem.structures import FibsemUser

    assert experiment.session.user.name == FibsemUser.from_environment().name


def test_a_declared_operator_beats_the_os_account(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Shared facility logins make the OS account name the workstation, not the
    person. Somebody who typed a name into the create dialog meant it."""
    experiment = _experiment(
        tmp_path, metadata={"user": "Dr Someone", "organisation": "Some Institute"}
    )

    experiment.register_metadata(microscope)

    assert experiment.session.user.name == "Dr Someone"
    assert experiment.session.user.organization == "Some Institute"


def test_the_hostname_still_comes_from_the_environment(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """It answers a different question -- which machine, not which person -- so a
    declared operator must not overwrite it."""
    from fibsem.structures import FibsemUser

    experiment = _experiment(tmp_path, metadata={"user": "Dr Someone"})

    experiment.register_metadata(microscope)

    assert experiment.session.user.hostname == FibsemUser.from_environment().hostname


def test_an_unrelated_metadata_entry_declares_nobody(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """Only the two fields the dialog offers count; a description is not a user."""
    from fibsem.structures import FibsemUser

    experiment = _experiment(tmp_path, metadata={"description": "a test run"})

    experiment.register_metadata(microscope)

    assert experiment.session.user.name == FibsemUser.from_environment().name


# ---------------------------------------------------------------------------
# Last run wins, and it reaches disk
# ---------------------------------------------------------------------------


def test_registering_again_records_the_later_session(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The chosen semantics: the latest session, not the first. Version drift is
    not lost by overwriting -- every image carries its own SystemInfo (FIB-445 D1)."""
    experiment = _experiment(tmp_path)
    experiment.register_metadata(microscope)
    first = experiment.session.system.software_version

    microscope.system.info.software_version = "9.9-after-the-upgrade"
    experiment.register_metadata(microscope)

    assert first != "9.9-after-the-upgrade"
    assert experiment.session.system.software_version == "9.9-after-the-upgrade"


def test_registering_persists_it(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """The value of this record is that it is *in experiment.yaml*.

    Leaving it to whatever saves next is how FIB-490 lost every failed task, so
    registration writes it itself.
    """
    experiment = _experiment(tmp_path)
    experiment.save()  # as `create` and `load` both guarantee

    experiment.register_metadata(microscope)

    written = _stored(experiment)
    assert written["session"]["system"]["serial_number"] == (
        microscope.system.info.serial_number
    )


def test_registering_does_not_conjure_a_file(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """An Experiment may legitimately exist without one -- constructing one does
    not touch the disk (FIB-420), and much of the suite drives TaskManager that
    way. Registering is not the moment to decide it should have a directory.
    """
    experiment = _experiment(tmp_path)

    experiment.register_metadata(microscope)

    assert not os.path.exists(os.path.join(experiment.path, "experiment.yaml"))
    # Still recorded in memory, so the next real save carries it.
    assert experiment.session is not None


def test_it_survives_a_round_trip(
    microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    experiment = _experiment(tmp_path, metadata={"user": "Dr Someone"})
    experiment.save()
    experiment.register_metadata(microscope)

    reloaded = Experiment.load(Path(experiment.path) / "experiment.yaml")

    assert reloaded.session.user.name == "Dr Someone"
    assert reloaded.session.system.model == experiment.session.system.model
    assert reloaded.session.plugins == experiment.session.plugins
    assert reloaded.session.recorded_at == experiment.session.recorded_at


def test_an_experiment_written_before_this_loads_with_none(tmp_path: Path) -> None:
    """Absent and unknown are the same answer, and None says it without guessing."""
    experiment = _experiment(tmp_path)
    experiment.save()

    stored = _stored(experiment)
    del stored["session"]
    with open(os.path.join(experiment.path, "experiment.yaml"), "w") as f:
        yaml.safe_dump(stored, f, indent=4)

    assert Experiment.load(Path(experiment.path) / "experiment.yaml").session is None


# ---------------------------------------------------------------------------
# The plugin collector
# ---------------------------------------------------------------------------


def test_only_installed_distributions_are_recorded(monkeypatch) -> None:
    """Having a distribution is the filter: built-ins have none, and a
    script-loaded extension carries a path instead."""
    from fibsem.plugins import report
    from fibsem.plugins.report import (
        Extension,
        ExtensionGroup,
        ExtensionSource,
        installed_plugin_versions,
    )

    group = ExtensionGroup(
        group="fibsem.patterns",
        label="Milling patterns",
        extensions=(
            Extension(
                group="fibsem.patterns",
                name="Trench",
                source=ExtensionSource.BUILTIN,
                target="fibsem.Trench",
            ),
            Extension(
                group="fibsem.patterns",
                name="Waffle",
                source=ExtensionSource.PLUGIN,
                target="lab.Waffle",
                distribution="lab-patterns",
                version="0.3.1",
            ),
            Extension(
                group="fibsem.patterns",
                name=None,
                source=ExtensionSource.FAILED,
                target="broken:Thing",
                distribution="broken-plugin",
                version="1.2",
                problem="ImportError",
            ),
        ),
    )
    monkeypatch.setattr(report, "collect_extensions", lambda: (group,))

    assert installed_plugin_versions() == {
        "lab-patterns": "0.3.1",
        # Installed, did not load, and shaped the run by being absent from the
        # registry -- omitting it would make the record match a machine that never
        # had it.
        "broken-plugin": "1.2",
    }


def test_a_distribution_without_a_version_is_still_recorded(monkeypatch) -> None:
    """An editable install can resolve a name but no version. Recording "unknown"
    beats dropping the plugin from the record entirely."""
    from fibsem.plugins import report
    from fibsem.plugins.report import (
        Extension,
        ExtensionGroup,
        ExtensionSource,
        installed_plugin_versions,
    )

    group = ExtensionGroup(
        group="fibsem.patterns",
        label="Milling patterns",
        extensions=(
            Extension(
                group="fibsem.patterns",
                name="Waffle",
                source=ExtensionSource.PLUGIN,
                target="lab.Waffle",
                distribution="lab-patterns",
                version=None,
            ),
        ),
    )
    monkeypatch.setattr(report, "collect_extensions", lambda: (group,))

    assert installed_plugin_versions() == {"lab-patterns": "unknown"}
