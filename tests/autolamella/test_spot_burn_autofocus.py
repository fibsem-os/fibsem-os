"""Tests for the FIB autofocus step in SpotBurnFiducialTask (FIB-378).

Covers the config field round-trip, the ordering (focus before the reference image
the points are placed on), and the `_run_autofocus` helper — which until now read
`self.image_settings`, an attribute the base class never defines.
"""

from unittest.mock import MagicMock

import pytest

from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.structures import BeamType, Point

# --- config -----------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_autofocus_survives_a_config_round_trip(value):
    """to_dict/from_dict are hand-written here — a new field is easy to drop."""
    config = SpotBurnFiducialTaskConfig(task_name="Spot Burn", autofocus=value)
    assert SpotBurnFiducialTaskConfig.from_dict(config.to_dict()).autofocus is value


def test_autofocus_is_off_by_default():
    """Opt-in: adds a sweep to every spot burn run, and the ranges are unvalidated."""
    assert SpotBurnFiducialTaskConfig(task_name="Spot Burn").autofocus is False


def test_autofocus_defaults_off_for_older_protocols():
    config = SpotBurnFiducialTaskConfig.from_dict(
        {"task_type": "SPOT_BURN_FIDUCIAL", "parameters": {}}
    )
    assert config.autofocus is False


def test_autofocus_is_exposed_as_a_task_parameter():
    """The parameters form is generated from this tuple, so the checkbox comes free."""
    assert "autofocus" in SpotBurnFiducialTaskConfig(task_name="Spot Burn").parameters


def test_autofocus_is_not_part_of_the_run_payload():
    """SpotBurnSettings is what to burn; autofocus is a workflow concern (see #211)."""
    settings = SpotBurnFiducialTaskConfig(
        task_name="Spot Burn", autofocus=True
    ).to_settings()
    assert not hasattr(settings, "autofocus")


# --- task ordering ----------------------------------------------------------


def _make_task(tmp_path, *, autofocus: bool):
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
        SpotBurnFiducialTask,
    )

    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    config = SpotBurnFiducialTaskConfig(
        task_name="Spot Burn Fiducial",
        autofocus=autofocus,
        coordinates=[Point(0.5, 0.5)],
    )
    return SpotBurnFiducialTask(
        microscope=MagicMock(), config=config, lamella=lamella, parent_ui=None
    )


def _record_run(task, calls):
    """Stub out everything _run does except the autofocus/reference-image ordering."""
    task._move_to_milling_pose = lambda: calls.append(("move", None))
    task._align_reference_image = lambda *a, **k: calls.append(("align", None))
    task._run_autofocus = lambda beam_type, hfw=None: calls.append(
        ("autofocus", (beam_type, hfw))
    )
    task._acquire_reference_image = lambda *a, **k: calls.append(
        ("reference", k.get("field_of_view"))
    )
    task._acquire_set_of_reference_images = lambda *a, **k: calls.append(
        ("final", None)
    )
    task.update_spot_burn_parameters_ui = lambda: calls.append(("burn", None))
    task.log_status_message = lambda *a, **k: None


def test_autofocus_runs_before_the_reference_image(tmp_path):
    """The points are placed on that image, so focusing after it would be too late."""
    task = _make_task(tmp_path, autofocus=True)
    calls = []
    _record_run(task, calls)

    task._run()

    steps = [name for name, _ in calls]
    assert steps.index("autofocus") < steps.index("reference")
    assert steps.index("autofocus") < steps.index("burn")


def test_autofocus_uses_the_reference_field_of_view(tmp_path):
    """Sweep at the field of view the task images at, not some unrelated hfw."""
    task = _make_task(tmp_path, autofocus=True)
    calls = []
    _record_run(task, calls)

    task._run()

    args = dict(calls)["autofocus"]
    assert args == (BeamType.ION, task.config.reference_imaging.field_of_view1)
    assert dict(calls)["reference"] == task.config.reference_imaging.field_of_view1


def test_no_autofocus_when_disabled(tmp_path):
    task = _make_task(tmp_path, autofocus=False)
    calls = []
    _record_run(task, calls)

    task._run()

    assert "autofocus" not in [name for name, _ in calls]


# --- _run_autofocus helper --------------------------------------------------


def _make_autofocus_task(tmp_path, monkeypatch):
    """A task with run_auto_focus stubbed, returning the kwargs it was called with."""
    import fibsem.autofunctions.autofocus as af

    task = _make_task(tmp_path, autofocus=True)
    task.log_status_message = lambda *a, **k: None
    (tmp_path / "lam").mkdir(parents=True, exist_ok=True)

    calls = []

    def _fake_run_auto_focus(microscope, **kwargs):
        calls.append(kwargs)
        return MagicMock(working_distance=16.5e-3, focus_score=1.0)

    monkeypatch.setattr(af, "run_auto_focus", _fake_run_auto_focus)
    return task, calls


def test_run_autofocus_defaults_hfw_from_the_task_config(tmp_path, monkeypatch):
    """Regression: this read self.image_settings, which AutoLamellaTask never defines.

    Only select_position and rough create that attribute, and only partway through
    their own _run() — so the default-hfw path raised AttributeError everywhere else,
    including in the spot burn task this feature targets.
    """
    task, calls = _make_autofocus_task(tmp_path, monkeypatch)

    task._run_autofocus(BeamType.ION)  # no explicit hfw

    assert calls[0]["hfw"] == task.config.imaging.hfw


def test_config_imaging_hands_back_the_stored_object_not_a_copy():
    """The invariant that makes swapping self.image_settings for config.imaging safe.

    select_position captures `self.image_settings = self.config.imaging` and the
    acquisition helpers then mutate `.hfw` on it. Reading config.imaging instead is
    only behaviour-preserving because the property returns that same object — if it
    ever starts returning a copy, the default hfw silently diverges from the field of
    view the task is actually imaging at.
    """
    config = SpotBurnFiducialTaskConfig(task_name="Spot Burn")
    assert config.imaging is config.reference_imaging.imaging
    assert config.imaging is config.imaging

    captured = config.imaging  # what select_position.py:58 does
    captured.hfw = 1.234e-4  # what _acquire_channels does
    assert config.imaging.hfw == 1.234e-4


def test_run_autofocus_honours_an_explicit_zero_hfw(tmp_path, monkeypatch):
    """`hfw or default` silently replaced 0; `hfw is None` does not."""
    task, calls = _make_autofocus_task(tmp_path, monkeypatch)

    task._run_autofocus(BeamType.ION, hfw=0)

    assert calls[0]["hfw"] == 0


def test_run_autofocus_passes_the_stop_event(tmp_path, monkeypatch):
    """run_auto_focus became cancellable in #210; the task has to opt in."""
    import threading

    task, calls = _make_autofocus_task(tmp_path, monkeypatch)
    task._stop_event = threading.Event()

    task._run_autofocus(BeamType.ION, hfw=150e-6)

    assert calls[0]["stop_event"] is task._stop_event


def test_run_autofocus_lets_cancellation_propagate(tmp_path, monkeypatch):
    """_is_cancellation turns this into "cancelled", not "failed" — don't swallow it."""
    import fibsem.autofunctions.autofocus as af
    from fibsem.cancellation import OperationCancelledError

    task, _ = _make_autofocus_task(tmp_path, monkeypatch)
    monkeypatch.setattr(
        af, "run_auto_focus", MagicMock(side_effect=OperationCancelledError("stopped"))
    )

    with pytest.raises(OperationCancelledError):
        task._run_autofocus(BeamType.ION, hfw=150e-6)

    assert task._is_cancellation(OperationCancelledError("stopped")) is True


# --- the None skip (FIB-508: backends that cannot set the working distance) --


def test_run_autofocus_skips_cleanly_when_the_sweep_is_declined(tmp_path, monkeypatch):
    """run_auto_focus returns None where the working distance is not settable
    (TESCAN ION). The task must say "skipped" and save nothing — before the guard
    this was an AttributeError on result.save, killing the task the first time a
    Tescan protocol enabled autofocus. A placeholder result instead of None would
    be worse: it would write a completed-looking artifact directory and log a
    working distance that was never applied."""
    import os

    import fibsem.autofunctions.autofocus as af

    task = _make_task(tmp_path, autofocus=True)
    (tmp_path / "lam").mkdir(parents=True, exist_ok=True)
    messages = []
    task.log_status_message = lambda *a, **k: messages.append((a, k))
    monkeypatch.setattr(af, "run_auto_focus", lambda microscope, **kwargs: None)

    task._run_autofocus(BeamType.ION, hfw=150e-6)  # must not raise

    assert any("skipped" in str(m).lower() for m in messages)
    assert not os.path.exists(os.path.join(task.lamella.path, "autofunctions"))


def test_run_autofocus_still_saves_when_the_sweep_ran(tmp_path, monkeypatch):
    """The guard must not swallow real results: a returned result is still saved."""
    import fibsem.autofunctions.autofocus as af

    task = _make_task(tmp_path, autofocus=True)
    (tmp_path / "lam").mkdir(parents=True, exist_ok=True)
    task.log_status_message = lambda *a, **k: None
    result = MagicMock(working_distance=16.5e-3, focus_score=1.0)
    monkeypatch.setattr(af, "run_auto_focus", lambda microscope, **kwargs: result)

    task._run_autofocus(BeamType.ION, hfw=150e-6)

    result.save.assert_called_once()
