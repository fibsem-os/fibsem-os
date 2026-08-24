"""Tests for the spot burn fiducial task: run_spot_burn hardening, config
numeric coercion, and the protocol-editor parameter-widget dispatch.

These cover the failure modes that crashed the unsupervised (automatic) path:
- numeric parameters arriving as strings (from a plain QLineEdit in the editor),
- spot-burn coordinates outside the 0-1 image bounds reaching set_spot on hardware,
- the widget factory rendering float/int fields as a text box instead of a spinbox.
"""

from unittest.mock import MagicMock

import pytest

from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.imaging.spot import SpotBurnSettings, run_spot_burn
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point

# The widget-dispatch tests need the UI stack (napari/PyQt5), which isn't installed
# in the core CI env (`pip install .`). Probe the third-party packages themselves,
# and import from fibsem unguarded: wrapping the fibsem import in the try instead
# also swallows a renamed symbol, and reports it as a missing dependency on a
# machine that has the whole UI stack -- which left these tests silently skipped
# everywhere once the parameter-widget classes were replaced (FIB-526/FIB-384).
try:
    import napari  # noqa: F401
    import superqt  # noqa: F401
    from PyQt5 import QtWidgets  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only in the no-UI CI env
    _MISSING_UI_DEPS = str(exc)
else:
    _MISSING_UI_DEPS = ""
    from fibsem.applications.autolamella.ui.autolamella_task_config_widget import (
        AutoLamellaTaskParametersConfigWidget,
        resolve_field_types,
    )
    from fibsem.ui.widgets.custom_widgets import IntegerValueSpinBox, ValueSpinBox

requires_ui = pytest.mark.skipif(
    bool(_MISSING_UI_DEPS), reason=f"UI dependencies not installed: {_MISSING_UI_DEPS}"
)

IMAGING_CURRENT = 20e-12


@pytest.fixture
def mock_microscope():
    """A microscope stub that records the calls run_spot_burn makes."""
    mic = MagicMock()
    mic.get_beam_current.return_value = IMAGING_CURRENT
    return mic


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip the real exposure countdown so tests run instantly."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


def _burned_points(mic: MagicMock) -> list:
    """The points passed to set_spot_scanning_mode, in order."""
    return [c.kwargs["point"] for c in mic.set_spot_scanning_mode.call_args_list]


# --- run_spot_burn: coordinate bounds filtering ---------------------------------


def test_run_spot_burn_filters_out_of_bounds_coordinates(mock_microscope):
    """Coordinates outside the 0-1 image bounds are skipped, not sent to set_spot."""
    coords = [
        Point(0.5, 0.5),   # valid
        Point(0.9, 0.2),   # valid
        Point(1.02, 0.5),  # x > 1
        Point(-0.1, 0.3),  # x < 0
        Point(0.5, 1.5),   # y > 1
    ]
    FibsemMicroscope.run_spot_burn(
        mock_microscope,
        settings=SpotBurnSettings(coordinates=coords, exposure_time=1.0,
                                  milling_current=30e-12),
        beam_type=BeamType.ION,
    )
    assert _burned_points(mock_microscope) == [Point(0.5, 0.5), Point(0.9, 0.2)]


def test_run_spot_burn_keeps_boundary_coordinates(mock_microscope):
    """Exact 0 and 1 boundaries are inclusive."""
    coords = [Point(0.0, 0.0), Point(1.0, 1.0)]
    FibsemMicroscope.run_spot_burn(
        mock_microscope,
        settings=SpotBurnSettings(coordinates=coords, exposure_time=1.0,
                                  milling_current=30e-12),
    )
    assert _burned_points(mock_microscope) == coords


def test_run_spot_burn_empty_coordinates_does_not_burn(mock_microscope):
    """No coordinates -> no spot exposures, but the beam state is still restored."""
    FibsemMicroscope.run_spot_burn(
        mock_microscope,
        settings=SpotBurnSettings(coordinates=[], exposure_time=1.0,
                                  milling_current=30e-12),
    )
    mock_microscope.set_spot_scanning_mode.assert_not_called()
    mock_microscope.set_full_frame_scanning_mode.assert_called_once()


# --- run_spot_burn: string parameter coercion -----------------------------------


def test_run_spot_burn_coerces_string_parameters(mock_microscope):
    """String milling_current/exposure_time (from the editor QLineEdit bug) don't crash."""
    FibsemMicroscope.run_spot_burn(
        mock_microscope,
        settings=SpotBurnSettings(coordinates=[Point(0.5, 0.5)], exposure_time="2",
                                  milling_current="3e-11"),
        beam_type=BeamType.ION,
    )
    # the milling current is applied as a real float, not the string "3e-11"
    first_current = mock_microscope.set_beam_current.call_args_list[0].kwargs["current"]
    assert isinstance(first_current, float)
    assert first_current == pytest.approx(3e-11)


# --- run_spot_burn: beam state restoration --------------------------------------


def test_run_spot_burn_restores_full_frame_and_imaging_current(mock_microscope):
    """After burning, scanning returns to full frame and the imaging current is restored."""
    FibsemMicroscope.run_spot_burn(
        mock_microscope,
        settings=SpotBurnSettings(coordinates=[Point(0.5, 0.5)], exposure_time=1.0,
                                  milling_current=30e-12),
    )
    mock_microscope.set_full_frame_scanning_mode.assert_called_once()
    last_current = mock_microscope.set_beam_current.call_args_list[-1].kwargs["current"]
    assert last_current == IMAGING_CURRENT


# --- run_spot_burn: progress reporting -----------------------------------------


def test_run_spot_burn_emits_progress_via_microscope(mock_microscope):
    """Progress is reported through microscope.spot_burn_progress_signal (both run paths)."""
    FibsemMicroscope.run_spot_burn(
        mock_microscope,
        settings=SpotBurnSettings(coordinates=[Point(0.5, 0.5), Point(0.6, 0.6)],
                                  exposure_time=1.0, milling_current=30e-12),
    )
    emitted = [
        c.args[0] for c in mock_microscope.spot_burn_progress_signal.emit.call_args_list
    ]
    # initial progress reports the total number of points
    assert emitted[0]["current_point"] == 0
    assert emitted[0]["total_points"] == 2
    # final emission signals completion
    assert emitted[-1] == {"finished": True}


def test_run_spot_burn_module_function_delegates_to_the_microscope(mock_microscope):
    """The module entry point dispatches polymorphically (FIB-297): the default
    implementation parks the beam per point, TESCAN overrides with DrawBeam dots."""
    settings = SpotBurnSettings(coordinates=[Point(0.5, 0.5)], exposure_time=1.0,
                                milling_current=30e-12)
    run_spot_burn(microscope=mock_microscope, settings=settings,
                  beam_type=BeamType.ION, stop_event=None)
    mock_microscope.run_spot_burn.assert_called_once_with(
        settings=settings, beam_type=BeamType.ION, stop_event=None
    )


@requires_ui
def test_build_spot_burn_progress_update_mapping():
    """Progress dicts map to ProgressUpdate: running, done, and failed states."""
    from fibsem.ui.FibsemSpotBurnWidget import build_spot_burn_progress_update

    running = build_spot_burn_progress_update(
        {"current_point": 1, "total_points": 3,
         "total_remaining_time": 20.0, "total_estimated_time": 30.0}
    )
    assert (running.current, running.total) == (1, 3)
    assert running.remaining_seconds == 20.0
    assert not running.finished

    done = build_spot_burn_progress_update({"finished": True})
    assert done.finished and not done.message

    failed = build_spot_burn_progress_update({"finished": True, "error": True})
    assert failed.finished and failed.message == "Spot burn failed"


# --- SpotBurnFiducialTask.update_spot_burn_parameters_ui ------------------------


def _make_headless_spot_burn_task(coordinates, tmp_path):
    """A SpotBurnFiducialTask with no parent UI (unsupervised/headless path)."""
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
        SpotBurnFiducialTask,
    )

    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    config = SpotBurnFiducialTaskConfig(
        task_name="Spot Burn Fiducial", coordinates=coordinates
    )
    return SpotBurnFiducialTask(
        microscope=MagicMock(), config=config, lamella=lamella, parent_ui=None
    )


def test_update_spot_burn_ui_skips_when_no_coordinates(monkeypatch, tmp_path):
    """Unsupervised/headless with no coordinates skips, rather than blocking on ask_user."""
    # patch where it is *used*: the task imports run_spot_burn at module level, so
    # patching fibsem.imaging.spot would leave the task's own binding untouched.
    import fibsem.applications.autolamella.workflows.tasks.spot_burn as sb_mod

    calls = []
    monkeypatch.setattr(sb_mod, "run_spot_burn", lambda **kw: calls.append(kw))

    task = _make_headless_spot_burn_task([], tmp_path)
    task.update_spot_burn_parameters_ui()  # must return, not hang

    assert calls == []


def test_update_spot_burn_ui_runs_stored_coordinates_headless(monkeypatch, tmp_path):
    """Unsupervised/headless with coordinates burns them directly."""
    import fibsem.applications.autolamella.workflows.tasks.spot_burn as sb_mod

    calls = []
    monkeypatch.setattr(sb_mod, "run_spot_burn", lambda **kw: calls.append(kw))

    coords = [Point(0.5, 0.5), Point(0.6, 0.6)]
    task = _make_headless_spot_burn_task(coords, tmp_path)
    task.update_spot_burn_parameters_ui()

    assert len(calls) == 1
    settings = calls[0]["settings"]
    assert settings.coordinates == coords
    assert settings.milling_current == task.config.milling_current
    assert settings.exposure_time == task.config.exposure_time


# --- SpotBurnFiducialTask supervised loop ---------------------------------------


def _make_supervised_spot_burn_task(monkeypatch, tmp_path, ask_user_responses):
    """A supervised SpotBurnFiducialTask with a mock parent UI + spot burn widget.

    ask_user_responses is an iterable of the booleans ask_user should return.
    Returns (task, widget, cleared_list).
    """
    import fibsem.applications.autolamella.workflows.tasks.spot_burn as sb_mod
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
        SpotBurnFiducialTask,
    )

    widget = MagicMock()
    widget.is_burning = False  # each burn "completes" immediately
    # the task now reads back the whole settings object (coordinates + current +
    # exposure), not just the coordinates — see update_spot_burn_parameters' typed contract
    widget.get_settings.return_value = SpotBurnSettings(
        coordinates=[Point(0.5, 0.5)], exposure_time=1.0, milling_current=1e-9
    )
    parent_ui = MagicMock()  # truthy experiment.task_protocol.get_supervision -> supervised
    parent_ui.spot_burn_widget = widget

    responses = iter(ask_user_responses)
    monkeypatch.setattr(sb_mod, "ask_user", lambda *a, **k: next(responses))
    monkeypatch.setattr(sb_mod, "update_spot_burn_parameters", lambda **k: None)
    cleared = []
    monkeypatch.setattr(sb_mod, "clear_spot_burn_ui", lambda ui: cleared.append(ui))

    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    config = SpotBurnFiducialTaskConfig(
        task_name="Spot Burn Fiducial", coordinates=[Point(0.5, 0.5)]
    )
    task = SpotBurnFiducialTask(
        microscope=MagicMock(), config=config, lamella=lamella, parent_ui=parent_ui
    )
    task.update_status_ui = lambda *a, **k: None  # isolate the loop
    task._check_for_abort = lambda: None
    return task, widget, cleared


def test_supervised_spot_burn_runs_then_continues(monkeypatch, tmp_path):
    """'Run Spot Burn' triggers the widget and waits; then Continue proceeds."""
    task, widget, cleared = _make_supervised_spot_burn_task(
        monkeypatch, tmp_path, ask_user_responses=[True, False]
    )

    task.update_spot_burn_parameters_ui()

    widget.start_spot_burn_signal.emit.assert_called_once()
    widget.get_settings.assert_called_once()
    assert cleared  # spot burn UI was cleared afterwards


def test_supervised_spot_burn_continue_without_running(monkeypatch, tmp_path):
    """Pressing Continue immediately runs no burn but still reads back + clears."""
    task, widget, cleared = _make_supervised_spot_burn_task(
        monkeypatch, tmp_path, ask_user_responses=[False]
    )

    task.update_spot_burn_parameters_ui()

    widget.start_spot_burn_signal.emit.assert_not_called()
    widget.get_settings.assert_called_once()
    assert cleared


def test_supervised_spot_burn_abort_cancels_burn(monkeypatch, tmp_path):
    """Workflow abort during the wait loop cancels the in-progress burn.

    Also covers the stop-vs-start race: a burn that starts after the workflow
    Stop already ran cancel_spot_burn (clearing its stop_event) is still taken
    down by the aborting task.
    """
    task, widget, cleared = _make_supervised_spot_burn_task(
        monkeypatch, tmp_path, ask_user_responses=[True]
    )
    widget.is_burning = True  # burn in progress
    task._check_for_abort = MagicMock(side_effect=InterruptedError("aborted"))

    with pytest.raises(InterruptedError):
        task.update_spot_burn_parameters_ui()

    widget.cancel_spot_burn.assert_called_once()
    assert not cleared  # abort propagates before the UI cleanup runs


# --- SpotBurnFiducialTaskConfig serialization -----------------------------------


def test_from_dict_coerces_string_numeric_params():
    """Protocols saved with string-typed params (pre-fix) are repaired on load."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial")
    d = cfg.to_dict()
    d["parameters"]["milling_current"] = "3e-11"
    d["parameters"]["exposure_time"] = "10"

    loaded = SpotBurnFiducialTaskConfig.from_dict(d)

    assert isinstance(loaded.milling_current, float)
    assert loaded.milling_current == pytest.approx(3e-11)
    assert isinstance(loaded.exposure_time, int)
    assert loaded.exposure_time == 10


def test_to_from_dict_preserves_coordinates_as_points():
    """Coordinates round-trip as Point objects (types stay consistent across save/load)."""
    cfg = SpotBurnFiducialTaskConfig(
        task_name="Spot Burn Fiducial",
        coordinates=[Point(0.5, 0.5), Point(0.9, 0.2)],
    )
    loaded = SpotBurnFiducialTaskConfig.from_dict(cfg.to_dict())

    assert all(isinstance(c, Point) for c in loaded.coordinates)
    assert loaded.coordinates[0].x == pytest.approx(0.5)
    assert loaded.coordinates[1].y == pytest.approx(0.2)


# --- widget factory: annotation resolution --------------------------------------


@requires_ui
def test_resolve_field_types_resolves_future_annotations():
    """`from __future__ import annotations` string types resolve to concrete types."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial")
    hints = resolve_field_types(cfg)

    assert hints["milling_current"] is float
    assert hints["exposure_time"] is int


def _control_widget(widget, field: str):
    """The input widget the form built for *field*.

    The per-type parameter widget classes this file used to reach for were
    replaced by the shared control builder (FIB-526/FIB-384): the form now keeps
    a row per field, each holding a `Control` whose `.widget` is the input.
    """
    row = next(r for r in widget._rows if r.field == field)
    return row.control.widget


@requires_ui
def test_spot_burn_params_render_as_spinboxes(qapp):
    """Regression: milling_current/exposure_time render as spinboxes, not QLineEdits."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial")
    widget = AutoLamellaTaskParametersConfigWidget(cfg)

    assert isinstance(_control_widget(widget, "milling_current"), ValueSpinBox)
    assert isinstance(_control_widget(widget, "exposure_time"), IntegerValueSpinBox)


@requires_ui
def test_exposure_time_spinbox_has_seconds_suffix(qapp):
    """exposure_time's units metadata ('s') is shown as a spinbox suffix."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial")
    widget = AutoLamellaTaskParametersConfigWidget(cfg)

    assert _control_widget(widget, "exposure_time").suffix() == " s"
