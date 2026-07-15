"""Tests for the spot burn fiducial task: run_spot_burn hardening, config
numeric coercion, and the protocol-editor parameter-widget dispatch.

These cover the failure modes that crashed the unsupervised (automatic) path:
- numeric parameters arriving as strings (from a plain QLineEdit in the editor),
- spot-burn coordinates outside the 0-1 image bounds reaching set_spot on hardware,
- the widget factory rendering float/int fields as a text box instead of a spinbox.
"""

import os
from unittest.mock import MagicMock

import pytest

from fibsem.imaging.spot import run_spot_burn
from fibsem.structures import BeamType, Point
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)

# The widget-dispatch tests need the UI stack (napari/PyQt5), which isn't installed
# in the core CI env (`pip install .`). Guard the import so they skip there instead
# of erroring; the run_spot_burn and config tests below have no UI dependency.
try:
    from fibsem.ui.widgets.autolamella_task_config_widget import (
        AutoLamellaTaskParametersConfigWidget,
        FloatParameterWidget,
        IntParameterWidget,
        resolve_field_types,
    )
    _HAS_UI_DEPS = True
except Exception:  # pragma: no cover - exercised only in the no-UI CI env
    _HAS_UI_DEPS = False

requires_ui = pytest.mark.skipif(
    not _HAS_UI_DEPS, reason="UI dependencies (napari/PyQt5) not installed"
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
    import fibsem.imaging.spot as spot
    monkeypatch.setattr(spot.time, "sleep", lambda *_: None)


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
    run_spot_burn(
        microscope=mock_microscope,
        coordinates=coords,
        exposure_time=1.0,
        milling_current=30e-12,
        beam_type=BeamType.ION,
    )
    assert _burned_points(mock_microscope) == [Point(0.5, 0.5), Point(0.9, 0.2)]


def test_run_spot_burn_keeps_boundary_coordinates(mock_microscope):
    """Exact 0 and 1 boundaries are inclusive."""
    coords = [Point(0.0, 0.0), Point(1.0, 1.0)]
    run_spot_burn(
        microscope=mock_microscope,
        coordinates=coords,
        exposure_time=1.0,
        milling_current=30e-12,
    )
    assert _burned_points(mock_microscope) == coords


def test_run_spot_burn_empty_coordinates_does_not_burn(mock_microscope):
    """No coordinates -> no spot exposures, but the beam state is still restored."""
    run_spot_burn(
        microscope=mock_microscope,
        coordinates=[],
        exposure_time=1.0,
        milling_current=30e-12,
    )
    mock_microscope.set_spot_scanning_mode.assert_not_called()
    mock_microscope.set_full_frame_scanning_mode.assert_called_once()


# --- run_spot_burn: string parameter coercion -----------------------------------


def test_run_spot_burn_coerces_string_parameters(mock_microscope):
    """String milling_current/exposure_time (from the editor QLineEdit bug) don't crash."""
    run_spot_burn(
        microscope=mock_microscope,
        coordinates=[Point(0.5, 0.5)],
        exposure_time="2",
        milling_current="3e-11",
        beam_type=BeamType.ION,
    )
    # the milling current is applied as a real float, not the string "3e-11"
    first_current = mock_microscope.set_beam_current.call_args_list[0].kwargs["current"]
    assert isinstance(first_current, float)
    assert first_current == pytest.approx(3e-11)


# --- run_spot_burn: beam state restoration --------------------------------------


def test_run_spot_burn_restores_full_frame_and_imaging_current(mock_microscope):
    """After burning, scanning returns to full frame and the imaging current is restored."""
    run_spot_burn(
        microscope=mock_microscope,
        coordinates=[Point(0.5, 0.5)],
        exposure_time=1.0,
        milling_current=30e-12,
    )
    mock_microscope.set_full_frame_scanning_mode.assert_called_once()
    last_current = mock_microscope.set_beam_current.call_args_list[-1].kwargs["current"]
    assert last_current == IMAGING_CURRENT


# --- run_spot_burn: progress reporting -----------------------------------------


def test_run_spot_burn_emits_progress_via_microscope(mock_microscope):
    """Progress is reported through microscope.spot_burn_progress_signal (both run paths)."""
    run_spot_burn(
        microscope=mock_microscope,
        coordinates=[Point(0.5, 0.5), Point(0.6, 0.6)],
        exposure_time=1.0,
        milling_current=30e-12,
    )
    emitted = [
        c.args[0] for c in mock_microscope.spot_burn_progress_signal.emit.call_args_list
    ]
    # initial progress reports the total number of points
    assert emitted[0]["current_point"] == 0
    assert emitted[0]["total_points"] == 2
    # final emission signals completion
    assert emitted[-1] == {"finished": True}


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
    import fibsem.imaging.spot as spot_mod

    calls = []
    monkeypatch.setattr(spot_mod, "run_spot_burn", lambda **kw: calls.append(kw))

    task = _make_headless_spot_burn_task([], tmp_path)
    task.update_spot_burn_parameters_ui()  # must return, not hang

    assert calls == []


def test_update_spot_burn_ui_runs_stored_coordinates_headless(monkeypatch, tmp_path):
    """Unsupervised/headless with coordinates burns them directly."""
    import fibsem.imaging.spot as spot_mod

    calls = []
    monkeypatch.setattr(spot_mod, "run_spot_burn", lambda **kw: calls.append(kw))

    coords = [Point(0.5, 0.5), Point(0.6, 0.6)]
    task = _make_headless_spot_burn_task(coords, tmp_path)
    task.update_spot_burn_parameters_ui()

    assert len(calls) == 1
    assert calls[0]["coordinates"] == coords


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


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - environment without Qt
        pytest.skip(f"PyQt5 not available: {exc}")
    return QApplication.instance() or QApplication([])


@requires_ui
def test_spot_burn_params_render_as_spinboxes(qapp):
    """Regression: milling_current/exposure_time render as spinboxes, not QLineEdits."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial")
    widget = AutoLamellaTaskParametersConfigWidget(cfg)

    assert isinstance(widget.parameter_widgets["milling_current"], FloatParameterWidget)
    assert isinstance(widget.parameter_widgets["exposure_time"], IntParameterWidget)


@requires_ui
def test_exposure_time_spinbox_has_seconds_suffix(qapp):
    """exposure_time's units metadata ('s') is shown as a spinbox suffix."""
    cfg = SpotBurnFiducialTaskConfig(task_name="Spot Burn Fiducial")
    widget = AutoLamellaTaskParametersConfigWidget(cfg)

    exposure_widget = widget.parameter_widgets["exposure_time"]
    assert exposure_widget.widget.suffix() == " s"
