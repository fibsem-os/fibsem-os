"""The coincidence viewer's monitor mode (FIB-912).

A supervised queued coincidence mill runs through the main window's milling
widget; the viewer attaches to the strategies of the config being run and
watches. It never launches the run, its Stop is the run's Stop, and exit puts
the manual controls back.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

import yaml
from PyQt5.QtWidgets import QApplication

from fibsem.structures import FibsemRectangle, Point


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def viewer(qapp, tmp_path, monkeypatch):
    from fibsem import config as cfg
    from fibsem import utils
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget,
    )

    monkeypatch.setattr(
        cfg, "COINCIDENCE_MILLING_CONFIG_PATH", str(tmp_path / "cmc.yaml")
    )
    with open(os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=str(path))
    experiment = Experiment(path=str(tmp_path), name="monitor-mode-test")
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    experiment.positions.append(lamella)
    widget = FluorescenceCoincidenceViewerWidget(
        microscope=microscope, experiment=experiment
    )
    widget.set_fm_image(microscope.fm.acquire_image())
    yield widget
    widget.close()
    microscope.disconnect()


def _running_config(bbox=None, supervised=True, drop=0.4):
    from copy import deepcopy

    from fibsem.applications.autolamella.workflows._default_milling_config import (
        DEFAULT_MILLING_CONFIG,
    )
    from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
        MILL_COINCIDENT_KEY,
    )

    config = deepcopy(DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY])
    for stage in config.enabled_stages:
        stage.pattern.point = Point(1.0e-6, 0.5e-6)
        stage.strategy.config.bbox = bbox
        stage.strategy.config.supervised = supervised
        stage.strategy.config.intensity_drop_fraction = drop
    return config


def _stats(value, peak, threshold, drop=False, below=0, timeout=1200.0):
    return {
        "value": value,
        "rolling_mean": value,
        "peak_rolling_mean": peak,
        "threshold_value": threshold,
        "warmup_complete": True,
        "drop_detected": drop,
        "drop_fraction": value / peak if peak else 1.0,
        "threshold_fraction": 0.6,
        "consecutive_count": 10,
        "below_threshold_count": below,
        "timeout_remaining": timeout,
        "elapsed_time": 372.0,
    }


def test_enter_attaches_to_the_running_strategies(viewer, qapp):
    config = _running_config(bbox=FibsemRectangle(0.3, 0.3, 0.2, 0.2), drop=0.35)
    stops = []

    viewer.enter_monitor_mode(
        config, on_stop=lambda: stops.append(1), title="Coincident Milling"
    )
    qapp.processEvents()

    assert viewer.in_monitor_mode
    assert viewer._is_milling_active
    strategy = config.enabled_stages[0].strategy
    assert viewer._active_strategies == [strategy]
    # controls seeded from the live strategy's config, not the other way round
    assert viewer.spin_drop_threshold.value() == 35
    assert viewer._supervised is True
    # the boxes as the run has them
    H, W = viewer.fm_canvas._img_shape
    rect = viewer.fm_canvas.rect_overlay.get_rect()
    assert rect["x0"] == pytest.approx(0.3 * W, abs=1)
    # nothing but the run's controls: the site is locked, setup buttons away
    assert not viewer.lamella_list_widget.isEnabled()
    assert not viewer.btn_setup_continue.isVisible()

    # the task-mode chrome: the run is not ours, the box is shown dashed
    assert viewer.label_task_lock.text() == "Task owns this run"
    assert viewer.label_task_lock.isVisible()
    assert viewer.fib_canvas.rect_overlay._linestyle == "--"

    # live stats reach the panel: a drop turns the chip orange, and the run
    # panel says what the drop does, how close the latch is, and the timeout
    strategy.intensity_stats_signal.emit(
        _stats(1000.0, 1800.0, 1080.0, drop=True, below=7, timeout=1428.0)
    )
    qapp.processEvents()
    assert "Intensity Drop" in viewer.label_threshold_chip.text()
    metrics = viewer._info_widget._run_metrics_label.text()
    assert "Supervised · drop alerts, you stop" in metrics
    assert "Below thr: 7 / 10 frames" in metrics
    assert "Timeout  : in 0:23:48" in metrics
    # flipping the mode is reflected in the panel
    viewer._set_supervised(False)
    assert (
        "Automated · drop stops the mill"
        in viewer._info_widget._run_metrics_label.text()
    )


def test_stop_is_the_runs_stop_and_the_toggle_reaches_the_strategy(viewer, qapp):
    config = _running_config()
    stops = []
    viewer.enter_monitor_mode(config, on_stop=lambda: stops.append("stop"))
    # the run's chrome arrives via the microscope's progress signal as for any mill
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    viewer._on_milling_progress(
        MillingProgress(
            status=MillingProgressStatus.STAGE_STARTED,
            stage_name="Coincident Milling 01",
            current_stage=0,
            total_stages=1,
        )
    )
    qapp.processEvents()
    assert viewer.btn_milling.text() == "Stop Milling"

    # Supervised → off lets the latch stop the mill: it lands on the live config
    strategy = config.enabled_stages[0].strategy
    viewer._set_supervised(False)
    assert strategy.config.supervised is False
    viewer.spin_drop_threshold.setValue(50)
    assert strategy.config.intensity_drop_fraction == pytest.approx(0.5)

    viewer.btn_milling.click()
    assert stops == ["stop"]
    # the viewer's own milling widget was never asked to run anything
    assert not viewer.milling_viewer_widget.milling_widget.is_milling


def test_exit_detaches_and_restores_the_manual_controls(viewer, qapp):
    manual_name = viewer.milling_viewer_widget.get_config().name
    config = _running_config()
    viewer.enter_monitor_mode(config)
    strategy = config.enabled_stages[0].strategy

    viewer.exit_monitor_mode()
    qapp.processEvents()

    assert not viewer.in_monitor_mode
    assert not viewer._is_milling_active
    assert viewer._active_strategies == []
    assert viewer.btn_milling.text() == "Start Milling"
    assert not viewer.label_task_lock.isVisible()
    assert viewer.fib_canvas.rect_overlay._linestyle == "solid"
    assert viewer.lamella_list_widget.isEnabled()
    assert viewer.milling_viewer_widget.get_config().name == manual_name
    # detached: a late stat from the strategy changes nothing
    chip_before = viewer.label_threshold_chip.text()
    strategy.intensity_stats_signal.emit(_stats(500.0, 1800.0, 1080.0, drop=True))
    qapp.processEvents()
    assert viewer.label_threshold_chip.text() == chip_before


def test_closing_the_window_detaches_without_stopping_the_run(viewer, qapp):
    stops = []
    viewer.enter_monitor_mode(_running_config(), on_stop=lambda: stops.append(1))
    viewer.close()
    qapp.processEvents()
    assert not viewer.in_monitor_mode
    assert stops == []


def test_a_manual_run_in_progress_refuses_to_be_hijacked(viewer, qapp):
    viewer._is_milling_active = True  # a manual mill is running here
    with pytest.raises(RuntimeError, match="manual mill"):
        viewer.enter_monitor_mode(_running_config())
    viewer._is_milling_active = False


# ── run mode: a supervised mill, checked, started and watched here ─────


def _fib():
    from fibsem.structures import FibsemImage

    return FibsemImage.generate_blank_image(resolution=(768, 512), hfw=80e-6)


def test_run_mode_locks_the_site_and_shows_the_mill(viewer, qapp):
    config = _running_config(bbox=FibsemRectangle(0.3, 0.3, 0.2, 0.2), drop=0.35)
    manual_name = viewer.milling_viewer_widget.get_config().name
    lamella = viewer.experiment.positions[0]

    viewer.enter_run_mode(
        lamella=lamella, milling_config=config, fib_image=_fib(), title="Coincident"
    )
    qapp.processEvents()

    assert viewer.in_run_mode and not viewer.in_monitor_mode
    assert viewer._selected_lamella is lamella
    assert not viewer.lamella_list_widget.isEnabled()
    # the viewer's own Start Milling, and Continue; no Save, no Skip Site
    assert viewer.btn_milling.isVisible()
    assert viewer.btn_setup_skip.text() == "Continue"
    assert viewer.btn_setup_skip.isVisible()
    assert not viewer.btn_setup_continue.isVisible()
    assert viewer.label_task_lock.text() == "Locked to test by the task"
    assert "not yet milled" in viewer.label_selected_lamella.text()
    assert viewer.spin_drop_threshold.value() == 35
    H, W = viewer.fm_canvas._img_shape
    assert viewer.fm_canvas.rect_overlay.get_rect()["x0"] == pytest.approx(
        0.3 * W, abs=1
    )
    # the mill's own strategies are untouched: the viewer runs a copy
    assert viewer._active_strategies
    assert viewer._active_strategies[0] is not config.enabled_stages[0].strategy
    # nothing ran: Continue answers None
    assert viewer.read_run_result() is None

    viewer.exit_run_mode()
    qapp.processEvents()
    assert not viewer.in_run_mode
    assert viewer.btn_setup_skip.text() == "Skip Site"
    assert not viewer.btn_setup_skip.isVisible()
    assert viewer.lamella_list_widget.isEnabled()
    assert viewer.milling_viewer_widget.get_config().name == manual_name


def test_start_milling_runs_the_edited_boxes_and_continue_answers_them(
    viewer, qapp, monkeypatch
):
    """Start Milling is the manual run, minus the pre-flight dialog; what it
    ran -- the moved boxes, the drop fraction -- is what Continue answers."""
    config = _running_config(bbox=FibsemRectangle(0.3, 0.3, 0.2, 0.2), drop=0.35)
    lamella = viewer.experiment.positions[0]
    answers = []
    viewer.enter_run_mode(
        lamella=lamella,
        milling_config=config,
        fib_image=_fib(),
        on_continue=lambda: answers.append("continue"),
    )
    qapp.processEvents()

    milling_widget = viewer.milling_viewer_widget.milling_widget
    ran = []

    actors = []

    def fake_run(config=None):
        from fibsem.acting import current_actor

        milling_widget._running_config = config
        ran.append(config)
        actors.append(current_actor())

    monkeypatch.setattr(milling_widget, "run_milling", fake_run)

    H, W = viewer.fm_canvas._img_shape
    viewer.fm_canvas.rect_overlay.set_rect(0.1 * W, 0.2 * H, 0.3 * W, 0.4 * H)
    viewer.milling_viewer_widget._move_patterns(Point(3.0e-6, 1.5e-6), move_all=True)
    viewer.spin_drop_threshold.setValue(50)
    qapp.processEvents()

    viewer.btn_milling.click()  # Start Milling
    qapp.processEvents()
    assert len(ran) == 1
    assert viewer._is_milling_active
    # started on the task's behalf, as the workflow's milling session is
    assert actors == ["task"]
    for stage in ran[0].enabled_stages:
        assert stage.pattern.point.x == pytest.approx(3.0e-6)
        assert stage.strategy.config.bbox.left == pytest.approx(0.1, abs=0.01)
        assert stage.strategy.config.intensity_drop_fraction == pytest.approx(0.5)

    # the (stubbed) mill finishes: still run mode, another run on offer
    ran[0].enabled_stages[0].strategy.end_reason = "drop"
    viewer._finalize_milling_ui()
    qapp.processEvents()
    assert viewer.in_run_mode and not viewer._is_milling_active
    assert viewer.btn_milling.isVisible() and viewer.btn_setup_skip.isVisible()
    assert "milled" in viewer.label_selected_lamella.text()
    assert not viewer.lamella_list_widget.isEnabled()

    result = viewer.read_run_result()
    assert result is not None
    assert result.enabled_stages[0].strategy.end_reason == "drop"
    assert result.enabled_stages[0].pattern.point.x == pytest.approx(3.0e-6)

    viewer.btn_setup_skip.click()  # Continue
    assert answers == ["continue"]


def test_exit_run_mode_stops_a_mill_still_running(viewer, qapp, monkeypatch):
    config = _running_config()
    viewer.enter_run_mode(
        lamella=viewer.experiment.positions[0], milling_config=config, fib_image=_fib()
    )
    milling_widget = viewer.milling_viewer_widget.milling_widget
    stops = []
    monkeypatch.setattr(milling_widget, "run_milling", lambda config=None: None)
    monkeypatch.setattr(milling_widget, "stop_milling", lambda: stops.append(1))
    viewer.btn_milling.click()
    assert viewer._is_milling_active

    viewer.exit_run_mode()
    assert stops == [1]
    assert not viewer.in_run_mode
    viewer._reset_run_chrome()  # the stubbed mill never finishes on its own
