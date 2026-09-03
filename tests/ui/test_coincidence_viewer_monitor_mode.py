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


def _stats(value, peak, threshold, drop=False):
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

    # live stats reach the panel: a drop turns the chip orange
    strategy.intensity_stats_signal.emit(_stats(1000.0, 1800.0, 1080.0, drop=True))
    qapp.processEvents()
    assert "Intensity Drop" in viewer.label_threshold_chip.text()
    assert viewer.line_plot_widget is not None


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
