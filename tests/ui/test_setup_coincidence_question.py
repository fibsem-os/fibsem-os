"""The setup-coincidence question over the Responder seam (FIB-911).

A real AutoLamellaUI on the simulated Arctis. The Setup Coincidence Milling
task runs on a worker thread as the workflow would, hands the operator the
coincidence viewer in setup mode, and blocks until Save and Continue. The test
drives the viewer as the operator would and checks the record the task leaves.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest
import yaml

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from fibsem.structures import FibsemRectangle, Point

SETUP_NAME = "Setup Coincidence Milling"


@pytest.fixture
def ui(qapp, tmp_path, monkeypatch):
    """A real AutoLamellaUI connected to the simulated Arctis (scene off), with an
    experiment holding one lamella at a milling pose."""
    import fibsem.config as fconfig
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

    with open(os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    monkeypatch.setattr(
        fconfig, "COINCIDENCE_MILLING_CONFIG_PATH", str(tmp_path / "cmc.yaml")
    )

    widget = AutoLamellaUI(parent_ui=None)
    # the system widget resolves its configuration by name; point it at the copy
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: str(path),
    )
    widget.system_widget.connect_to_microscope()
    assert widget.microscope is not None and widget.microscope.fm is not None

    experiment = Experiment(path=str(tmp_path), name="setup-question-test")
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = widget.microscope.get_microscope_state()
    fm_pose = widget.microscope.get_microscope_state()
    fm_pose.objective_position = 2.45e-3
    lamella.fluorescence_pose = fm_pose
    experiment.positions.append(lamella)
    widget.experiment = experiment
    widget._lamella = lamella

    yield widget
    viewer = getattr(widget, "_coincidence_viewer_window", None)
    if viewer is not None:
        viewer.close()
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _run_task_on_worker_thread(ui, lamella):
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTask,
        SetupCoincidenceMillingTaskConfig,
    )

    config = SetupCoincidenceMillingTaskConfig(task_name=SETUP_NAME)
    lamella.task_config[SETUP_NAME] = config
    task = SetupCoincidenceMillingTask(
        microscope=ui.microscope, config=config, lamella=lamella, parent_ui=ui
    )
    outcome = {}

    def target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome, config


def _wait_for_setup_mode(ui, qapp, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        viewer = getattr(ui, "_coincidence_viewer_window", None)
        if viewer is not None and viewer.in_setup_mode:
            return viewer
        time.sleep(0.01)
    raise AssertionError("the viewer never entered setup mode")


def _join(thread, qapp, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not thread.is_alive(), "the task never finished"


def test_save_and_continue_records_what_the_operator_left(ui, qapp):
    lamella = ui._lamella
    thread, outcome, config = _run_task_on_worker_thread(ui, lamella)

    viewer = _wait_for_setup_mode(ui, qapp)
    # the task put the objective at the FM pose's height before handing off
    assert ui.microscope.fm.objective.position == pytest.approx(2.45e-3)
    assert viewer._selected_lamella is lamella
    assert ui.WAITING_FOR_USER_INTERACTION

    # the operator places the boxes and saves
    H, W = viewer.fm_canvas._img_shape
    viewer.fm_canvas.rect_overlay.set_rect(0.3 * W, 0.3 * H, 0.2 * W, 0.2 * H)
    viewer.milling_viewer_widget._move_patterns(Point(2.0e-6, 1.0e-6), move_all=True)
    viewer.spin_drop_threshold.setValue(35)
    qapp.processEvents()
    viewer.btn_setup_continue.click()

    _join(thread, qapp)
    assert "error" not in outcome, outcome.get("error")

    # the record
    assert config.is_set_up
    assert config.objective_position == pytest.approx(2.45e-3)
    assert config.fm_roi is not None
    assert config.fm_roi.left == pytest.approx(0.3, abs=0.01)
    assert config.fm_roi.width == pytest.approx(0.2, abs=0.01)
    assert config.pattern_offset.x == pytest.approx(2.0e-6)
    assert config.pattern_offset.y == pytest.approx(1.0e-6)
    assert config.intensity_drop_fraction == pytest.approx(0.35)
    # the viewer is released and the prompt is down
    assert not viewer.in_setup_mode
    assert viewer.btn_milling.isVisible()
    assert not ui.WAITING_FOR_USER_INTERACTION


def test_skip_site_records_nothing(ui, qapp):
    lamella = ui._lamella
    thread, outcome, config = _run_task_on_worker_thread(ui, lamella)

    viewer = _wait_for_setup_mode(ui, qapp)
    viewer.btn_setup_skip.click()

    _join(thread, qapp)
    assert "error" not in outcome, outcome.get("error")
    assert not config.is_set_up
    assert config.fm_roi is None
    assert not viewer.in_setup_mode


def test_copy_to_unset_seeds_the_other_sites(ui, qapp, tmp_path):
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTask,
        SetupCoincidenceMillingTaskConfig,
    )

    lamella = ui._lamella
    other = Lamella(path=tmp_path / "other", number=1, petname="other")
    other.path.mkdir(parents=True, exist_ok=True)
    other.task_config[SETUP_NAME] = SetupCoincidenceMillingTaskConfig(
        task_name=SETUP_NAME
    )
    ui.experiment.positions.append(other)

    config = SetupCoincidenceMillingTaskConfig(task_name=SETUP_NAME)
    lamella.task_config[SETUP_NAME] = config
    manager = TaskManager(
        microscope=ui.microscope, experiment=ui.experiment, parent_ui=ui
    )
    task = SetupCoincidenceMillingTask(
        microscope=ui.microscope,
        config=config,
        lamella=lamella,
        parent_ui=ui,
        task_manager=manager,
    )
    outcome = {}

    def target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    viewer = _wait_for_setup_mode(ui, qapp)
    H, W = viewer.fm_canvas._img_shape
    viewer.fm_canvas.rect_overlay.set_rect(0.4 * W, 0.4 * H, 0.2 * W, 0.2 * H)
    viewer.chk_copy_setup.setChecked(True)
    qapp.processEvents()
    viewer.btn_setup_continue.click()
    _join(thread, qapp)
    assert "error" not in outcome, outcome.get("error")

    seeded = other.task_config[SETUP_NAME]
    assert seeded.fm_roi == config.fm_roi
    assert seeded.fm_roi is not None and seeded.fm_roi.left == pytest.approx(
        0.4, abs=0.01
    )
    assert seeded.intensity_drop_fraction == config.intensity_drop_fraction
    # never the objective height: that is per site
    assert not seeded.is_set_up
