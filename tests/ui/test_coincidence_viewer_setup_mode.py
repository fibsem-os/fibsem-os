"""The coincidence viewer's setup mode (FIB-911).

A Setup Coincidence Milling task hands the viewer one site: the boxes are
pre-drawn from the site's record, the operator adjusts them, and Save and
Continue reads them back. The manual path must come back untouched on exit.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

import yaml
from PyQt5.QtWidgets import QApplication

from fibsem.structures import FibsemImage, FibsemRectangle, Point


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def viewer(qapp, tmp_path, monkeypatch):
    """The whole viewer on the simulated Arctis (scene off), config path redirected
    so this machine's saved coincidence config never leaks in or out."""
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
    assert microscope.fm is not None
    experiment = Experiment(path=str(tmp_path), name="setup-mode-test")
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    experiment.positions.append(lamella)
    widget = FluorescenceCoincidenceViewerWidget(
        microscope=microscope, experiment=experiment
    )
    widget._lamella = lamella
    yield widget
    widget.close()
    microscope.disconnect()


def _site_config(**kwargs):
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTaskConfig,
    )

    return SetupCoincidenceMillingTaskConfig(
        task_name="Setup Coincidence Milling", **kwargs
    )


def _milling_config(offset: Point):
    from copy import deepcopy

    from fibsem.applications.autolamella.workflows._default_milling_config import (
        DEFAULT_MILLING_CONFIG,
    )
    from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
        MILL_COINCIDENT_KEY,
    )

    config = deepcopy(DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY])
    for stage in config.enabled_stages:
        stage.pattern.point = deepcopy(offset)
    return config


def _frames(viewer):
    fib = FibsemImage.generate_blank_image(
        resolution=(768, 512), hfw=80e-6, random=True
    )
    fm = viewer.microscope.fm.acquire_image()
    return fib, fm


def test_enter_shows_the_stored_boxes_and_locks_the_site(viewer, qapp):
    lamella = viewer._lamella
    config = _site_config(
        fm_roi=FibsemRectangle(0.25, 0.25, 0.5, 0.5),
        pattern_offset=Point(2.0e-6, -1.0e-6),
        intensity_drop_fraction=0.3,
    )
    fib, fm = _frames(viewer)

    viewer.enter_setup_mode(
        lamella=lamella,
        config=config,
        milling_config=_milling_config(config.pattern_offset),
        fib_image=fib,
        fm_image=fm,
    )
    qapp.processEvents()

    assert viewer.in_setup_mode
    assert viewer._selected_lamella is lamella
    assert not viewer.lamella_list_widget.isEnabled()
    assert not viewer.selected_lamella_widget.isEnabled()
    # the setup controls replace Start Milling, and the drop % is editable pre-run
    assert not viewer.btn_milling.isVisible()
    assert viewer.btn_setup_continue.isVisible()
    assert viewer.btn_setup_skip.isVisible()
    assert viewer.spin_drop_threshold.isVisible()
    assert viewer.spin_drop_threshold.value() == 30
    # the stored FM region is on the canvas, in pixels of the frame
    H, W = viewer.fm_canvas._img_shape
    rect = viewer.fm_canvas.rect_overlay.get_rect()
    assert rect["x0"] == pytest.approx(0.25 * W, abs=1)
    assert rect["width"] == pytest.approx(0.5 * W, abs=1)
    # the milling box sits at the stored offset on every stage
    for stage in viewer.milling_viewer_widget.get_config().enabled_stages:
        assert stage.pattern.point.x == pytest.approx(2.0e-6)
        assert stage.pattern.point.y == pytest.approx(-1.0e-6)
    # the task-mode chrome: who owns the viewer, where the objective started,
    # what the setup reads as
    assert viewer.label_task_lock.isVisible()
    assert "Locked to test by the task" == viewer.label_task_lock.text()
    assert viewer.label_objective_hint.isVisible()
    assert "No objective height known" in viewer.label_objective_hint.text()
    # the objective controls are open, not collapsed as for manual use
    assert viewer.objective_panel._btn_collapse.isChecked()
    assert viewer._info_widget._setup_container.isVisible()
    assert "30 % drop" in viewer._info_widget._setup_label.text()
    assert "as stored" in viewer._info_widget._setup_label.text()


def test_read_result_reports_what_the_operator_left(viewer, qapp):
    lamella = viewer._lamella
    config = _site_config(pattern_offset=Point(0.0, 0.0))
    fib, fm = _frames(viewer)
    viewer.enter_setup_mode(
        lamella=lamella,
        config=config,
        milling_config=_milling_config(config.pattern_offset),
        fib_image=fib,
        fm_image=fm,
    )
    qapp.processEvents()

    # the operator drags the FM region and the milling box, sets the drop %
    H, W = viewer.fm_canvas._img_shape
    viewer.fm_canvas.rect_overlay.set_rect(0.1 * W, 0.2 * H, 0.3 * W, 0.4 * H)
    viewer.milling_viewer_widget._move_patterns(Point(3.0e-6, 1.5e-6), move_all=True)
    viewer.spin_drop_threshold.setValue(55)
    viewer.chk_copy_setup.setChecked(True)
    qapp.processEvents()

    # anything touched reads as unsaved until Save and Continue
    assert "unsaved changes" in viewer._info_widget._setup_label.text()
    assert "55 % drop" in viewer._info_widget._setup_label.text()

    result = viewer.read_setup_result()

    assert result.fm_roi is not None
    assert result.fm_roi.left == pytest.approx(0.1, abs=0.01)
    assert result.fm_roi.top == pytest.approx(0.2, abs=0.01)
    assert result.fm_roi.width == pytest.approx(0.3, abs=0.01)
    assert result.fm_roi.height == pytest.approx(0.4, abs=0.01)
    assert result.pattern_offset.x == pytest.approx(3.0e-6)
    assert result.pattern_offset.y == pytest.approx(1.5e-6)
    assert result.intensity_drop_fraction == pytest.approx(0.55)
    assert result.copy_to_unset is True
    # the objective height is wherever the objective is
    assert result.objective_position == pytest.approx(
        viewer.microscope.fm.objective.position
    )


def test_fib_drag_moves_every_stage_in_setup_mode(viewer, qapp):
    """A two-stage mill (top to bottom, then bottom to top) shares one position."""
    from copy import deepcopy

    lamella = viewer._lamella
    config = _site_config()
    milling = _milling_config(Point(0.0, 0.0))
    second = deepcopy(milling.stages[0])
    second.name = "Coincident Milling 02"
    second.pattern.scan_direction = "BottomToTop"
    milling.stages.append(second)
    fib, fm = _frames(viewer)
    viewer.enter_setup_mode(
        lamella=lamella,
        config=config,
        milling_config=milling,
        fib_image=fib,
        fm_image=fm,
    )
    qapp.processEvents()

    px = viewer.fib_canvas.canvas.pixel_size
    w, h = viewer.fib_canvas.canvas.img_width, viewer.fib_canvas.canvas.img_height
    # a drag that puts the box centre 4 µm right, 2 µm up of the image centre
    viewer._on_fib_rect_changed(
        {"cx": w / 2 + 4e-6 / px, "cy": h / 2 - 2e-6 / px, "width": 10, "height": 10}
    )
    qapp.processEvents()

    stages = viewer.milling_viewer_widget.get_config().enabled_stages
    assert len(stages) == 2
    for stage in stages:
        assert stage.pattern.point.x == pytest.approx(4e-6, rel=0.05)
        assert stage.pattern.point.y == pytest.approx(2e-6, rel=0.05)
    # both stages' real shapes are drawn on the FIB canvas, not just one box,
    # each with its scan direction: the first arrow down, the second up
    from matplotlib.text import Annotation

    overlay = viewer.fib_canvas.pattern_overlay
    assert len(overlay._stages) == 2
    assert overlay._image is not None
    assert len(overlay._artists) >= 2
    arrows = [a for a in overlay._artists if isinstance(a, Annotation)]
    assert len(arrows) == 2
    assert arrows[0].xy[1] > arrows[0].xyann[1]  # TopToBottom: y grows downward
    assert arrows[1].xy[1] < arrows[1].xyann[1]  # BottomToTop


def test_continue_and_skip_fire_their_callbacks_and_exit_restores(viewer, qapp):
    lamella = viewer._lamella
    fib, fm = _frames(viewer)
    manual_before = viewer.milling_viewer_widget.get_config().name
    fired = []

    viewer.enter_setup_mode(
        lamella=lamella,
        config=_site_config(),
        milling_config=_milling_config(Point(0.0, 0.0)),
        fib_image=fib,
        fm_image=fm,
        on_continue=lambda: fired.append("continue"),
        on_skip=lambda: fired.append("skip"),
    )
    viewer.btn_setup_continue.click()
    assert fired == ["continue"]
    # the answer's reader exits the mode; do it as the responder would
    viewer.exit_setup_mode()
    qapp.processEvents()

    assert not viewer.in_setup_mode
    assert viewer.btn_milling.isVisible()
    assert not viewer.btn_setup_continue.isVisible()
    assert not viewer.spin_drop_threshold.isVisible()
    assert not viewer.label_task_lock.isVisible()
    assert not viewer.label_objective_hint.isVisible()
    assert not viewer._info_widget._setup_container.isVisible()
    assert viewer.lamella_list_widget.isEnabled()
    assert viewer.milling_viewer_widget.get_config().name == manual_before

    viewer.enter_setup_mode(
        lamella=lamella,
        config=_site_config(),
        milling_config=_milling_config(Point(0.0, 0.0)),
        on_skip=lambda: fired.append("skip"),
    )
    viewer.btn_setup_skip.click()
    assert fired[-1] == "skip"


def test_closing_the_window_mid_setup_is_a_skip(viewer, qapp):
    fired = []
    viewer.enter_setup_mode(
        lamella=viewer._lamella,
        config=_site_config(),
        milling_config=_milling_config(Point(0.0, 0.0)),
        on_skip=lambda: (fired.append("skip"), viewer.exit_setup_mode()),
    )
    viewer.close()
    qapp.processEvents()
    assert fired == ["skip"]
    assert not viewer.in_setup_mode


def test_a_held_exit_keeps_the_lock_until_the_next_site_or_the_end(viewer, qapp):
    """Between one site's Save and the next site's hand-off the viewer stays
    locked, showing what was saved, and the manual state restored at the end
    is the operator's -- not the first site's task config."""
    from copy import deepcopy

    lamella = viewer._lamella
    manual_before = viewer.milling_viewer_widget.get_config().name
    fib, fm = _frames(viewer)
    site1 = _milling_config(Point(1.0e-6, 0.0))
    site1.name = "site-1 task config"

    viewer.enter_setup_mode(
        lamella=lamella,
        config=_site_config(),
        milling_config=site1,
        fib_image=fib,
        fm_image=fm,
    )
    viewer.exit_setup_mode(hold=True)
    qapp.processEvents()

    assert not viewer.in_setup_mode
    assert viewer.is_holding_setup
    assert not viewer.lamella_list_widget.isEnabled()
    assert not viewer.btn_milling.isVisible()
    assert not viewer.btn_setup_continue.isVisible()
    assert viewer.label_task_lock.isVisible()
    assert viewer.label_task_lock.text() == "Waiting for the next site"
    assert viewer.milling_viewer_widget.get_config().name == "site-1 task config"

    # the next site takes over without a manual interlude
    site2 = _milling_config(Point(-1.0e-6, 0.0))
    site2.name = "site-2 task config"
    viewer.enter_setup_mode(
        lamella=lamella,
        config=_site_config(),
        milling_config=site2,
        fib_image=fib,
        fm_image=fm,
    )
    assert viewer.in_setup_mode and not viewer.is_holding_setup
    assert viewer.btn_setup_continue.isVisible()
    assert viewer.milling_viewer_widget.get_config().name == "site-2 task config"

    # the end of the run restores the operator's config, not site 1's
    viewer.exit_setup_mode(hold=True)
    viewer.exit_setup_mode()
    qapp.processEvents()
    assert not viewer.is_holding_setup
    assert viewer.lamella_list_widget.isEnabled()
    assert viewer.btn_milling.isVisible()
    assert viewer.milling_viewer_widget.get_config().name == manual_before


def test_shift_scroll_on_the_fm_quadrant_steps_the_objective(viewer, qapp):
    """As on the main FM canvas: Shift+scroll nudges the objective by the step
    size, plain scroll is left to the canvas zoom."""
    objective = viewer.fm_objective_widget
    objective.update_objective_position_labels()
    before = objective.doubleSpinBox_objective_position.value()
    step = objective.doubleSpinBox_objective_step_size.value()

    viewer.fm_canvas.canvas.canvas_scrolled.emit(10.0, 10.0, 1, ("Shift",))
    qapp.processEvents()
    assert objective.doubleSpinBox_objective_position.value() == pytest.approx(
        before + step
    )

    viewer.fm_canvas.canvas.canvas_scrolled.emit(10.0, 10.0, 1, ())
    qapp.processEvents()
    assert objective.doubleSpinBox_objective_position.value() == pytest.approx(
        before + step
    )
