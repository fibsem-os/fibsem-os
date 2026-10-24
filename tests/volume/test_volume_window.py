"""The volume window on Demo: positions added and posed from the Positions
panel, their stored fields edited on the Positions tab, and the protocol from
the Protocol tab, each by the experiment's or
``workflow_edits``' own call, as the App Server makes it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.workflows.attention import Attention
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.status import TaskState, TaskStatus
from fibsem.workflows.task_config import TaskPlan
from tests.fixtures.toasts import toasts_fixture

pytest.importorskip("PyQt5")


toasts = toasts_fixture()


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    import fibsem.config as cfg
    from fibsem.applications.volume.ui.window import VolumeWindow

    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(tmp_path / "prefs.yaml"))
    # closeEvent ends in app.quit(), which latches on the shared test QApplication
    monkeypatch.setattr(qapp, "quit", lambda: None)
    window = VolumeWindow()
    window.system_widget.connect_to_microscope()
    window.session.adopt(VolumeExperiment.create(tmp_path, "vol"))
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_add_position_here_takes_the_microscope_state(window):
    panel = window.panel.experiment_tab
    assert window.status_bar.text.startswith("Add a position")
    assert not window.workflow_controls.run_btn.isEnabled()

    panel.add_button.click()
    panel.add_button.click()

    experiment = window.experiment
    first, second = (p.name for p in experiment.positions)
    assert first.startswith("01-") and second.startswith("02-")
    pose = experiment.positions[0].poses[MILLING_POSE]
    assert pose.stage_position == window.microscope.get_stage_position()
    assert panel.selected.name == second
    assert row_text(panel, 0)[:2] == [first, "0 of 10"]
    assert window.status_bar.text.startswith("Ready to run")
    assert window.workflow_controls.run_btn.isEnabled()


def row_text(panel, row: int) -> list:
    from PyQt5.QtWidgets import QLabel

    widget = panel.list.itemWidget(panel.list.item(row))
    return [label.text() for label in widget.findChildren(QLabel)]


def row_button(panel, row: int, text: str):
    from PyQt5.QtWidgets import QPushButton

    widget = panel.list.itemWidget(panel.list.item(row))
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


def test_a_rows_update_sets_its_milling_pose_from_current(window, toasts):
    panel = window.panel.experiment_tab
    panel.add_button.click()
    panel.add_button.click()
    microscope = window.microscope
    stage = microscope.get_stage_position()
    stage.x += 20e-6
    microscope.move_stage_absolute(stage)

    row_button(panel, 0, "Update").click()

    first, second = window.experiment.positions
    assert first.poses[MILLING_POSE].stage_position.x == pytest.approx(stage.x)
    assert second.poses[MILLING_POSE].stage_position.x != pytest.approx(stage.x)
    assert toasts == []


def test_a_position_that_has_cut_keeps_its_milling_pose(window, toasts):
    panel = window.panel.experiment_tab
    panel.add_button.click()
    position = window.experiment.positions[0]
    position.task_history.append(
        TaskState(name="Volume FIB Milling", status=TaskStatus.Completed, iteration=0)
    )

    panel.set_pose_buttons[MILLING_POSE].click()

    assert [m for m, _ in toasts] == [
        f"{position.name} has cut slices at its milling pose."
    ]
    panel.set_pose_buttons["SEM_IMAGING"].click()
    assert "SEM_IMAGING" in position.poses


def test_the_panel_edits_nothing_stored_on_the_image(window):
    panel = window.panel.experiment_tab
    for name in ("slices_spin", "poi_button", "area_button"):
        assert not hasattr(panel, name)


@pytest.fixture
def position_view(window, qapp):
    """The Positions tab's Position view on a position with a saved FIB image
    of its face, taken at its milling pose."""
    from fibsem import acquire
    from fibsem.structures import BeamType, ImageSettings

    from .conftest import set_up_position

    experiment = window.experiment
    position = set_up_position(experiment, window.microscope, "aspen")
    folder = experiment.position_path(position)
    settings = ImageSettings(beam_type=BeamType.ION, hfw=50e-6, resolution=(256, 170))
    image = acquire.acquire_image(window.microscope, settings)
    image.save(str(folder / "ref_setup_face_ib.tif"))
    tab = window.positions_tab
    tab.refresh()
    qapp.processEvents()
    return tab.editor


def test_the_position_view_shows_a_saved_fib_image_not_the_alignment_one(
    position_view,
):
    editor = position_view
    combo = editor.image_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["ref_setup_face"]
    assert editor.image is not None
    assert editor.hint.text().startswith("Right-click")


def test_the_position_view_edits_through_the_experiment(window, position_view, toasts):
    from fibsem.structures import BeamType, FibsemRectangle, Point

    editor = position_view
    experiment = editor.experiment

    editor.place_poi(Point(1e-6, -2e-6))
    assert experiment.positions[0].poi == Point(1e-6, -2e-6)
    assert editor.poi_label.text() == "x 1.00 µm · y -2.00 µm"

    editor.area_check.setChecked(True)
    area = FibsemRectangle(0.3, 0.3, 0.2, 0.2)
    editor.view_controller.overlay_edited.emit(BeamType.ION, "position_alignment", area)
    assert experiment.positions[0].alignment_area == area

    editor.slices_spin.setValue(40)
    editor.slices_spin.editingFinished.emit()
    assert experiment.positions[0].iterations == 40

    editor.description_edit.setText("near the crack")
    editor.description_edit.editingFinished.emit()
    assert experiment.positions[0].description == "near the crack"

    assert editor.name_label.text() == "aspen"
    assert toasts == []


def test_the_position_view_draws_the_milling_where_it_mills(position_view):
    from fibsem.applications.volume.position_milling import position_millings
    from fibsem.applications.volume.tasks.milling import slice_at

    editor = position_view
    position = editor.position
    position.iterations = 100
    editor.show_position(position.name)
    combo = editor.milling_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Volume milling",
        "Volume FIB Milling",
        "Setup · Clearing trench",
        "Setup · Redeposition lanes",
        "Setup · Start slice",
        "Setup · Fiducial",
    ]
    config = editor.experiment.task_protocol.task_config["Volume FIB Milling"]

    def millings():
        return {m.label: m for m in position_millings(editor.experiment, position)}

    # nothing cut: slice 1 next against the start slice, the rest to cut past it
    progress = millings()["Volume FIB Milling"]
    stages = progress.milling.stages
    assert [s.name for s in stages] == ["Next · slice 1", "To cut · slices 2 to 100"]
    assert progress.styles == ("outline", "dashed")
    assert (
        stages[0].pattern.point == slice_at(config, position, 0).stages[0].pattern.point
    )
    band = stages[1].pattern
    first = slice_at(config, position, 1).stages[0].pattern
    last = slice_at(config, position, 99).stages[0].pattern
    assert band.point.y + band.height / 2 == pytest.approx(
        first.point.y + first.height / 2
    )
    assert band.point.y - band.height / 2 == pytest.approx(
        last.point.y - last.height / 2
    )
    start = millings()["Setup · Start slice"].milling
    assert start.stages[0].pattern.point == position.poi
    # the whole volume draws the slices in one colour, each setup piece in its own
    whole = millings()["Volume milling"]
    assert len(whole.milling.stages) == 2 + 1 + 2 + 1 + 1
    assert whole.colours[0] == whole.colours[1] != whole.colours[2]
    assert editor.depth_label.text() == "0.0 of 20.0 µm (100 × 0.200 µm)"

    # two slices cut: those filled, slice 3 next, two thicknesses on
    for iteration in (0, 1):
        position.task_history.append(
            TaskState(
                name="Volume FIB Milling",
                status=TaskStatus.Completed,
                iteration=iteration,
            )
        )
    editor.show_position(position.name)
    assert combo.currentText() == "Volume milling"
    progress = millings()["Volume FIB Milling"]
    stages = progress.milling.stages
    assert [s.name for s in stages] == [
        "Cut · slices 1 to 2",
        "Next · slice 3",
        "To cut · slices 4 to 100",
    ]
    assert progress.styles == ("filled", "outline", "dashed")
    assert stages[1].pattern.point.y == pytest.approx(
        slice_at(config, position, 0).stages[0].pattern.point.y - 2 * 200e-9
    )
    assert editor.depth_label.text().startswith("0.4 of ")


def test_the_position_view_says_the_experiments_refusals(position_view, toasts):
    from fibsem.structures import Point

    editor = position_view
    position = editor.position
    position.task_history.append(
        TaskState(name="Volume FIB Milling", status=TaskStatus.Completed, iteration=0)
    )

    editor.place_poi(Point(1e-6, 0))

    assert [m for m, _ in toasts] == [
        "aspen has cut slices from its point of interest."
    ]
    assert position.poi == Point(0, 1e-6)
    assert editor.poi_label.text() == "x 0.00 µm · y 1.00 µm"


def test_the_protocol_tab_edits_through_workflow_edits(window, qapp, toasts):
    tab = window.protocol_tab
    protocol = window.experiment.task_protocol
    names = list(protocol.workflow_config.workflow)
    assert tab.selected_task_names() == names
    recorded = []
    window.session.record_event = lambda kind, payload: recorded.append(payload)

    tab._on_attention(TaskPlan(name=names[-1]))
    assert protocol.get_attention(names[-1]) is Attention.supervised

    order = [names[1], names[0], *names[2:]]
    tab._on_reorder([TaskPlan(name=n) for n in order])
    qapp.processEvents()
    assert protocol.workflow_config.workflow == order
    assert [t.name for t in tab.tasks.get_tasks()] == order
    names = order

    tab._on_remove(TaskPlan(name=names[-1]))
    qapp.processEvents()
    assert names[-1] not in protocol.task_config
    assert [t.name for t in tab.tasks.get_tasks()] == names[:-1]
    assert [p["field"] for p in recorded] == ["attention", "order", "tasks"]
    assert toasts == []


def test_the_task_clicked_shows_its_settings_and_writes_them(window, qapp, monkeypatch):
    import copy

    tab = window.protocol_tab
    config = window.experiment.task_protocol.task_config["Volume FIB Milling"]
    recorded = []
    monkeypatch.setattr(
        window.microscope,
        "record_event",
        lambda kind, payload: recorded.append(payload),
    )

    tab.tasks.set_current("Volume FIB Milling")
    editor = tab.task_editor
    assert editor.task_name == "Volume FIB Milling"
    # the millings have their own editor, not a row of the form
    assert [row.field for row in editor.parameters._rows] == [
        "mill_trench",
        "lane_left",
        "lane_right",
        "slice_thickness",
        "flipped",
    ]
    assert editor.reference_imaging.isHidden()
    combo = editor._milling_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Clearing trench",
        "Redeposition lane",
        "Start slice",
        "Slice",
    ]

    editor.set_values({"slice_thickness": 300e-9})
    combo.setCurrentIndex(3)
    sent = copy.deepcopy(config.milling)
    sent.field_of_view = 80e-6
    editor.milling.settings_changed.emit(sent)
    assert config.slice_thickness == 300e-9
    assert config.milling.field_of_view == 80e-6
    assert config.trench.field_of_view != 80e-6
    assert config.milling is not sent

    tab.flush_pending_save()
    assert sorted(p["target"] for p in recorded) == [
        "protocol.milling",
        "protocol.slice_thickness",
    ]


def test_volume_milling_shows_each_piece_in_one_form_and_writes_it(window, qapp):
    from fibsem.applications.volume.ui.volume_milling import preview_milling

    tab = window.protocol_tab
    config = window.experiment.task_protocol.task_config["Volume FIB Milling"]
    tab.tasks.set_current("Volume FIB Milling")
    form, editor = tab.volume_form, tab.task_editor
    assert not form.isHidden() and not tab.volume_preview.isHidden()
    assert editor.isHidden() and editor.milling.isHidden()
    assert form.warning.isHidden()  # 200 nm cut, 200 nm slices

    form._controls["slice_thickness"].write(250e-9)
    assert config.slice_thickness == pytest.approx(250e-9)
    assert "0.050 µm is left between slices" in form.warning.text()
    assert not form.warning.isHidden()

    form._controls["start_slice.width"].write(12e-6)
    assert config.start_slice.stages[0].pattern.width == pytest.approx(12e-6)
    # a current the microscope offers
    form._controls["trench.milling_current"].write(2e-9)
    assert config.trench.stages[0].milling.milling_current == pytest.approx(2e-9)
    form._controls["lane_left"].write(False)
    assert config.lane_left is False

    # the preview: every piece, around a point of interest at the centre
    names = [s.name for s in preview_milling(config).milling.stages]
    assert names == [
        "Next · slice 1",
        "To cut · slices 2 to 50",
        "Clearing trench",
        "Redeposition lane · right",
        "Start slice",
    ]

    form.all_settings_button.click()
    assert not editor.isHidden() and not editor.milling.isHidden()
    assert form.isHidden() and tab.volume_preview.isHidden()
    tab.volume_settings_button.click()
    assert not form.isHidden() and editor.isHidden()

    # another task shows the full editor, and coming back shows the form
    tab.tasks.set_current("Setup FIB Milling Position")
    assert form.isHidden() and not editor.isHidden()
    tab.tasks.set_current("Volume FIB Milling")
    assert not form.isHidden()


def test_the_settings_go_with_the_microscope(window, qapp):
    tab = window.protocol_tab
    assert tab.task_editor is not None
    window.system_widget.connect_to_microscope()  # connected: it disconnects
    qapp.processEvents()
    assert tab.task_editor is None


def test_the_positions_tab_scrubs_a_positions_slices(window, qapp):
    from fibsem.applications.volume.manager import VolumeTaskManager

    from .conftest import set_up_position

    experiment = window.experiment
    set_up_position(experiment, window.microscope, "aspen", iterations=3)
    set_up_position(experiment, window.microscope, "birch", iterations=5)
    milling = experiment.task_protocol.task_config["Volume FIB Milling"]
    milling.milling.acquisition.acquire_fib = True
    VolumeTaskManager(microscope=window.microscope, experiment=experiment).execute(
        ["Volume FIB Milling", "Volume SEM Acquisition"], ["aspen"]
    )

    tab = window.positions_tab
    tab.refresh()
    cards = [tab.list.itemWidget(tab.list.item(i)) for i in range(tab.list.count())]
    assert [c.name.text() for c in cards] == ["aspen", "birch"]
    assert [c.slices.text() for c in cards] == ["3 of 3 slices", "0 of 5 slices"]
    assert not cards[0].thumb.pixmap().isNull()
    assert cards[1].state.text() == "Not started"

    scrubber = tab.scrubber
    assert tab.selected == "aspen"
    # Follow: on the newest slice
    assert scrubber.index == 2
    thickness = milling.slice_thickness
    assert scrubber.readout.text().startswith(
        f"Slice 3 of 3 · z {2 * thickness * 1e6:.1f} µm"
    )
    scrubber.step(-1)
    assert scrubber.index == 1
    assert not scrubber.follow.isChecked()
    assert scrubber.readout.text().startswith("Slice 2 of 3")

    assert not scrubber.fib_button.isHidden()
    scrubber.fib_button.click()
    assert [s.filename for s in scrubber.slices][0].endswith("_ib.tif")
    assert scrubber.index == 1

    tab.views.setCurrentWidget(scrubber)
    tab.list.setCurrentRow(1)
    assert tab.title.text() == "birch"
    assert scrubber.slices == []
    assert scrubber.image.text() == "No slices yet"
