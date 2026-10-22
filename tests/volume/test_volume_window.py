"""The volume window on Demo: positions added and edited from the Positions
panel, and the protocol from the Protocol tab, each by the experiment's or
``workflow_edits``' own call, as the App Server makes it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.workflows.attention import Attention
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.status import TaskState, TaskStatus
from fibsem.workflows.task_config import TaskPlan

pytest.importorskip("PyQt5")


@pytest.fixture
def toasts(monkeypatch):
    from fibsem.ui import notification_service

    said = []
    monkeypatch.setattr(
        notification_service, "show_toast", lambda msg, *a, **k: said.append(msg)
    )
    return said


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
    names = [p.name for p in experiment.positions]
    assert [n[:3] for n in names] == ["01-", "02-"]
    pose = experiment.positions[0].poses[MILLING_POSE]
    assert pose.stage_position == window.microscope.get_stage_position()
    assert panel.selected.name == names[1]
    assert panel.list.item(0).text().endswith("0 of 10")
    assert window.status_bar.text.startswith("Ready to run")
    assert window.workflow_controls.run_btn.isEnabled()


def test_the_panel_edits_through_the_experiment(window, toasts):
    panel = window.panel.experiment_tab
    panel.add_button.click()
    panel.add_button.click()

    # names are fixed when a position is added
    assert panel.name_label.text() == window.experiment.positions[1].name

    panel.slices_spin.setValue(40)
    panel.slices_spin.editingFinished.emit()
    assert window.experiment.positions[1].iterations == 40
    assert panel.list.item(1).text().endswith("0 of 40")


def test_a_position_that_has_cut_keeps_its_milling_pose(window, toasts):
    panel = window.panel.experiment_tab
    panel.add_button.click()
    position = window.experiment.positions[0]
    position.task_history.append(
        TaskState(name="Volume FIB Milling", status=TaskStatus.Completed, iteration=0)
    )

    panel.set_pose_button.click()

    assert toasts == [f"{position.name} has cut slices at its milling pose."]
    panel.pose_combo.setCurrentText("SEM_IMAGING")
    panel.set_pose_button.click()
    assert "SEM_IMAGING" in position.poses


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


def test_the_point_picked_on_the_fib_image_is_the_positions(window, toasts):
    from fibsem.structures import Point

    panel = window.panel.experiment_tab
    panel.add_button.click()

    class Marker:
        shown_at = None

        def show(self, poi, fallback=None):
            self.shown_at = poi
            return True

        def read(self):
            return Point(1e-6, -2e-6)

        def remove(self):
            pass

    panel.poi_marker = marker = Marker()
    panel.poi_button.click()
    assert marker.shown_at == Point(0, 0)
    assert panel.poi_button.text() == "Done"
    panel.poi_button.click()

    assert window.experiment.positions[0].poi == Point(1e-6, -2e-6)
    assert panel.poi_label.text() == "x 1.00 µm · y -2.00 µm"
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
    # the milling has its own editor, not a row of the form
    assert [row.field for row in editor.parameters._rows] == ["slice_thickness"]
    assert editor.milling_choice.isHidden()
    assert editor.reference_imaging.isHidden()

    editor.set_values({"slice_thickness": 300e-9})
    sent = copy.deepcopy(config.milling)
    sent.field_of_view = 80e-6
    editor.milling.settings_changed.emit(sent)
    assert config.slice_thickness == 300e-9
    assert config.milling.field_of_view == 80e-6
    assert config.milling is not sent

    tab.flush_pending_save()
    assert sorted(p["target"] for p in recorded) == [
        "protocol.milling",
        "protocol.slice_thickness",
    ]


def test_a_task_with_several_millings_picks_the_one_shown(window):
    tab = window.protocol_tab
    config = window.experiment.task_protocol.task_config["Setup FIB Milling Position"]
    tab.tasks.set_current("Setup FIB Milling Position")
    editor = tab.task_editor
    combo = editor._milling_combo
    assert not editor.milling_choice.isHidden()
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Trench",
        "Fiducial",
        "Face",
    ]

    combo.setCurrentIndex(2)
    sent = editor.milling.get_config()
    sent.field_of_view = 90e-6
    editor.milling.settings_changed.emit(sent)
    assert config.face.field_of_view == 90e-6
    assert config.trench.field_of_view != 90e-6


def test_the_settings_go_with_the_microscope(window, qapp):
    tab = window.protocol_tab
    assert tab.task_editor is not None
    window.system_widget.connect_to_microscope()  # connected: it disconnects
    qapp.processEvents()
    assert tab.task_editor is None
