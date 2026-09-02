"""The grid protocol editor: the grid tasks in the experiment's protocol and their
settings, hosted on the Protocol tab under the Lamella | Grid selector."""

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.grid_protocol_widget import (
    AddGridTaskDialog,
    GridProtocolWidget,
)
from fibsem.applications.autolamella.ui.protocol_details_dialog import (
    ProtocolDetailsDialog,
    ProtocolHeaderWidget,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.fm.structures import ChannelSettings, ZParameters
from fibsem.structures import BeamType

BEAM = BeamOverviewGridTaskConfig.task_type
FM = FluorescenceOverviewGridTaskConfig.task_type


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    return exp


@pytest.fixture
def widget(qapp, experiment):
    w = GridProtocolWidget()
    w.set_experiment(experiment)
    return w


def saved_protocol(experiment) -> dict:
    return yaml.safe_load((Path(experiment.path) / "protocol.yaml").read_text())


def test_without_a_task_protocol_there_is_nothing_to_edit(qapp, tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    w = GridProtocolWidget()
    w.set_experiment(exp)
    assert "task protocol first" in w.editor_panel.hint.text()
    assert not w.task_list.btn_add.isVisibleTo(w)


def test_an_empty_protocol_invites_a_task(widget):
    assert widget.task_names == []
    assert "No grid tasks yet" in widget.editor_panel.hint.text()
    assert widget.task_list.btn_add.isVisibleTo(widget)
    assert not widget.editor_panel.btn_reset.isEnabled()


def test_adding_a_beam_task_lists_it_selects_it_and_saves(widget, experiment):
    config = widget.add_task(BEAM, "overview_sem")
    assert isinstance(config, BeamOverviewGridTaskConfig)
    assert widget.task_names == ["overview_sem"]
    assert widget.selected_task_name == "overview_sem"
    assert widget.editor_panel.title.text() == "overview_sem · Beam overview"
    assert saved_protocol(experiment)["grid_tasks"]["order"] == ["overview_sem"]


def test_the_form_edits_the_config_on_save(widget, experiment):
    widget.add_task(BEAM, "overview_fib")
    editor = widget.editor_panel.editor_for(BEAM)
    editor.orientation.setCurrentText("FIB")
    editor.filename.setText("survey")
    settings = editor.settings.get_settings()
    settings.nrows, settings.ncols = 4, 5
    settings.image_settings.beam_type = BeamType.ION
    editor.settings.update_from_settings(settings)

    config = widget.apply_selected()
    assert config.orientation == "FIB" and config.filename == "survey"
    assert (config.settings.nrows, config.settings.ncols) == (4, 5)
    assert config.beam_type is BeamType.ION and config.role == "overview_fib"
    saved = saved_protocol(experiment)["grid_tasks"]["tasks"]["overview_fib"]
    assert saved["orientation"] == "FIB" and saved["settings"]["nrows"] == 4


def test_an_edit_in_the_form_is_saved_as_it_is_made(widget, experiment):
    widget.add_task(BEAM, "overview_sem")
    editor = widget.editor_panel.editor_for(BEAM)
    editor.orientation.setCurrentText("FIB")  # no Save anywhere
    assert experiment.grid_protocol.task_config["overview_sem"].orientation == "FIB"
    assert (
        saved_protocol(experiment)["grid_tasks"]["tasks"]["overview_sem"]["orientation"]
        == "FIB"
    )


def test_filling_the_form_from_a_config_does_not_write_it_back(widget, experiment):
    """Loading must not count as an edit, or selecting a task would save it."""
    widget.add_task(BEAM, "overview_sem")
    saves = []
    widget.protocol_changed.connect(lambda: saves.append(True))
    widget.refresh()
    assert saves == []


def test_reset_puts_the_defaults_back(widget, experiment):
    widget.add_task(BEAM, "overview_sem")
    editor = widget.editor_panel.editor_for(BEAM)
    editor.orientation.setCurrentText("FIB")
    widget.apply_selected()
    fresh = widget.reset_selected()
    assert fresh.orientation == "SEM"
    assert experiment.grid_protocol.task_config["overview_sem"] is fresh
    assert editor.orientation.currentText() == "SEM"


def test_the_trash_icon_asks_first(widget, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    widget.add_task(BEAM, "overview_sem")
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: asked.append(True) or QMessageBox.No
    )
    widget.task_list.btn_remove.click()
    assert asked == [True] and widget.task_names == ["overview_sem"]


def test_remove_drops_the_task_and_saves(widget, experiment):
    widget.add_task(BEAM, "overview_sem")
    widget.add_task(BEAM, "overview_fib")
    widget.remove_task("overview_sem")
    assert widget.task_names == ["overview_fib"]
    assert saved_protocol(experiment)["grid_tasks"]["order"] == ["overview_fib"]


def test_the_add_dialog_suggests_a_unique_name(widget, experiment):
    widget.add_task(BEAM, "overview_sem")
    dialog = AddGridTaskDialog(experiment.grid_protocol)
    dialog.type_combo.setCurrentIndex(
        next(
            i
            for i in range(dialog.type_combo.count())
            if dialog.type_combo.itemData(i) == BEAM
        )
    )
    assert dialog.name_edit.text() == "overview_sem_2"
    dialog.type_combo.setCurrentIndex(
        next(
            i
            for i in range(dialog.type_combo.count())
            if dialog.type_combo.itemData(i) == FM
        )
    )
    assert dialog.name_edit.text() == "overview_fm"


def test_the_fluorescence_editor_round_trips_channels_and_z(widget, experiment):
    config = widget.add_task(FM, "overview_fm")
    config.channels = [
        ChannelSettings(name="GFP", color="green"),
        ChannelSettings(name="mCherry", color="red"),
    ]
    config.zparams = ZParameters(zmin=-3e-6, zmax=3e-6, zstep=1e-6)
    config.overview.rows, config.overview.cols = 2, 3
    widget.refresh()  # reload the form from the record
    editor = widget.editor_panel.editor_for(FM)
    assert [c.name for c in editor.channels.channel_settings] == ["GFP", "mCherry"]
    assert editor.settings.parameters.rows == 2
    assert editor.settings.z_widget.z_parameters.zstep == pytest.approx(1e-6)

    saved = widget.apply_selected()
    assert [c.name for c in saved.channels] == ["GFP", "mCherry"]
    assert (saved.overview.rows, saved.overview.cols) == (2, 3)
    assert saved.zparams.zmax == pytest.approx(3e-6)
    on_disk = saved_protocol(experiment)["grid_tasks"]["tasks"]["overview_fm"]
    assert [c["name"] for c in on_disk["channels"]] == ["GFP", "mCherry"]


def test_the_fluorescence_editor_takes_the_fm_when_one_connects(widget):
    widget.add_task(FM, "overview_fm")
    arctis, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    widget.set_microscope(arctis)
    assert widget.editor_panel.editor_for(FM).channels.fm is arctis.fm


def test_the_header_line_and_the_details_dialog(qapp, experiment):
    protocol = experiment.task_protocol
    protocol.name, protocol.version, protocol.description = "Screening", "1.2", "Arctis"
    header = ProtocolHeaderWidget()
    header.update_from_protocol(protocol)
    assert header.label.text() == "Screening" and header.version_label.text() == "v1.2"
    assert header.toolTip() == "Arctis"
    dialog = ProtocolDetailsDialog(protocol)
    dialog.edits["name"].setText("Screening 2 ")
    assert dialog.values() == {
        "name": "Screening 2",
        "description": "Arctis",
        "version": "1.2",
    }


@pytest.fixture
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    original = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


class _NoMinimap:
    def set_experiment(self) -> None:
        pass


def test_the_protocol_tab_hosts_it_under_the_selector(main_ui, tmp_path):
    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol.load(cfg.AUTOLAMELLA_TASK_PROTOCOL_PATH)
    ui.experiment = exp
    main_ui.minimap_widget = _NoMinimap()
    main_ui._on_experiment_update()

    editor = main_ui.task_widget
    tabs = editor.protocol_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Lamella", "Grid"]
    assert editor.protocol_header.label.text() == exp.task_protocol.name
    # the Grid page follows the flag, like the Grids tab
    was = main_ui._preferences.features.grid_workflow
    try:
        main_ui._preferences.features.grid_workflow = False
        main_ui._apply_grid_workflow_visibility()
        assert not tabs.isTabVisible(1) and tabs.tabBar().isHidden()
        main_ui._preferences.features.grid_workflow = True
        main_ui._apply_grid_workflow_visibility()
        assert tabs.isTabVisible(1) and not tabs.tabBar().isHidden()

        tabs.setCurrentIndex(1)
        assert editor.grid_protocol_active
        assert editor.grid_protocol.editor_panel.isVisibleTo(editor)
        assert not editor.task_parameters_config_widget.isVisibleTo(editor)
        assert not editor.milling_task_editor.isVisibleTo(editor)
        config = editor.grid_protocol.add_task(BEAM, "overview_sem")
        assert saved_protocol(exp)["grid_tasks"]["order"] == ["overview_sem"]
        assert editor.grid_protocol.editor_panel.title.text() == (
            "overview_sem · Beam overview"
        )
        assert isinstance(config, BeamOverviewGridTaskConfig)

        tabs.setCurrentIndex(0)
        assert not editor.grid_protocol.editor_panel.isVisibleTo(editor)
        assert editor.task_parameters_config_widget.isVisibleTo(editor)
        # the Grids tab no longer carries a protocol view
        assert not hasattr(main_ui.grids_tab, "protocol_widget")
    finally:
        main_ui._preferences.features.grid_workflow = was
