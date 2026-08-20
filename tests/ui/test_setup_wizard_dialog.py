"""The setup wizard dialog, and the first-run offer that launches it.

The arithmetic these tests do not repeat is in ``tests/test_setup_wizard.py``, which
runs on CI. What is here is the part only a real widget can answer: which step comes
next, what each control is prefilled with, and whether a choice made on one step is
still true after going back and changing another.

``isHidden()`` rather than ``isVisible()`` throughout. A widget whose parent has never
been shown reports ``isVisible() == False`` regardless of its own state, so a headless
assertion that something is not visible passes for the wrong reason and would keep
passing if the widget were shown.
"""

import os

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem import config as cfg
from fibsem import setup_wizard as wizard
from fibsem.ui.widgets.setup_wizard_dialog import (
    STEP_CONNECTION,
    STEP_FOLDERS,
    STEP_MICROSCOPE,
    STEP_REVIEW,
    STEP_STAGE,
    ChoiceCard,
    SetupWizardDialog,
)


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """A private configuration directory, so no test can write into the checkout.

    The shipped files are copied in rather than the directory merely redirected:
    ``cfg.CONFIG_PATH`` is both where model configurations are *read from* and where
    new ones are *written to*, so pointing it somewhere empty makes every model
    unreadable -- and the wizard's save then fails into a modal error box that a
    headless test cannot dismiss.
    """
    import shutil

    # Named files, not `*.yaml`. A glob also copies whatever `user-*.yaml` the
    # developer's own runs have left in the config directory, and those are exactly
    # the files these tests are about -- the fixture would then inherit a machine that
    # has already been set up, and every first-run assertion would fail on a working
    # tree and pass on a clean one.
    for model in wizard.MICROSCOPE_MODELS:
        shutil.copy(model.path, tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(tmp_path))

    configurations = {"default-configuration": {"path": cfg.MICROSCOPE_CONFIGURATION_PATH}}
    monkeypatch.setattr(cfg, "USER_CONFIGURATIONS", configurations)
    monkeypatch.setattr(
        cfg,
        "USER_CONFIGURATIONS_YAML",
        {"configurations": configurations, "default": "default-configuration"},
    )
    monkeypatch.setattr(
        cfg, "USER_CONFIGURATIONS_PATH", str(tmp_path / "user-configurations.yaml")
    )
    monkeypatch.setattr(
        cfg, "USER_PREFERENCES_PATH", str(tmp_path / "user-preferences.yaml")
    )
    # Self-checking, because the failure mode above is silent: leaked state does not
    # error, it just makes every first-run test assert against a configured machine.
    assert wizard.is_first_run(), "the isolated state is not a fresh install"
    return tmp_path


@pytest.fixture
def dialog(qapp, isolated_state):
    widget = SetupWizardDialog()
    yield widget
    # Detached first: the dialog closes whatever connection it believes it owns, and a
    # test that lent it a shared one should not have that closed underneath the fixture
    # that owns it.
    widget._microscope = None
    widget.reject()
    widget.deleteLater()


@pytest.fixture
def demo_microscope():
    """A real Demo session, for the steps that read from an instrument."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    yield microscope
    microscope.disconnect()


def summary_rows(widget) -> dict:
    """The review panel, as ``{caption: value}``."""
    rows = {}
    for index in range(widget._summary_layout.count()):
        container = widget._summary_layout.itemAt(index).widget()
        layout = container.layout()
        rows[layout.itemAt(0).widget().text()] = layout.itemAt(1).widget().text()
    return rows


# ---------------------------------------------------------------------------
# Which step comes next
# ---------------------------------------------------------------------------


def test_a_recognised_model_skips_the_stage_step(dialog):
    dialog._select_model("tfs-arctis")
    assert dialog._index == STEP_MICROSCOPE
    dialog._on_next()
    assert dialog._index == STEP_CONNECTION
    dialog._on_next()
    assert dialog._index == STEP_FOLDERS


def test_start_blank_stops_at_the_stage_step(dialog):
    dialog._select_model("other")
    dialog._on_next()
    dialog._on_next()
    assert dialog._index == STEP_STAGE


def test_only_the_compustage_skips_the_stage_step(dialog):
    """Every other model prefills the step and still asks."""
    dialog._select_model("tfs-arctis")
    assert dialog._is_skipped(STEP_STAGE)
    for key in ("tfs-hydra", "tfs-aquilos2", "tescan", "other"):
        dialog._select_model(key)
        assert not dialog._is_skipped(STEP_STAGE), key


def test_the_stage_step_is_prefilled_from_the_chosen_model(dialog):
    """Confirming a number is easier than recalling one, so the step opens on the
    shipped values rather than empty or on a generic 35."""
    dialog._select_model("tfs-hydra")
    dialog._show_step(STEP_STAGE)
    assert dialog._rotation_spin.value() == pytest.approx(0.0)
    assert dialog._pre_tilt_spin.value() == pytest.approx(35.0)

    dialog._select_model("tescan")
    assert dialog._rotation_spin.value() == pytest.approx(180.0)
    assert dialog._pre_tilt_spin.value() == pytest.approx(0.0)


def test_a_prefilled_value_is_still_written_as_an_answer(dialog):
    """Leaving the prefill alone is a confirmation, not a skip -- it has to be written.

    Otherwise "I checked, 35 is right" and "nobody looked" would produce the same file
    while meaning opposite things.
    """
    dialog._select_model("tfs-hydra")
    dialog._show_step(STEP_STAGE)
    dialog._read_current_step()
    assert dialog.choices.shuttle_pre_tilt == pytest.approx(35.0)
    assert dialog.choices.rotation_reference == pytest.approx(0.0)


def test_back_skips_the_same_step_forwards_and_backwards(dialog):
    """A skipped step reachable only by Back is a step nothing filled in."""
    dialog._select_model("tfs-arctis")
    dialog._show_step(STEP_FOLDERS)
    dialog._on_back()
    assert dialog._index == STEP_CONNECTION


def test_the_connection_step_is_never_skipped(dialog):
    """The address cannot be implied, and the stage step reads through it."""
    for key in [model.key for model in wizard.MICROSCOPE_MODELS]:
        dialog._select_model(key)
        assert not dialog._is_skipped(STEP_CONNECTION), key


def test_the_rail_says_a_skipped_step_was_answered(dialog):
    """Dropping it silently would hide that a value was chosen for you."""
    dialog._select_model("tfs-arctis")
    assert dialog._rail_rows[STEP_STAGE]._note.text() == "set by the configuration"
    dialog._select_model("other")
    assert dialog._rail_rows[STEP_STAGE]._note.text() == ""


def test_the_last_step_offers_to_save(dialog):
    dialog._show_step(STEP_REVIEW)
    assert "Save" in dialog.button_next.text()
    dialog._show_step(STEP_MICROSCOPE)
    assert dialog.button_next.text() == "Next"
    assert not dialog.button_back.isEnabled()


# ---------------------------------------------------------------------------
# Step 1: the microscope
# ---------------------------------------------------------------------------


def test_exactly_one_model_card_is_selected(dialog):
    dialog._select_model("tescan")
    selected = [key for key, card in dialog._model_cards.items() if card.is_selected()]
    assert selected == ["tescan"]


def test_the_manufacturer_is_only_asked_when_the_model_does_not_answer_it(dialog):
    dialog._select_model("tfs-arctis")
    assert dialog._manufacturer_row.isHidden()
    dialog._select_model("other")
    assert not dialog._manufacturer_row.isHidden()


def test_a_card_can_be_chosen_from_the_keyboard(dialog, qapp):
    """An instrument PC is often driven without a mouse worth using."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeyEvent

    card = dialog._model_cards["tescan"]
    assert isinstance(card, ChoiceCard)
    card.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Space, Qt.NoModifier))
    assert dialog.choices.model_key == "tescan"


# ---------------------------------------------------------------------------
# Step 2: the computer and the address
# ---------------------------------------------------------------------------


def test_the_support_pc_is_selected_and_prefilled(dialog):
    dialog._show_step(STEP_CONNECTION)
    assert dialog.choices.location_key == wizard.LOCATION_SUPPORT
    assert dialog._location_cards[wizard.LOCATION_SUPPORT].is_selected()
    assert dialog._address_edit.text() == cfg.DEFAULT_IP_ADDRESS


def test_the_address_follows_the_computer_choice(dialog):
    dialog._show_step(STEP_CONNECTION)
    dialog._select_location(wizard.LOCATION_MICROSCOPE)
    assert dialog._address_edit.text() == "localhost"
    dialog._select_location(wizard.LOCATION_SUPPORT)
    assert dialog._address_edit.text() == cfg.DEFAULT_IP_ADDRESS


def test_offline_has_no_address_to_give(dialog):
    dialog._show_step(STEP_CONNECTION)
    dialog._select_location(wizard.LOCATION_OFFLINE)
    assert dialog._address_edit.text() == ""
    assert not dialog._address_edit.isEnabled()
    assert not dialog.button_test.isEnabled()
    dialog._read_current_step()
    assert dialog.choices.address == ""


def test_an_existing_connection_is_borrowed_rather_than_duplicated(
    qapp, isolated_state, demo_microscope
):
    """Two AutoScript clients from one process is not a thing to discover on a first run.

    A real Demo session rather than a stand-in: the status card reads the connection's
    ``system.info`` to name what answered, and a stub with only the attributes this
    test happens to think of is exactly how that read went unnoticed the first time.
    """
    widget = SetupWizardDialog(microscope=demo_microscope)
    try:
        widget._show_step(STEP_CONNECTION)
        assert not widget.button_test.isEnabled()
        assert widget._active_microscope() is demo_microscope
        assert "Already connected" in widget._connection_status._title.text()
        # And it is not this dialog's to close -- the fixture still owns it after this.
        widget.reject()
        assert widget._microscope is None
    finally:
        widget.deleteLater()


def test_a_late_connection_does_not_touch_a_closed_dialog(dialog):
    """PyQt5 turns an exception in a slot into a process abort (FIB-329).

    A connection that lands after the dialog closed must close itself rather than
    paint into a deleted widget.
    """

    class FakeMicroscope:
        def __init__(self):
            self.disconnected = False

        def disconnect(self):
            self.disconnected = True

    dialog.reject()
    late = FakeMicroscope()
    dialog._on_connected(late)
    assert late.disconnected
    assert dialog._microscope is None


# ---------------------------------------------------------------------------
# Step 3: the stage
# ---------------------------------------------------------------------------


def test_the_stage_cannot_be_read_without_a_connection(dialog):
    dialog._select_model("other")
    dialog._show_step(STEP_STAGE)
    assert not dialog.button_read_stage.isEnabled()
    assert "Not connected" in dialog._stage_note.text()


def test_reading_the_stage_fills_in_the_reference_rotation(dialog, demo_microscope):
    """Read from where the operator put the stage, in degrees, and never moved there.

    A real Demo session rather than a stub: the conversion from the position's radians
    is the whole behaviour, and a stub would be free to return the answer already in
    the units the assertion wants.
    """
    import numpy as np

    from fibsem.structures import FibsemStagePosition

    demo_microscope.move_stage_absolute(
        FibsemStagePosition(r=np.radians(250.0), t=np.radians(17.0))
    )
    before = demo_microscope.get_stage_position()

    dialog._select_model("other")
    dialog._microscope = demo_microscope
    dialog._show_step(STEP_STAGE)
    dialog._on_read_stage()

    assert dialog._rotation_spin.value() == pytest.approx(250.0, abs=1e-3)
    assert "17.0" in dialog._stage_note.text()
    # The page promises the wizard reads and never moves; a wizard that drives the
    # stage while someone is leaning into the chamber is not acceptable.
    after = demo_microscope.get_stage_position()
    assert after.r == pytest.approx(before.r)
    assert after.t == pytest.approx(before.t)


def test_a_failed_stage_read_reports_rather_than_raising(dialog):
    class Broken:
        def get_stage_position(self):
            raise RuntimeError("no stage")

    dialog._select_model("other")
    dialog._microscope = Broken()
    dialog._show_step(STEP_STAGE)
    dialog._on_read_stage()
    assert "no stage" in dialog._stage_note.text()


def test_the_diagram_follows_the_number_that_was_entered(dialog):
    """A picture of 35 degrees beside a spin box reading 27 is worse than no picture."""
    dialog._select_model("other")
    dialog._show_step(STEP_STAGE)
    dialog._pre_tilt_spin.setValue(12.0)
    assert dialog._stage_diagram._pre_tilt == pytest.approx(12.0)
    dialog._pre_tilt_spin.setValue(0.0)
    assert dialog._stage_diagram._pre_tilt == pytest.approx(0.0)


def test_the_diagram_stays_flat_until_a_stage_is_read(dialog, demo_microscope):
    import numpy as np

    from fibsem.structures import FibsemStagePosition

    dialog._select_model("other")
    dialog._show_step(STEP_STAGE)
    assert dialog._stage_diagram._stage_tilt == pytest.approx(0.0)

    demo_microscope.move_stage_absolute(FibsemStagePosition(t=np.radians(17.0)))
    dialog._microscope = demo_microscope
    dialog._on_read_stage()
    assert dialog._stage_diagram._stage_tilt == pytest.approx(17.0, abs=1e-3)


def test_the_sample_faces_the_beam_it_is_drawn_facing(qapp):
    """At pre-tilt + stage tilt = the FIB's own angle, the beam hits the sample square on.

    This was wrong: the stage and shuttle rotated the opposite way, so the sample turned
    *away* from the FIB and the beam struck the back of the stage -- a picture that
    contradicted the numbers printed beside it.

    Analytic rather than pixel-based, so it pins the geometry helper the painter rotates
    by rather than a particular rendering.
    """
    import math

    from fibsem.ui.widgets.setup_wizard_dialog import StageDiagram

    diagram = StageDiagram(pre_tilt=35.0, stage_tilt=StageDiagram.FIB_ANGLE - 35.0)
    try:
        normal = diagram.sample_normal()
        fib = StageDiagram.beam_direction(StageDiagram.FIB_ANGLE)
        dot = normal[0] * fib[0] + normal[1] * fib[1]
        assert math.degrees(math.acos(max(-1.0, min(1.0, dot)))) == pytest.approx(0, abs=1e-6)

        # And a flat stage faces the SEM instead, which is the other end of the same rule.
        flat = StageDiagram(pre_tilt=0.0, stage_tilt=0.0)
        sem = StageDiagram.beam_direction(0.0)
        assert flat.sample_normal() == pytest.approx(sem, abs=1e-9)
    finally:
        diagram.deleteLater()


@pytest.mark.parametrize("pre_tilt, stage_tilt", [(0, 0), (35, 17), (-90, 90), (90, -90)])
def test_the_diagram_paints_at_the_extremes(qapp, pre_tilt, stage_tilt):
    """The spin boxes allow the whole range, and a paintEvent that raises is fatal."""
    from PyQt5.QtGui import QPixmap

    from fibsem.ui.widgets.setup_wizard_dialog import StageDiagram

    diagram = StageDiagram(pre_tilt, stage_tilt)
    diagram.resize(360, 212)
    try:
        pixmap = diagram.grab()  # forces a real paintEvent
        assert not pixmap.isNull()
        assert isinstance(pixmap, QPixmap)
    finally:
        diagram.deleteLater()


def test_stage_answers_do_not_survive_a_change_of_model(dialog):
    """Left in place they would be written over the shipped values.

    For a compustage, whose shipped pair is rotation reference 0 with ``rotation_180``
    also 0, a number typed for some other model is exactly the wrong thing to keep.
    """
    dialog._select_model("other")
    dialog._show_step(STEP_STAGE)
    dialog._rotation_spin.setValue(250.0)
    dialog._pre_tilt_spin.setValue(35.0)
    dialog._read_current_step()
    assert dialog.choices.rotation_reference == pytest.approx(250.0)

    dialog._select_model("tfs-arctis")
    assert dialog.choices.rotation_reference is None
    assert dialog.choices.shuttle_pre_tilt is None

    config = wizard.build_configuration(dialog.choices)
    assert config["stage"]["rotation_reference"] == 0
    assert config["stage"]["rotation_180"] == 0


# ---------------------------------------------------------------------------
# Step 5: review and save
# ---------------------------------------------------------------------------


def test_the_review_is_built_from_what_will_be_written(dialog):
    dialog._select_model("tfs-hydra")
    dialog._show_step(STEP_CONNECTION)
    dialog._select_location(wizard.LOCATION_MICROSCOPE)
    dialog._read_current_step()
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Hydra 1")
    dialog._update_review()

    rows = summary_rows(dialog)
    assert rows["Microscope"] == "ThermoFisher Hydra"
    assert rows["Computer"] == "Microscope PC"
    assert rows["Address"] == "localhost"
    assert rows["Manufacturer"] == "Thermo"


def test_the_review_says_which_values_nobody_typed(dialog):
    """A value chosen for you should be visible as such."""
    dialog._select_model("tfs-arctis")
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Bay 2")
    dialog._update_review()
    rows = summary_rows(dialog)
    assert "shipped configuration" in rows["Reference rotation"]
    assert "shipped configuration" in rows["Shuttle pre-tilt"]
    # Nothing was derived, so there is nothing to report as derived.
    assert "Rotated 180°" not in rows


def test_the_review_shows_the_derived_opposite_rotation(dialog):
    dialog._select_model("other")
    dialog._show_step(STEP_STAGE)
    dialog._rotation_spin.setValue(250.0)
    dialog._read_current_step()
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Bench")
    dialog._update_review()
    assert summary_rows(dialog)["Rotated 180°"].startswith("70.00")


def test_the_filename_hint_follows_the_name(dialog):
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Arctis Bay 2")
    assert "arctis-bay-2.yaml" in dialog._filename_hint.text()


def test_the_folders_step_asks_where_the_configuration_goes(dialog, tmp_path):
    """It used to ask only about experiments while silently writing the configuration
    somewhere it never named, which invited the two being answered as one."""
    dialog._show_step(STEP_FOLDERS)
    assert dialog._configuration_dir.text() == cfg.CONFIG_PATH

    elsewhere = str(tmp_path / "configurations")
    dialog._configuration_dir.setText(elsewhere)
    dialog._experiment_dir.setText(str(tmp_path / "experiments"))
    dialog._read_current_step()

    assert dialog.choices.configuration_directory == elsewhere
    assert dialog.choices.experiment_directory == str(tmp_path / "experiments")


def test_the_review_names_the_path_the_file_will_be_written_to(dialog, tmp_path):
    """A wizard whose whole output is one file should say where that file goes."""
    elsewhere = str(tmp_path / "configurations")
    dialog._show_step(STEP_FOLDERS)
    dialog._configuration_dir.setText(elsewhere)
    dialog._read_current_step()

    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Arctis Bay 2")
    assert os.path.join(elsewhere, "arctis-bay-2.yaml") in dialog._filename_hint.text()
    assert summary_rows(dialog)["Configuration folder"] == elsewhere


def test_saving_without_a_name_asks_for_one(dialog, monkeypatch):
    from PyQt5 import QtWidgets

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("")
    dialog._on_next()
    assert warnings
    assert dialog.result() == 0  # still open, nothing written


def test_saving_registers_the_configuration_and_reports_its_name(
    dialog, monkeypatch, isolated_state
):
    from PyQt5 import QtWidgets

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)

    saved = []
    dialog.configuration_saved.connect(saved.append)

    dialog._select_model("tfs-arctis")
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Arctis Bay 2")
    dialog._on_next()

    assert saved == ["Arctis Bay 2"]
    assert "Arctis Bay 2" in cfg.USER_CONFIGURATIONS
    assert os.path.exists(cfg.USER_CONFIGURATIONS["Arctis Bay 2"]["path"])
    assert cfg.USER_CONFIGURATIONS_YAML["default"] == "Arctis Bay 2"


def test_a_save_that_fails_leaves_the_dialog_open(dialog, monkeypatch):
    from PyQt5 import QtWidgets

    from fibsem.ui.widgets import setup_wizard_dialog as module

    critical = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *a, **k: critical.append(a)
    )

    def boom(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(module.wizard, "apply_setup", boom)
    dialog._show_step(STEP_REVIEW)
    dialog._name_edit.setText("Bay 2")
    dialog._on_next()
    assert critical
    assert dialog.result() == 0


# ---------------------------------------------------------------------------
# The first-run offer on the connection tab
# ---------------------------------------------------------------------------


def _enable_flag(enabled: bool = True) -> None:
    """Write the feature flag, as the preferences dialog would."""
    preferences = cfg.load_user_preferences()
    preferences.features.setup_wizard_enabled = enabled
    cfg.save_user_preferences(preferences)


@pytest.fixture
def connection_tab(qapp, isolated_state, monkeypatch):
    """The connection tab with the flag on, since that is what most of these test.

    Nothing is faked here any more. Enabling the flag writes the preferences file, and
    that is now simply not the first-run signal -- which is the whole point of the fix.
    The previous version of this fixture wrote the flag, deleted the file and
    monkeypatched the reader to keep both true at once; that it had to invent a state
    the disk cannot hold was the defect, showing up in the harness.
    """
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    _enable_flag(True)
    widget = FibsemSystemSetupWidget()
    yield widget
    widget.deleteLater()


def test_the_offer_appears_on_a_fresh_install(connection_tab):
    assert not connection_tab._frame_first_run.isHidden()


def test_turning_the_flag_on_does_not_suppress_the_offer(qapp, isolated_state, monkeypatch):
    """The defect this whole arrangement exists to prevent, at the widget.

    The flag lives in the preferences file. When that file's absence was the first-run
    signal, every route to enabling the flag -- the dialog, or writing the file by
    hand -- also ended the first run, so the callout could not be reached at all.
    """
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    _enable_flag(True)  # writes user-preferences.yaml, exactly as the dialog does
    assert os.path.exists(cfg.USER_PREFERENCES_PATH)

    widget = FibsemSystemSetupWidget()
    try:
        assert not widget._frame_first_run.isHidden()
    finally:
        widget.deleteLater()


def test_the_offer_is_behind_the_feature_flag(qapp, isolated_state, monkeypatch):
    """A fresh install with the flag off is offered nothing.

    The wizard writes a configuration file and two registry entries, and it is offered
    to someone with no way yet to judge whether it is trustworthy. Until it has bench
    time that offer is opt-in.
    """
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    assert wizard.is_first_run()  # the offer has something to appear for
    widget = FibsemSystemSetupWidget()
    try:
        assert widget._frame_first_run.isHidden()
        # And it appears the moment the flag is turned on, without a restart.
        preferences = cfg.load_user_preferences()
        preferences.features.setup_wizard_enabled = True
        widget.refresh_first_run_offer(preferences)
        assert not widget._frame_first_run.isHidden()
        preferences.features.setup_wizard_enabled = False
        widget.refresh_first_run_offer(preferences)
        assert widget._frame_first_run.isHidden()
    finally:
        widget.deleteLater()


def test_a_dismissed_offer_stays_dismissed_with_the_flag_on(
    qapp, isolated_state, monkeypatch
):
    """Dismissal is its own answer, not a side effect of anything else."""
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    _enable_flag(True)
    wizard.dismiss_first_run()

    widget = FibsemSystemSetupWidget()
    try:
        assert widget._frame_first_run.isHidden()
        assert wizard.is_first_run()  # still nothing configured
    finally:
        widget.deleteLater()


def test_dismissing_the_offer_makes_it_stay_dismissed(connection_tab, isolated_state):
    connection_tab._dismiss_first_run()
    assert connection_tab._frame_first_run.isHidden()
    assert wizard.is_offer_dismissed()

    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    second = FibsemSystemSetupWidget()
    try:
        assert second._frame_first_run.isHidden()
    finally:
        second.deleteLater()


def test_the_offer_is_absent_once_a_configuration_is_registered(
    qapp, isolated_state, monkeypatch
):
    """Setting a microscope up is what ends the first run, however it was done."""
    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    _enable_flag(True)
    cfg.register_configuration(path=cfg.MICROSCOPE_CONFIGURATION_PATH, configuration_name="Bay 2")
    assert not wizard.is_first_run()

    widget = FibsemSystemSetupWidget()
    try:
        assert widget._frame_first_run.isHidden()
    finally:
        widget.deleteLater()


def test_running_the_wizard_selects_what_it_saved(connection_tab, monkeypatch):
    """A configuration written but not selected leaves the tab pointing at the old one."""
    from fibsem.ui import FibsemSystemSetupWidget as module

    path = str(cfg.MICROSCOPE_CONFIGURATION_PATH)
    cfg.USER_CONFIGURATIONS["Arctis Bay 2"] = {"path": path}
    monkeypatch.setattr(
        module, "open_setup_wizard", lambda **kwargs: "Arctis Bay 2", raising=False
    )
    monkeypatch.setattr(
        "fibsem.ui.widgets.setup_wizard_dialog.open_setup_wizard",
        lambda **kwargs: "Arctis Bay 2",
    )

    name = connection_tab.run_setup_wizard()
    assert name == "Arctis Bay 2"
    assert connection_tab.comboBox_configuration.currentText() == "Arctis Bay 2"
    assert connection_tab._frame_first_run.isHidden()


def test_cancelling_the_wizard_changes_nothing_and_keeps_the_offer(
    connection_tab, monkeypatch
):
    """Backing out is not declining -- someone who cancelled to go and find the
    instrument's address should not have to hunt through the menus to get back."""
    before = connection_tab.comboBox_configuration.currentText()
    monkeypatch.setattr(
        "fibsem.ui.widgets.setup_wizard_dialog.open_setup_wizard",
        lambda **kwargs: None,
    )
    assert connection_tab.run_setup_wizard() is None
    assert connection_tab.comboBox_configuration.currentText() == before
    assert not connection_tab._frame_first_run.isHidden()
