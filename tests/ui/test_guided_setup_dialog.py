"""The guided setup dialog, and the first-run offer that launches it.

The arithmetic these tests do not repeat is in ``tests/test_guided_setup.py``, which
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
from fibsem import guided_setup as wizard
from fibsem.ui.widgets.guided_setup_dialog import (
    STEP_CONNECTION,
    STEP_FOLDERS,
    STEP_MICROSCOPE,
    STEP_REVIEW,
    STEP_STAGE,
    ChoiceCard,
    GuidedSetupDialog,
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

    configurations = {
        "default-configuration": {"path": cfg.MICROSCOPE_CONFIGURATION_PATH}
    }
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
    widget = GuidedSetupDialog()
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


def test_every_model_walks_the_same_five_steps(dialog):
    """Nothing is skipped. The compustage is shown the stage step read-only instead,
    so the diagram can say why there is nothing to enter."""
    dialog._select_model("tfs-arctis")
    assert dialog._index == STEP_MICROSCOPE
    dialog._on_next()
    assert dialog._index == STEP_CONNECTION
    dialog._on_next()
    assert dialog._index == STEP_STAGE
    dialog._on_next()
    assert dialog._index == STEP_FOLDERS


def test_start_blank_stops_at_the_stage_step(dialog):
    dialog._select_model("tfs-other")
    dialog._on_next()
    dialog._on_next()
    assert dialog._index == STEP_STAGE


def test_the_compustage_is_shown_the_step_but_cannot_edit_it(dialog):
    """Disabled rather than hidden: an empty page says nothing, while a filled-in one
    that cannot be edited says the answer is already known."""
    dialog._select_model("tfs-arctis")
    dialog._show_step(STEP_STAGE)
    assert not dialog._rotation_spin.isEnabled()
    assert not dialog._pre_tilt_spin.isEnabled()
    assert not dialog.button_read_stage.isEnabled()
    # The panels explain instead of instructing. Telling someone to "confirm it rather
    # than assume it" beside a field they cannot touch is worse than saying nothing.
    assert "compustage" in dialog._rotation_blurb.text()
    assert "Nothing to enter" in dialog._pre_tilt_blurb.text()
    # And the rail says so wherever you are, since it describes the step not the place.
    assert dialog._rail_rows[STEP_STAGE]._note.text() == "set by the configuration"

    # Derived from the data rather than listed, so a model added later is covered
    # without anyone remembering to add it here.
    for model in wizard.MICROSCOPE_MODELS:
        if model.knows_stage:
            continue
        dialog._select_manufacturer(model.manufacturer_key)
        dialog._select_model(model.key)
        assert dialog._rotation_spin.isEnabled(), model.key
        assert dialog._pre_tilt_spin.isEnabled(), model.key
        assert "SEM orientation" in dialog._rotation_blurb.text(), model.key
        assert dialog._rail_rows[STEP_STAGE]._note.text() == "", model.key


def test_the_compustage_writes_its_shipped_values_untouched(dialog):
    """Read-only means the wizard did not ask, so the shipped values stand -- including
    ``rotation: false``, which is what stops the derivation handing a compustage a
    half turn it cannot make (FIB-834)."""
    from fibsem.structures import StageSystemSettings

    dialog._select_model("tfs-arctis")
    dialog._show_step(STEP_STAGE)
    dialog._read_current_step()
    assert dialog.choices.rotation_reference is None
    assert dialog.choices.shuttle_pre_tilt is None

    config = wizard.build_configuration(dialog.choices)
    assert config["stage"]["rotation_reference"] == 0
    stage = StageSystemSettings.from_dict(config["stage"])
    assert stage.rotation is False
    assert stage.rotation_180 == 0


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


def test_back_walks_the_same_steps_as_forwards(dialog):
    """A step reachable only in one direction is a step half the users never see."""
    dialog._select_model("tfs-arctis")
    dialog._show_step(STEP_FOLDERS)
    dialog._on_back()
    assert dialog._index == STEP_STAGE


def test_the_connection_step_is_never_skipped(dialog):
    """The address cannot be implied, and the stage step reads through it."""
    for key in [model.key for model in wizard.MICROSCOPE_MODELS]:
        dialog._select_model(key)
        assert not dialog._is_skipped(STEP_CONNECTION), key


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


def test_choosing_a_manufacturer_shows_only_its_instruments(dialog):
    dialog._select_manufacturer(wizard.MANUFACTURER_TESCAN)
    for model in wizard.MICROSCOPE_MODELS:
        shown = not dialog._model_cards[model.key].isHidden()
        assert shown == (model.manufacturer_key == wizard.MANUFACTURER_TESCAN), (
            model.key
        )


def test_changing_manufacturer_lands_on_one_of_its_instruments(dialog):
    """Left alone, Next would carry a model belonging to the manufacturer just
    abandoned -- a Tescan configuration written as a ThermoFisher."""
    dialog._select_manufacturer(wizard.MANUFACTURER_THERMO)
    dialog._select_model("tfs-hydra")
    dialog._select_manufacturer(wizard.MANUFACTURER_SIMULATOR)
    assert dialog.choices.model.manufacturer_key == wizard.MANUFACTURER_SIMULATOR


def test_an_instrument_survives_reselecting_its_own_manufacturer(dialog):
    """Only a *change* of manufacturer should move the instrument."""
    dialog._select_manufacturer(wizard.MANUFACTURER_THERMO)
    dialog._select_model("tfs-aquilos2")
    dialog._select_manufacturer(wizard.MANUFACTURER_THERMO)
    assert dialog.choices.model_key == "tfs-aquilos2"


def test_the_api_note_says_what_is_missing_on_this_computer(dialog):
    """Said here rather than at connect time, three steps later."""
    dialog._select_manufacturer(wizard.MANUFACTURER_SIMULATOR)
    assert "nothing else to install" in dialog._api_note.text()

    dialog._select_manufacturer(wizard.MANUFACTURER_THERMO)
    note = dialog._api_note.text()
    installed = wizard.api_is_installed(
        wizard.get_manufacturer(wizard.MANUFACTURER_THERMO)
    )
    assert "AutoScript" in note
    # Whichever way this machine answers, the note has to state it rather than warn in
    # general terms about software the user may already have.
    assert ("is installed" in note) is bool(installed)


def test_return_while_editing_does_not_leave_the_step(dialog, qapp):
    """Typing a rotation and pressing Return should commit the number, not advance.

    Qt hands Return to a dialog's default button, and failing that to the first
    ``autoDefault`` button -- which is every QPushButton unless told otherwise. So
    confirming a value the way people confirm values used to skip a step, and on the
    review step it saved outright.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    # QTest.keyClick rather than calling keyPressEvent directly. Calling the handler
    # on the spin box skips the propagation to the dialog that *is* the bug, so the
    # direct version passed with the fix reverted -- a harness friendlier than the
    # real thing, which is the only way this test could have been useless.
    dialog._select_manufacturer(wizard.MANUFACTURER_THERMO)
    dialog._select_model("tfs-hydra")
    dialog.show()
    dialog._show_step(STEP_STAGE)
    dialog._rotation_spin.setFocus()
    QTest.keyClick(dialog._rotation_spin, Qt.Key_Return)
    qapp.processEvents()
    assert dialog._index == STEP_STAGE


def test_the_rail_goes_back_to_a_visited_step(dialog):
    """The rail is the only thing showing where you are in a sequence, so it is what
    people try to click to get back somewhere."""
    dialog._show_step(STEP_STAGE)
    dialog._rail_rows[STEP_MICROSCOPE].clicked.emit()
    assert dialog._index == STEP_MICROSCOPE


def test_the_rail_will_not_jump_forwards(dialog):
    """Forwards would skip the read that happens on leaving a step, so the review
    would be built from answers the steps in between never contributed."""
    dialog._show_step(STEP_MICROSCOPE)
    dialog._rail_rows[STEP_REVIEW].clicked.emit()
    assert dialog._index == STEP_MICROSCOPE
    # The current step is not a way back to itself either.
    dialog._rail_rows[STEP_MICROSCOPE].clicked.emit()
    assert dialog._index == STEP_MICROSCOPE


def test_only_visited_rail_rows_offer_to_be_clicked(dialog):
    """The guard and the affordance are set in different places; they have to agree,
    or the rail invites a click it will refuse."""
    from PyQt5.QtCore import Qt

    dialog._show_step(STEP_STAGE)
    for index, row in enumerate(dialog._rail_rows):
        navigable = row.cursor().shape() == Qt.PointingHandCursor
        assert navigable == (index < STEP_STAGE), index


def test_no_button_claims_the_return_key(dialog):
    """Clearing ``default`` on Next alone would hand Return to the next button along."""
    from PyQt5 import QtWidgets

    for button in dialog.findChildren(QtWidgets.QPushButton):
        assert not button.isDefault(), button.text()
        assert not button.autoDefault(), button.text()


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


def test_the_simulator_has_no_address_to_give(dialog):
    """Chosen in step 1, and step 2 has nothing left to ask."""
    dialog._select_manufacturer(wizard.MANUFACTURER_SIMULATOR)
    dialog._show_step(STEP_CONNECTION)
    assert dialog._address_edit.text() == ""
    assert not dialog._address_edit.isEnabled()
    assert not dialog.button_test.isEnabled()
    for card in dialog._location_cards.values():
        assert not card.isEnabled()
    dialog._read_current_step()
    assert dialog.choices.address == ""


def test_the_connection_step_is_shown_not_skipped_for_the_simulator(dialog):
    """An answered step reads as an answer; an absent one reads as an omission."""
    dialog._select_manufacturer(wizard.MANUFACTURER_SIMULATOR)
    dialog._show_step(STEP_CONNECTION)
    assert dialog._index == STEP_CONNECTION
    assert not dialog._is_skipped(STEP_CONNECTION)


def test_leaving_the_simulator_gives_the_address_back(dialog):
    """The field is disabled rather than cleared, so going back has to re-enable it."""
    dialog._select_manufacturer(wizard.MANUFACTURER_SIMULATOR)
    dialog._show_step(STEP_CONNECTION)
    assert not dialog._address_edit.isEnabled()
    dialog._select_manufacturer(wizard.MANUFACTURER_THERMO)
    assert dialog._address_edit.isEnabled()
    assert dialog._address_edit.text() == cfg.DEFAULT_IP_ADDRESS


def test_an_existing_connection_is_borrowed_rather_than_duplicated(
    qapp, isolated_state, demo_microscope
):
    """Two AutoScript clients from one process is not a thing to discover on a first run.

    A real Demo session rather than a stand-in: the status card reads the connection's
    ``system.info`` to name what answered, and a stub with only the attributes this
    test happens to think of is exactly how that read went unnoticed the first time.
    """
    widget = GuidedSetupDialog(microscope=demo_microscope)
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
    dialog._select_model("tfs-other")
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

    dialog._select_model("tfs-other")
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

    dialog._select_model("tfs-other")
    dialog._microscope = Broken()
    dialog._show_step(STEP_STAGE)
    dialog._on_read_stage()
    assert "no stage" in dialog._stage_note.text()


def test_the_diagram_follows_the_number_that_was_entered(dialog):
    """A picture of 35 degrees beside a spin box reading 27 is worse than no picture."""
    dialog._select_model("tfs-other")
    dialog._show_step(STEP_STAGE)
    dialog._pre_tilt_spin.setValue(12.0)
    assert dialog._stage_diagram._pre_tilt == pytest.approx(12.0)
    dialog._pre_tilt_spin.setValue(0.0)
    assert dialog._stage_diagram._pre_tilt == pytest.approx(0.0)


def test_the_diagram_stays_flat_until_a_stage_is_read(dialog, demo_microscope):
    import numpy as np

    from fibsem.structures import FibsemStagePosition

    dialog._select_model("tfs-other")
    dialog._show_step(STEP_STAGE)
    assert dialog._stage_diagram._stage_tilt is None

    demo_microscope.move_stage_absolute(FibsemStagePosition(t=np.radians(17.0)))
    dialog._microscope = demo_microscope
    dialog._on_read_stage()
    assert dialog._stage_diagram._stage_tilt == pytest.approx(17.0, abs=1e-3)


@pytest.mark.parametrize("pre_tilt", [0.0, 35.0, 45.0])
def test_the_sample_comes_flat_when_the_stage_matches_the_pre_tilt(qapp, pre_tilt):
    """The relationship the diagram exists to let someone check.

    ``microscope.py`` puts the SEM orientation at stage tilt = ``shuttle_pre_tilt``, so setting
    the stage to the number you typed should bring the sample square to the electron
    beam. Somebody can hold that against the real instrument; if the picture disagrees,
    the number is wrong.

    Analytic rather than pixel-based, so it pins the geometry the painter rotates by
    rather than a particular rendering.
    """
    from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

    diagram = StageDiagram(pre_tilt=pre_tilt, stage_tilt=pre_tilt)
    try:
        sem = StageDiagram.beam_direction(0.0)
        assert diagram.surface_tilt() == pytest.approx(0.0, abs=1e-9)
        assert diagram.sample_normal() == pytest.approx(sem, abs=1e-9)
    finally:
        diagram.deleteLater()


@pytest.mark.parametrize(
    "pre_tilt, stage_tilt",
    [
        (35.0, 35.0),
        (35.0, 12.0),
        (0.0, 0.0),
        (0.0, -23.0),
        (0.0, -128.0),
        (0.0, -180.0),
    ],
)
def test_the_drawn_milling_angle_matches_the_codebase_formula(
    qapp, pre_tilt, stage_tilt
):
    """The diagram must not invent its own answer for a quantity the code already computes.

    `convert_stage_tilt_to_milling_angle` is a signed linear expression, and an angle
    between a beam and a plane is always acute -- so it is folded into [0, 90] before
    comparing. Unmirrored only: at the FIB orientation the stage has turned 180 degrees,
    which flips the pre-tilt's sign, and the formula does not model that.
    """
    import numpy as np

    from fibsem.transformations import convert_stage_tilt_to_milling_angle
    from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

    diagram = StageDiagram(pre_tilt, stage_tilt)
    try:
        signed = np.degrees(
            convert_stage_tilt_to_milling_angle(
                np.radians(stage_tilt),
                np.radians(pre_tilt),
                np.radians(StageDiagram.FIB_ANGLE),
            )
        )
        acute = 90.0 - abs(90.0 - (abs(signed) % 180.0))
        assert diagram.milling_angle() == pytest.approx(acute, abs=1e-6)
    finally:
        diagram.deleteLater()


def test_the_milling_orientation_is_the_configured_milling_angle(qapp):
    """The MILLING orientation exists to put the sample at the milling angle, so the
    diagram of it should read back the number that was asked for."""
    from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

    # Stage tilts read from microscope.orientations for a 35 deg shuttle and a
    # compustage, both configured for the default 15 degree milling angle.
    for pre_tilt, stage_tilt in ((35.0, 12.0), (0.0, -23.0)):
        diagram = StageDiagram(pre_tilt, stage_tilt)
        try:
            assert diagram.milling_angle() == pytest.approx(15.0, abs=0.05)
        finally:
            diagram.deleteLater()


def test_the_diagram_opens_on_the_flat_reference(qapp):
    """Before anything is read there is no stage tilt to draw, so it shows the position
    the pre-tilt can be checked at rather than a stage sitting at zero."""
    from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

    diagram = StageDiagram(pre_tilt=35.0)
    try:
        assert diagram._stage_tilt is None
        assert diagram.effective_stage_tilt() == pytest.approx(35.0)
        assert diagram.surface_tilt() == pytest.approx(0.0)
        # A stage genuinely read at zero is a different thing, and says so.
        diagram.set_stage_tilt(0.0)
        assert diagram.effective_stage_tilt() == pytest.approx(0.0)
        assert diagram.surface_tilt() == pytest.approx(-35.0)
    finally:
        diagram.deleteLater()


@pytest.mark.parametrize(
    "pre_tilt, stage_tilt", [(0, 0), (35, 17), (-90, 90), (90, -90)]
)
def test_the_diagram_paints_at_the_extremes(qapp, pre_tilt, stage_tilt):
    """The spin boxes allow the whole range, and a paintEvent that raises is fatal."""
    from PyQt5.QtGui import QPixmap

    from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

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

    For a compustage, which reaches the other side of the grid by tilting rather than
    turning round, a reference typed for some other model is exactly the wrong thing to
    keep -- it would be carried into a file whose ``rotation: false`` describes a
    different stage.
    """
    from fibsem.structures import StageSystemSettings

    dialog._select_model("tfs-other")
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
    assert StageSystemSettings.from_dict(config["stage"]).rotation_180 == 0


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
    # The instrument label carries no manufacturer, because the manufacturer is the
    # row above it.
    assert rows["Microscope"] == "Hydra"
    assert rows["Computer"] == "Microscope PC"
    assert rows["Address"] == "localhost"
    # What the file will say, not what the card said: the review's job is to show what
    # is about to be written.
    assert rows["Manufacturer"] == "ThermoFisher"


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
    dialog._select_model("tfs-other")
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

    from fibsem.ui.widgets import guided_setup_dialog as module

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


@pytest.fixture
def connection_tab(qapp, isolated_state, monkeypatch):
    """The connection tab on a fresh install, since that is what most of these test."""
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    widget = FibsemSystemSetupWidget()
    yield widget
    widget.deleteLater()


def test_the_offer_appears_on_a_fresh_install(connection_tab):
    assert not connection_tab._frame_first_run.isHidden()


def test_writing_preferences_does_not_suppress_the_offer(
    qapp, isolated_state, monkeypatch
):
    """The defect this whole arrangement exists to prevent, at the widget.

    Dismissal used to be inferred from the preferences file's absence, so any write to
    that file -- for any preference -- also ended the first run, and the callout could
    not be reached at all.
    """
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
    preferences = cfg.load_user_preferences()
    preferences.display.sound_enabled = True
    cfg.save_user_preferences(preferences)
    assert os.path.exists(cfg.USER_PREFERENCES_PATH)

    widget = FibsemSystemSetupWidget()
    try:
        assert not widget._frame_first_run.isHidden()
    finally:
        widget.deleteLater()


def test_a_dismissed_offer_stays_dismissed(qapp, isolated_state, monkeypatch):
    """Dismissal is its own answer, not a side effect of anything else."""
    pytest.importorskip("napari")
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast", lambda *a, **k: None
    )
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

    cfg.register_configuration(
        path=cfg.MICROSCOPE_CONFIGURATION_PATH, configuration_name="Bay 2"
    )
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
        module, "open_guided_setup", lambda **kwargs: "Arctis Bay 2", raising=False
    )
    monkeypatch.setattr(
        "fibsem.ui.widgets.guided_setup_dialog.open_guided_setup",
        lambda **kwargs: "Arctis Bay 2",
    )

    name = connection_tab.run_guided_setup()
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
        "fibsem.ui.widgets.guided_setup_dialog.open_guided_setup",
        lambda **kwargs: None,
    )
    assert connection_tab.run_guided_setup() is None
    assert connection_tab.comboBox_configuration.currentText() == before
    assert not connection_tab._frame_first_run.isHidden()
