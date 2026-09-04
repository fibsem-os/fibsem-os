"""What the guided setup writes, tested without Qt.

CI installs ``.[test]``, not ``.[ui]``, so every test that imports PyQt5 is skipped
there. The wizard's file-writing half is the half worth protecting -- it creates a
configuration, registers it, and can make it the default -- so it lives in a Qt-free
module and is tested here, where CI actually runs it.
"""

import os

import pytest
import yaml

from fibsem import config as cfg
from fibsem import guided_setup as wizard


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """The user configuration registry, per test, on disk.

    Module-level globals in ``fibsem.config``, so without this a test would append to
    the developer's real ``user-configurations.yaml`` and could change which
    configuration their application opens with.
    """
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
    # Self-checking: leaked real state would not error, it would just make every
    # first-run assertion below run against a machine that is already configured.
    assert wizard.is_first_run(), "the isolated registry is not a fresh install"
    return tmp_path


# ---------------------------------------------------------------------------
# The registry of what can be chosen
# ---------------------------------------------------------------------------


def test_every_offered_model_ships_a_configuration_file():
    """A card that cannot be built from is a dead end at step one."""
    for model in wizard.MICROSCOPE_MODELS:
        assert os.path.exists(model.path), f"{model.label} has no {model.filename}"


def test_every_model_carries_both_stage_values():
    """Every model's file is read, either to answer the step or to prefill it.

    If a shipped file lost one of the two, the Arctis would skip a step nothing had
    answered, and the rest would prefill a zero that looks like a measurement.
    """
    from fibsem import utils

    for model in wizard.MICROSCOPE_MODELS:
        stage = utils.load_yaml(model.path).get("stage", {})
        assert "rotation_reference" in stage, model.label
        assert "shuttle_pre_tilt" in stage, model.label


def test_unknown_keys_fall_back_rather_than_raising():
    """Read while painting a dialog, where an exception is a dead window."""
    assert wizard.get_model("no-such-model") is wizard.MICROSCOPE_MODELS[0]
    assert wizard.get_location("no-such-place") is wizard.COMPUTER_LOCATIONS[0]


def test_support_pc_is_the_default_location():
    assert wizard.SetupChoices().location_key == wizard.LOCATION_SUPPORT
    assert wizard.COMPUTER_LOCATIONS[0].key == wizard.LOCATION_SUPPORT


def test_address_follows_the_computer_not_the_model_file():
    """Two shipped files record ``localhost``, which describes a computer.

    Seeding a support PC from ``tfs-arctis-configuration.yaml`` would prefill an
    address that can only ever work on the microscope PC.
    """
    assert wizard.default_address(wizard.LOCATION_SUPPORT) == cfg.DEFAULT_IP_ADDRESS
    assert wizard.default_address(wizard.LOCATION_MICROSCOPE) == "localhost"


# ---------------------------------------------------------------------------
# Which steps are still questions
# ---------------------------------------------------------------------------


def test_only_the_compustage_is_shown_the_stage_step_read_only():
    """The Arctis reaches the other side by tilting, so there is no reference rotation
    to measure, and it has no pre-tilted shuttle. Both values are design facts."""
    assert wizard.SetupChoices(model_key="tfs-arctis").stage_is_readonly
    assert wizard.get_model("tfs-arctis").is_compustage


def test_a_pre_tilted_shuttle_system_still_has_to_ask():
    """Carrying a value is not the same as the value being right for this instrument.

    The pre-tilt belongs to whichever shuttle is fitted and the reference rotation to
    how a sample is loaded here -- site facts, not model facts, so the shipped numbers
    are a starting point rather than an answer.
    """
    for key in ("tfs-hydra", "tfs-aquilos2", "tescan"):
        assert wizard.SetupChoices(model_key=key).stage_is_editable, key


def test_start_blank_still_has_to_ask():
    assert wizard.SetupChoices(model_key="tfs-other").stage_is_editable


def test_the_shipped_values_are_offered_as_a_starting_point():
    """What the step is prefilled with, for the models that are still asked."""
    assert wizard.shipped_stage_values(wizard.get_model("tfs-hydra")) == {
        "rotation_reference": 0.0,
        "shuttle_pre_tilt": 35.0,
    }
    assert wizard.shipped_stage_values(wizard.get_model("tescan")) == {
        "rotation_reference": 180.0,
        "shuttle_pre_tilt": 0.0,
    }


def test_no_model_writes_a_rotation_180():
    """The wizard writes the reference; the opposite is derived from it (FIB-834).

    Asserted over every model, asked and unasked, because the key leaving the file is
    the whole change: a `rotation_180` written here would be a number nothing reads,
    sitting next to the reference it is supposed to track and free to disagree with it.

    Tescan is the case worth having. It ships reference 180, so its opposite is 0 -- and
    a derivation without the modulo would write 360, which is the same rotation spelled
    a way that `rotation_angle_is_smaller` handles but a person reading the file does
    not.
    """
    for model in wizard.MICROSCOPE_MODELS:
        config = wizard.build_configuration(wizard.SetupChoices(model_key=model.key))
        assert "rotation_180" not in config["stage"], model.key


def test_the_derived_opposite_matches_what_each_model_needs():
    """The derivation, read back through the settings that consume it.

    The asked models all rotate, so each lands a half turn from its own reference --
    Tescan at 0, from a reference of 180.
    """
    from fibsem.structures import StageSystemSettings

    expected = {"tfs-hydra": 180.0, "tfs-aquilos2": 180.0, "tescan": 0.0}
    for key, opposite in expected.items():
        config = wizard.build_configuration(wizard.SetupChoices(model_key=key))
        stage = StageSystemSettings.from_dict(config["stage"])
        assert stage.rotation is True, key
        assert stage.rotation_180 == opposite, key


def test_the_simulator_does_not_skip_the_stage_step():
    """A simulated instrument still has a pre-tilt, and should behave like the real one."""
    choices = wizard.SetupChoices(
        manufacturer_key=wizard.MANUFACTURER_SIMULATOR, model_key="sim-demo"
    )
    assert choices.is_simulator
    assert choices.stage_is_editable


def test_the_simulator_closes_the_connection_question():
    """There is no address to reach, so the step explains instead of asking."""
    simulated = wizard.SetupChoices(
        manufacturer_key=wizard.MANUFACTURER_SIMULATOR, model_key="sim-demo"
    )
    assert not simulated.connection_is_editable
    real = wizard.SetupChoices(
        manufacturer_key=wizard.MANUFACTURER_THERMO, model_key="tfs-hydra"
    )
    assert real.connection_is_editable


def test_every_manufacturer_offers_at_least_one_instrument():
    """Picking a manufacturer selects its first instrument, so an empty list would
    leave the step with nothing chosen and Next still enabled."""
    for manufacturer in wizard.MANUFACTURERS:
        assert wizard.models_for(manufacturer.key), manufacturer.key


def test_every_instrument_belongs_to_an_offered_manufacturer():
    keys = {m.key for m in wizard.MANUFACTURERS}
    for model in wizard.MICROSCOPE_MODELS:
        assert model.manufacturer_key in keys, model.key


def test_each_manufacturer_names_its_own_control_software():
    """Naming both would name one piece of software the reader does not have."""
    assert wizard.get_manufacturer(wizard.MANUFACTURER_THERMO).control_software == "xT"
    assert (
        wizard.get_manufacturer(wizard.MANUFACTURER_TESCAN).control_software
        == "Essence"
    )
    # The simulator has no vendor software, so it falls back to the generic phrase
    # rather than naming somebody else's product.
    simulator = wizard.get_manufacturer(wizard.MANUFACTURER_SIMULATOR)
    assert simulator.control_software == "the microscope control software"


def test_the_microscope_pc_card_names_the_chosen_manufacturers_software():
    card = wizard.get_location(wizard.LOCATION_MICROSCOPE)
    thermo = wizard.location_summary(
        card, wizard.get_manufacturer(wizard.MANUFACTURER_THERMO)
    )
    tescan = wizard.location_summary(
        card, wizard.get_manufacturer(wizard.MANUFACTURER_TESCAN)
    )
    assert "xT" in thermo and "Essence" not in thermo
    assert "Essence" in tescan and "xT" not in tescan
    # No placeholder should ever reach the screen as literal braces.
    for location in wizard.COMPUTER_LOCATIONS:
        for manufacturer in wizard.MANUFACTURERS:
            assert "{" not in wizard.location_summary(location, manufacturer)


def test_the_raw_coordinate_caution_appears_only_where_it_is_true():
    """``microscope.py`` drives an ordinary ThermoFisher stage in RAW but a compustage
    in SPECIMEN, so the caution must not follow the manufacturer alone."""
    ordinary = wizard.SetupChoices(
        manufacturer_key=wizard.MANUFACTURER_THERMO, model_key="tfs-hydra"
    )
    assert wizard.STAGE_COORDINATE_SYSTEM in ordinary.stage_coordinate_note
    assert "xT" in ordinary.stage_coordinate_note

    compustage = wizard.SetupChoices(
        manufacturer_key=wizard.MANUFACTURER_THERMO, model_key="tfs-arctis"
    )
    assert compustage.stage_coordinate_note == ""

    for manufacturer_key in (wizard.MANUFACTURER_TESCAN, wizard.MANUFACTURER_SIMULATOR):
        model = wizard.models_for(manufacturer_key)[0]
        choices = wizard.SetupChoices(
            manufacturer_key=manufacturer_key, model_key=model.key
        )
        assert choices.stage_coordinate_note == "", manufacturer_key


def test_only_the_simulator_needs_no_vendor_api():
    """The label on the step is driven by this, so a manufacturer that quietly lost
    its module name would start claiming there is nothing to install."""
    for manufacturer in wizard.MANUFACTURERS:
        needs_api = wizard.api_is_installed(manufacturer) is not None
        assert needs_api is not manufacturer.is_simulator, manufacturer.key


# ---------------------------------------------------------------------------
# Building the configuration
# ---------------------------------------------------------------------------


def test_a_compustage_derives_no_opposite_rotation():
    """The case a blanket `reference + 180` would get wrong.

    A compustage reaches the other side of the grid by tilting, not by turning round,
    so it has no second rotation -- and ``tfs-arctis`` says so with ``rotation: false``
    rather than by setting two numbers equal. Both simulated and real Arctis are
    checked: the simulator is the only place the compustage path runs, so a sim config
    that disagreed with the instrument would be a hole in every compustage test.
    """
    from fibsem.structures import StageSystemSettings

    for key in ("tfs-arctis", "sim-arctis"):
        config = wizard.build_configuration(
            wizard.SetupChoices(model_key=key, name="Bay 2")
        )
        stage = StageSystemSettings.from_dict(config["stage"])
        assert stage.rotation is False, key
        assert stage.rotation_reference == 0, key
        assert stage.rotation_180 == 0, key


def test_a_supplied_rotation_reference_derives_its_opposite():
    from fibsem.structures import StageSystemSettings

    config = wizard.build_configuration(
        wizard.SetupChoices(
            model_key="tfs-other", rotation_reference=250.0, name="Bench"
        )
    )
    assert config["stage"]["rotation_reference"] == 250.0
    assert StageSystemSettings.from_dict(config["stage"]).rotation_180 == 70.0


def test_the_name_becomes_the_configurations_name():
    config = wizard.build_configuration(wizard.SetupChoices(name="Arctis Bay 2"))
    assert config["info"]["name"] == "Arctis Bay 2"


def test_the_simulator_manufacturer_writes_a_demo_configuration():
    choices = wizard.SetupChoices(
        manufacturer_key=wizard.MANUFACTURER_SIMULATOR,
        model_key="sim-demo",
        address="",
    )
    config = wizard.build_configuration(choices)
    assert config["info"]["manufacturer"] == "Demo"


def test_a_simulated_arctis_is_a_compustage():
    """The simulator overwrites ``info.model``, so only the flag survives to be read."""
    config = wizard.build_configuration(
        wizard.SetupChoices(
            manufacturer_key=wizard.MANUFACTURER_SIMULATOR,
            model_key="sim-arctis",
            address="",
        )
    )
    assert config["sim"]["is_compustage"] is True


def test_the_demo_microscope_is_not_a_compustage():
    config = wizard.build_configuration(
        wizard.SetupChoices(
            manufacturer_key=wizard.MANUFACTURER_SIMULATOR,
            model_key="sim-demo",
            address="",
        )
    )
    assert config["sim"]["is_compustage"] is False


def test_a_real_configuration_gains_no_simulator_block():
    """``sim`` means something only to the simulator; elsewhere it is noise in a file
    people read by hand."""
    config = wizard.build_configuration(
        wizard.SetupChoices(model_key="tfs-hydra", address="192.168.0.1")
    )
    assert "is_compustage" not in config.get("sim", {})


def test_the_generic_base_takes_its_manufacturer_from_the_first_step():
    """The generic base is a Demo configuration file.

    Taken at face value it would write ``manufacturer: Demo`` and a setup that can
    never reach an instrument. The manufacturer chosen in step one overrides it.
    """
    config = wizard.build_configuration(
        wizard.SetupChoices(
            manufacturer_key=wizard.MANUFACTURER_THERMO,
            model_key="tfs-other",
            name="Bench",
        )
    )
    assert config["info"]["manufacturer"] == "ThermoFisher"


def test_every_shipped_file_already_agrees_with_its_manufacturers_column_tilts():
    """``build_configuration`` applies the manufacturer's column tilts unconditionally.

    That is only safe because it is a no-op for every shipped file. If one ever
    disagreed, the wizard would silently overwrite a value someone chose deliberately
    -- and column tilt is geometry, so it would mis-place patterns without ever
    looking like a configuration problem.
    """
    from fibsem import utils

    for model in wizard.MICROSCOPE_MODELS:
        shipped = utils.load_yaml(model.path)
        defaults = cfg.DEFAULT_CONFIGURATION_VALUES[model.manufacturer.config_value]
        built = wizard.build_configuration(
            wizard.SetupChoices(
                manufacturer_key=model.manufacturer_key, model_key=model.key
            )
        )
        assert built["ion"]["column_tilt"] == defaults["ion-column-tilt"], model.key
        # The generic base is the one file deliberately allowed to differ, because it
        # is a Demo configuration standing in for someone else's instrument.
        if model.filename != "microscope-configuration.yaml":
            assert shipped["ion"]["column_tilt"] == defaults["ion-column-tilt"], (
                model.key
            )


def test_a_recognised_model_keeps_its_own_column_tilt():
    config = wizard.build_configuration(wizard.SetupChoices(model_key="tfs-hydra"))
    assert config["ion"]["column_tilt"] == 52


# ---------------------------------------------------------------------------
# Naming and file placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Arctis Bay 2", "arctis-bay-2"),
        ("  Hydra  ", "hydra"),
        ("TFS/Aquilos 2", "tfs-aquilos-2"),
        ("", "microscope-configuration"),
        ("!!!", "microscope-configuration"),
    ],
)
def test_configuration_slug(name, expected):
    assert wizard.configuration_slug(name) == expected


def test_a_name_collision_does_not_overwrite_the_file_already_there(tmp_path):
    first = wizard.configuration_path("Bay 2", str(tmp_path))
    open(first, "w").close()
    second = wizard.configuration_path("Bay 2", str(tmp_path))
    assert first != second
    assert os.path.basename(second) == "bay-2-2.yaml"


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------


def test_apply_setup_writes_registers_and_defaults(isolated_registry):
    choices = wizard.SetupChoices(
        model_key="tfs-arctis",
        address="192.168.0.55",
        name="Arctis Bay 2",
        configuration_directory=str(isolated_registry),
        experiment_directory=str(isolated_registry / "experiments"),
    )
    result = wizard.apply_setup(choices)

    assert os.path.exists(result.path)
    with open(result.path) as f:
        written = yaml.safe_load(f)
    assert written["info"]["ip_address"] == "192.168.0.55"
    assert written["info"]["name"] == "Arctis Bay 2"

    # Registering is what makes it selectable; a file nobody registered is a file
    # nobody can choose.
    assert result.configuration_name in cfg.USER_CONFIGURATIONS
    assert cfg.USER_CONFIGURATIONS[result.configuration_name]["path"] == result.path
    assert result.is_default
    assert cfg.USER_CONFIGURATIONS_YAML["default"] == result.configuration_name

    preferences = cfg.load_user_preferences()
    assert preferences.experiment.default_experiment_directory == str(
        isolated_registry / "experiments"
    )


def test_the_configuration_folder_defaults_to_the_shipped_one(isolated_registry):
    """Unanswered means today's location, so no new convention is invented here."""
    assert wizard.SetupChoices().resolved_configuration_directory == cfg.CONFIG_PATH


def test_the_configuration_can_live_outside_the_package(isolated_registry, tmp_path):
    """The whole point of asking: a configuration kept where an upgrade cannot reach it.

    Already supported by the registry -- ``register_configuration`` stores an absolute
    path and ``load_configuration`` loads from it, which is what importing a
    configuration from the connection tab has always relied on. The wizard simply
    never offered the choice.
    """
    elsewhere = tmp_path / "somewhere" / "else"
    result = wizard.apply_setup(
        wizard.SetupChoices(name="Off Package", configuration_directory=str(elsewhere))
    )

    assert os.path.dirname(result.path) == str(elsewhere)
    assert not result.path.startswith(cfg.CONFIG_PATH)
    # Registered by absolute path, so it is selectable from wherever it landed.
    assert cfg.USER_CONFIGURATIONS[result.configuration_name]["path"] == result.path


def test_the_two_folder_answers_do_not_touch_each_other(isolated_registry, tmp_path):
    """They are separate questions, and were being answered as one.

    The configuration folder decides where a file is written; the experiment folder is
    a preference the new-experiment dialog reads. Nothing should make one follow the
    other.
    """
    configurations = tmp_path / "configurations"
    experiments = tmp_path / "experiments"
    result = wizard.apply_setup(
        wizard.SetupChoices(
            name="Split",
            configuration_directory=str(configurations),
            experiment_directory=str(experiments),
        )
    )

    assert os.path.dirname(result.path) == str(configurations)
    saved = cfg.load_user_preferences().experiment.default_experiment_directory
    assert saved == str(experiments)
    assert saved != os.path.dirname(result.path)


def test_apply_setup_can_leave_the_default_alone(isolated_registry):
    choices = wizard.SetupChoices(
        name="Second Rig",
        set_as_default=False,
        configuration_directory=str(isolated_registry),
    )
    result = wizard.apply_setup(choices)
    assert not result.is_default
    assert cfg.USER_CONFIGURATIONS_YAML["default"] == "default-configuration"
    # Still selectable, which is the point of registering separately from defaulting.
    assert result.configuration_name in cfg.USER_CONFIGURATIONS


def test_a_taken_name_is_reported_as_the_name_it_actually_got(isolated_registry):
    """``register_configuration`` suffixes rather than refusing, so the caller has to
    use the name it returns -- selecting the one that was typed would find nothing."""
    wizard.apply_setup(
        wizard.SetupChoices(
            name="Bay 2", configuration_directory=str(isolated_registry)
        )
    )
    second = wizard.apply_setup(
        wizard.SetupChoices(
            name="Bay 2", configuration_directory=str(isolated_registry)
        )
    )
    assert second.configuration_name != "Bay 2"
    assert second.configuration_name in cfg.USER_CONFIGURATIONS


def test_the_written_configuration_loads_back_as_settings(isolated_registry):
    """The end of the wizard is the start of a connection, and this is the seam.

    A file that saves and registers but cannot be turned into ``MicroscopeSettings``
    fails at connect time, several screens away from anything that would explain it.
    """
    from fibsem import utils

    result = wizard.apply_setup(
        wizard.SetupChoices(
            manufacturer_key=wizard.MANUFACTURER_SIMULATOR,
            model_key="sim-arctis",
            address="",
            name="Sim Rig",
            configuration_directory=str(isolated_registry),
        )
    )
    settings = utils.load_microscope_configuration(result.path)
    assert settings.system.info.manufacturer == "Demo"
    assert settings.system.sim["is_compustage"] is True


# ---------------------------------------------------------------------------
# First-run detection
# ---------------------------------------------------------------------------


def test_first_run_is_the_absence_of_a_registered_configuration(isolated_registry):
    assert wizard.is_first_run()
    wizard.apply_setup(
        wizard.SetupChoices(
            name="Bay 2", configuration_directory=str(isolated_registry)
        )
    )
    assert os.path.exists(cfg.USER_CONFIGURATIONS_PATH)
    assert not wizard.is_first_run()


def test_writing_preferences_does_not_end_the_first_run(isolated_registry):
    """The regression test for the defect that made the offer unreachable.

    When dismissal *was* inferred from the preferences file's absence, any write to
    that file -- for any preference, guided setup or otherwise -- looked the same as
    dismissing the offer. Any signal a preference write can extinguish is the wrong
    signal for "nothing has been configured yet".
    """
    assert wizard.is_first_run()

    preferences = cfg.load_user_preferences()
    preferences.display.sound_enabled = True
    cfg.save_user_preferences(preferences)

    assert os.path.exists(cfg.USER_PREFERENCES_PATH)
    assert wizard.is_first_run()


def test_finishing_the_wizard_ends_the_first_run(isolated_registry):
    """Otherwise the offer to set the microscope up would survive setting it up.

    Through the registration rather than the preferences write, so it still holds if
    the folder preferences fail to save.
    """
    assert wizard.is_first_run()
    wizard.apply_setup(
        wizard.SetupChoices(
            name="Bay 2", configuration_directory=str(isolated_registry)
        )
    )
    assert not wizard.is_first_run()


def test_dismissal_is_recorded_without_faking_a_configuration(isolated_registry):
    """Someone who waves the offer away has still configured nothing."""
    assert not wizard.is_offer_dismissed()
    wizard.dismiss_first_run()
    assert wizard.is_offer_dismissed()
    # Still a fresh install -- the dismissal says "do not ask", not "it is done".
    assert wizard.is_first_run()


def test_dismissal_keeps_the_preferences_that_were_already_there(isolated_registry):
    """It records that the offer was declined; it is not a reset."""
    preferences = cfg.load_user_preferences()
    preferences.experiment.user = "someone"
    cfg.save_user_preferences(preferences)

    wizard.dismiss_first_run()
    assert cfg.load_user_preferences().experiment.user == "someone"
    assert cfg.load_user_preferences().display.guided_setup_dismissed
