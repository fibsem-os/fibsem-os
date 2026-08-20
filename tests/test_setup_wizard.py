"""What the setup wizard writes, tested without Qt.

CI installs ``.[test]``, not ``.[ui]``, so every test that imports PyQt5 is skipped
there. The wizard's file-writing half is the half worth protecting -- it creates a
configuration, registers it, and can make it the default -- so it lives in a Qt-free
module and is tested here, where CI actually runs it.
"""

import os

import pytest
import yaml

from fibsem import config as cfg
from fibsem import setup_wizard as wizard


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """The user configuration registry, per test, on disk.

    Module-level globals in ``fibsem.config``, so without this a test would append to
    the developer's real ``user-configurations.yaml`` and could change which
    configuration their application opens with.
    """
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


def test_recognised_models_carry_both_stage_values():
    """The skip rule depends on this, and it is a property of the shipped files.

    If a shipped file ever loses one of the two, the wizard would skip a step that
    nothing had answered and write whatever the file happened to default to.
    """
    from fibsem import utils

    for model in wizard.MICROSCOPE_MODELS:
        if not model.knows_stage:
            continue
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
    assert wizard.default_address(wizard.LOCATION_OFFLINE) == ""


# ---------------------------------------------------------------------------
# Which steps are still questions
# ---------------------------------------------------------------------------


def test_a_recognised_model_answers_the_stage_step():
    for key in ("tfs-arctis", "tfs-hydra", "tfs-aquilos2", "tescan"):
        assert not wizard.SetupChoices(model_key=key).needs_stage_step, key


def test_start_blank_still_has_to_ask():
    assert wizard.SetupChoices(model_key="other").needs_stage_step


def test_offline_does_not_skip_the_stage_step():
    """A simulated instrument still has a pre-tilt, and should behave like the real one."""
    choices = wizard.SetupChoices(
        model_key="other", location_key=wizard.LOCATION_OFFLINE
    )
    assert choices.needs_stage_step


def test_offline_closes_the_manufacturer_question():
    blank = wizard.SetupChoices(model_key="other")
    assert blank.needs_manufacturer
    blank.location_key = wizard.LOCATION_OFFLINE
    assert not blank.needs_manufacturer


# ---------------------------------------------------------------------------
# Building the configuration
# ---------------------------------------------------------------------------


def test_a_recognised_model_keeps_its_shipped_stage_values():
    """The compustage case, which a derivation would get wrong.

    ``tfs-arctis`` ships rotation reference 0 with ``rotation_180`` also 0, because a
    compustage reaches the other side by tilting rather than rotating. Deriving the
    pair would hand it a 180-degree rotation it does not have.
    """
    config = wizard.build_configuration(
        wizard.SetupChoices(model_key="tfs-arctis", name="Bay 2")
    )
    assert config["stage"]["rotation_reference"] == 0
    assert config["stage"]["rotation_180"] == 0


def test_a_supplied_rotation_reference_derives_its_opposite():
    config = wizard.build_configuration(
        wizard.SetupChoices(model_key="other", rotation_reference=250.0, name="Bench")
    )
    assert config["stage"]["rotation_reference"] == 250.0
    assert config["stage"]["rotation_180"] == 70.0


def test_the_name_becomes_the_configurations_name():
    config = wizard.build_configuration(wizard.SetupChoices(name="Arctis Bay 2"))
    assert config["info"]["name"] == "Arctis Bay 2"


def test_offline_forces_the_simulator_whatever_else_was_chosen():
    choices = wizard.SetupChoices(
        model_key="tfs-arctis",
        location_key=wizard.LOCATION_OFFLINE,
        manufacturer="Thermo",
        address="",
    )
    config = wizard.build_configuration(choices)
    assert config["info"]["manufacturer"] == "Demo"


def test_an_offline_arctis_is_a_compustage():
    """The simulator overwrites ``info.model``, so only the flag survives to be read."""
    config = wizard.build_configuration(
        wizard.SetupChoices(
            model_key="tfs-arctis", location_key=wizard.LOCATION_OFFLINE, address=""
        )
    )
    assert config["sim"]["is_compustage"] is True


def test_an_offline_hydra_is_not_a_compustage():
    config = wizard.build_configuration(
        wizard.SetupChoices(
            model_key="tfs-hydra", location_key=wizard.LOCATION_OFFLINE, address=""
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


def test_the_blank_start_takes_its_column_tilts_from_the_manufacturer():
    """Its base file is a Demo configuration, so its ion column tilt is Thermo's.

    Column tilt is geometry -- every projection between the two beams runs through it
    -- so a Tescan set up from the blank start must not inherit 52 degrees.
    """
    config = wizard.build_configuration(
        wizard.SetupChoices(model_key="other", manufacturer="Tescan", name="Bench")
    )
    assert config["info"]["manufacturer"] == "Tescan"
    assert config["ion"]["column_tilt"] == 55


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
        wizard.SetupChoices(name="Bay 2", configuration_directory=str(isolated_registry))
    )
    second = wizard.apply_setup(
        wizard.SetupChoices(name="Bay 2", configuration_directory=str(isolated_registry))
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
            model_key="tfs-arctis",
            location_key=wizard.LOCATION_OFFLINE,
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
        wizard.SetupChoices(name="Bay 2", configuration_directory=str(isolated_registry))
    )
    assert os.path.exists(cfg.USER_CONFIGURATIONS_PATH)
    assert not wizard.is_first_run()


def test_writing_preferences_does_not_end_the_first_run(isolated_registry):
    """The regression test for the defect that made the offer unreachable.

    The feature flag that gates the offer lives in the preferences file. When that
    file's absence *was* the first-run signal, turning the flag on created it and the
    offer could never appear -- through the dialog or by hand, every route to enabling
    it also suppressed it. Any signal a preference write can extinguish is the wrong
    signal for "nothing has been configured yet".
    """
    assert wizard.is_first_run()

    preferences = cfg.load_user_preferences()
    preferences.features.setup_wizard_enabled = True
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
        wizard.SetupChoices(name="Bay 2", configuration_directory=str(isolated_registry))
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
    assert cfg.load_user_preferences().display.setup_wizard_dismissed
