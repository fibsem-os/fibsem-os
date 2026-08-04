"""Registering imported microscope configurations (fibsem/config.py).

Importing a configuration used to be conditional on the user accepting a prompt,
while the combo box entry was added either way. Declining left the combo box
holding a name that resolved to nothing, and selecting it raised a KeyError out
of a Qt slot -- which aborts the process rather than logging. Registration is now
unconditional, so it has to tolerate the names it will be handed repeatedly.
"""

import os

import pytest

import fibsem.config as cfg


@pytest.fixture
def user_configurations(tmp_path, monkeypatch) -> dict:
    """Isolate the module-level user configuration state and its yaml file."""
    configurations = {"default-configuration": {"path": str(tmp_path / "default.yaml")}}
    yaml_state = {"configurations": configurations, "default": "default-configuration"}

    monkeypatch.setattr(cfg, "USER_CONFIGURATIONS", configurations)
    monkeypatch.setattr(cfg, "USER_CONFIGURATIONS_YAML", yaml_state)
    monkeypatch.setattr(
        cfg, "USER_CONFIGURATIONS_PATH", str(tmp_path / "user-configurations.yaml")
    )
    return configurations


def test_registers_a_new_configuration(user_configurations, tmp_path):
    path = str(tmp_path / "my-microscope.yaml")

    name = cfg.register_configuration(path=path)

    assert name == "my-microscope"
    assert user_configurations[name]["path"] == path


def test_registration_persists_to_disk(user_configurations, tmp_path):
    cfg.register_configuration(path=str(tmp_path / "my-microscope.yaml"))

    assert os.path.exists(cfg.USER_CONFIGURATIONS_PATH)
    written = cfg.load_yaml(cfg.USER_CONFIGURATIONS_PATH)
    assert "my-microscope" in written["configurations"]


def test_reimporting_the_same_file_is_a_no_op(user_configurations, tmp_path):
    path = str(tmp_path / "my-microscope.yaml")

    first = cfg.register_configuration(path=path)
    second = cfg.register_configuration(path=path)

    assert first == second == "my-microscope"
    assert len(user_configurations) == 2  # the default, plus this one


def test_reimport_survives_a_path_written_in_another_form(user_configurations, tmp_path):
    """The stored path and the one the file dialog hands back need not match textually."""
    path = str(tmp_path / "my-microscope.yaml")

    first = cfg.register_configuration(path=path)
    second = cfg.register_configuration(path=str(tmp_path / "." / "my-microscope.yaml"))

    assert first == second
    assert len(user_configurations) == 2


def test_a_different_file_with_the_same_name_gets_its_own_entry(
    user_configurations, tmp_path
):
    """Two configurations can share a basename; neither may displace the other."""
    first_path = str(tmp_path / "a" / "microscope-configuration.yaml")
    second_path = str(tmp_path / "b" / "microscope-configuration.yaml")

    first = cfg.register_configuration(path=first_path)
    second = cfg.register_configuration(path=second_path)
    third = cfg.register_configuration(path=str(tmp_path / "c" / "microscope-configuration.yaml"))

    assert first == "microscope-configuration"
    assert second == "microscope-configuration (2)"
    assert third == "microscope-configuration (3)"
    assert user_configurations[first]["path"] == first_path
    assert user_configurations[second]["path"] == second_path


def test_registration_never_raises_on_a_taken_name(user_configurations, tmp_path):
    """add_configuration refuses duplicates; register_configuration must not.

    An unconditional import runs this on every file the user picks, including one
    named after an entry they already have, and it is called straight from a
    button handler where raising would take the app down.
    """
    with pytest.raises(ValueError):
        cfg.add_configuration(
            configuration_name="default-configuration", path=str(tmp_path / "other.yaml")
        )

    name = cfg.register_configuration(path=str(tmp_path / "default-configuration.yaml"))
    assert name == "default-configuration (2)"


def test_registered_name_always_resolves_to_a_path(user_configurations, tmp_path):
    """The property the combo box depends on: whatever name comes back is selectable."""
    for path in ["a/config.yaml", "b/config.yaml", "b/config.yaml", "other.yml"]:
        name = cfg.register_configuration(path=str(tmp_path / path))
        assert cfg.USER_CONFIGURATIONS.get(name, {}).get("path") is not None


def test_yml_suffix_is_stripped(user_configurations, tmp_path):
    """The import dialog accepts *.yml too, which .removesuffix('.yaml') left on the name."""
    assert cfg.register_configuration(path=str(tmp_path / "my-microscope.yml")) == "my-microscope"


def test_explicit_name_overrides_the_filename(user_configurations, tmp_path):
    name = cfg.register_configuration(
        path=str(tmp_path / "my-microscope.yaml"), configuration_name="METEOR"
    )

    assert name == "METEOR"
    assert "METEOR" in user_configurations
