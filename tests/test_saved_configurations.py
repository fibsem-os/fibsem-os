"""The saved configurations, read from the file each time."""

import yaml

from fibsem import config as cfg


def _saved(tmp_path, monkeypatch, data):
    saved = tmp_path / "user-configurations.yaml"
    saved.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(cfg, "USER_CONFIGURATIONS_PATH", str(saved))


def test_a_configuration_registered_since_import_is_listed(tmp_path, monkeypatch):
    arctis = tmp_path / "arctis.yaml"
    arctis.write_text("{}")
    configurations = {"arctis": {"path": str(arctis)}, "empty": {"path": None}}
    _saved(
        tmp_path, monkeypatch, {"configurations": configurations, "default": "arctis"}
    )
    assert cfg.saved_configurations() == [
        {"name": "arctis", "path": str(arctis), "default": True}
    ]


def test_a_default_without_its_file_is_the_built_in_one(tmp_path, monkeypatch):
    gone = {"gone": {"path": str(tmp_path / "gone.yaml")}}
    _saved(tmp_path, monkeypatch, {"configurations": gone, "default": "gone"})
    default = [c for c in cfg.saved_configurations() if c["default"]]
    assert default == [
        {
            "name": cfg.DEFAULT_CONFIGURATION_NAME,
            "path": cfg.DEFAULT_CONFIGURATION_PATH,
            "default": True,
        }
    ]
