"""A recipe's ``driver`` blocks: each driver's own milling settings, by its name.

A block is read as the class the driver's registry entry names, so a file keeps a
known driver's typed settings and saves back any block it doesn't know unchanged. On
the wire an unknown block, or an unknown key in a known one, is refused.
"""

import pytest

from fibsem.codec import decode
from fibsem.drivers.tescan.settings import TescanMillingSettings
from fibsem.structures.settings import FibsemMillingSettings


def test_a_known_driver_always_has_its_block():
    recipe = FibsemMillingSettings()
    assert recipe.driver["Tescan"] == TescanMillingSettings()
    # a block at its defaults is not written
    assert recipe.to_dict()["driver"] == {}


def test_a_block_reads_as_its_drivers_class_and_writes_only_what_differs():
    recipe = FibsemMillingSettings.from_dict(
        {"driver": {"Tescan": {"preset": "30 keV; 1 nA", "rate": 2e-8}}}
    )
    block = recipe.driver["Tescan"]
    assert isinstance(block, TescanMillingSettings)
    assert (block.preset, block.rate) == ("30 keV; 1 nA", 2e-8)
    assert recipe.to_dict()["driver"] == {
        "Tescan": {"preset": "30 keV; 1 nA", "rate": 2e-8}
    }


def test_a_block_name_reads_in_any_case():
    recipe = FibsemMillingSettings.from_dict({"driver": {"tescan": {"rate": 2e-8}}})
    assert recipe.driver["Tescan"].rate == 2e-8


def test_the_same_block_twice_is_refused():
    with pytest.raises(ValueError, match="Tescan"):
        FibsemMillingSettings.from_dict(
            {"driver": {"tescan": {"rate": 2e-8}, "Tescan": {"rate": 3e-8}}}
        )


def test_another_instruments_block_saves_back_unchanged():
    data = {"driver": {"Jeol": {"probe": 3, "mode": "fast"}}}
    recipe = FibsemMillingSettings.from_dict(data)
    assert recipe.driver["Jeol"] == {"probe": 3, "mode": "fast"}
    assert recipe.to_dict()["driver"] == data["driver"]


def test_the_old_top_level_tescan_keys_are_ignored():
    recipe = FibsemMillingSettings.from_dict(
        {"preset": "30 keV; 1 nA", "spacing": 2.0, "milling_current": 1e-9}
    )
    assert recipe.driver["Tescan"] == TescanMillingSettings()
    assert "preset" not in recipe.to_dict()


def test_a_value_outside_its_limits_is_refused():
    with pytest.raises(ValueError, match="spacing 1000.0 is outside"):
        FibsemMillingSettings.from_dict({"driver": {"Tescan": {"spacing": 1000.0}}})


def test_the_wire_refuses_an_unknown_block_or_key():
    with pytest.raises(ValueError, match="no block is called 'Jeol'"):
        decode(FibsemMillingSettings, {"driver": {"Jeol": {}}}, strict=True)
    with pytest.raises(ValueError, match="Tescan.ratee is not a setting"):
        decode(FibsemMillingSettings, {"driver": {"Tescan": {"ratee": 1}}}, strict=True)


def test_a_setting_is_read_and_set_by_its_path():
    recipe = FibsemMillingSettings()
    recipe.set_setting("driver.Tescan.preset", "30 keV; 1 nA")
    recipe.set_setting("hfw", 80e-6)
    assert recipe.setting("driver.Tescan.preset") == "30 keV; 1 nA"
    assert recipe.setting("hfw") == 80e-6
    with pytest.raises(ValueError, match="outside"):
        recipe.set_setting("driver.Tescan.spacing", -1.0)


def test_a_patch_reaches_a_drivers_own_setting():
    from fibsem.workflows.config_patch import apply_patch

    recipe = FibsemMillingSettings()
    apply_patch(recipe, {"driver.Tescan.rate": 2e-8})
    assert recipe.driver["Tescan"].rate == 2e-8
