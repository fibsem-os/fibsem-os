"""The dotted-path patch engine, against the real config dataclasses.

Real objects throughout: a MillRoughTaskConfig with its actual TrenchPattern
stages, so type coercion, enum handling, and the display-unit bounds are
exercised against exactly the structures the server patches in production.
"""

import pytest

from fibsem.server.config_patch import PatchError, apply_patch


@pytest.fixture
def config():
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )

    return MillRoughTaskConfig(task_name="Rough Milling")


def test_a_float_field_patches_and_reports_old_and_new(config):
    old = config.milling["mill_rough"].stages[0].pattern.depth
    changes = apply_patch(config, {"milling.mill_rough.stages.0.pattern.depth": 2.5e-6})
    assert changes == [("milling.mill_rough.stages.0.pattern.depth", old, 2.5e-6)]
    assert config.milling["mill_rough"].stages[0].pattern.depth == 2.5e-6


def test_enums_patch_by_member_name(config):
    from fibsem.structures import CrossSectionPattern

    apply_patch(
        config,
        {"milling.mill_rough.stages.0.pattern.cross_section": "Rectangle"},
    )
    assert (
        config.milling["mill_rough"].stages[0].pattern.cross_section
        is CrossSectionPattern.Rectangle
    )
    with pytest.raises(PatchError, match="CrossSectionPattern"):
        apply_patch(
            config,
            {"milling.mill_rough.stages.0.pattern.cross_section": "Diagonal"},
        )


def test_type_mismatches_are_refused(config):
    with pytest.raises(PatchError, match="is a number"):
        apply_patch(config, {"milling.mill_rough.stages.0.pattern.depth": "deep"})
    with pytest.raises(PatchError, match="is a boolean"):
        apply_patch(config, {"sync_polishing_position": 1})


def test_unknown_names_are_refused_with_the_valid_ones(config):
    with pytest.raises(PatchError, match="fields:"):
        apply_patch(config, {"milling.mill_rough.stages.0.pattern.dept": 1e-6})
    with pytest.raises(PatchError, match="known keys"):
        apply_patch(config, {"milling.mill_smooth.stages.0.pattern.depth": 1e-6})
    with pytest.raises(PatchError, match="out of range"):
        apply_patch(config, {"milling.mill_rough.stages.9.pattern.depth": 1e-6})


def test_sections_cannot_be_replaced_wholesale(config):
    with pytest.raises(PatchError, match="section, not a value"):
        apply_patch(config, {"milling.mill_rough.stages.0.pattern": {}})


def test_bounds_are_checked_in_si(config):
    # depth limits: 0.01 µm to 1 mm, declared in metres.
    # 2 µm (2e-6 m) is valid; 2 mm (2e-3 m) is above the maximum.
    apply_patch(config, {"milling.mill_rough.stages.0.pattern.depth": 2e-6})
    with pytest.raises(PatchError, match="above the maximum"):
        apply_patch(config, {"milling.mill_rough.stages.0.pattern.depth": 2e-3})


def test_a_failing_entry_applies_nothing(config):
    stage = config.milling["mill_rough"].stages[0]
    depth_before = stage.pattern.depth
    with pytest.raises(PatchError):
        apply_patch(
            config,
            {
                "milling.mill_rough.stages.0.pattern.depth": 2.5e-6,
                "milling.mill_rough.stages.0.pattern.nope": 1.0,
            },
        )
    assert stage.pattern.depth == depth_before


def test_a_beam_bound_field_keeps_its_own_limits(config):
    # the milling field of view takes its display from the beam, and keeps its
    # own 20 to 950 µm limits
    apply_patch(config, {"milling.mill_rough.stages.0.milling.hfw": 400e-6})
    with pytest.raises(PatchError, match="above the maximum"):
        apply_patch(config, {"milling.mill_rough.stages.0.milling.hfw": 2e-3})


# -- a milling recipe's driver settings (FIB-1437) -----------------------------------

_DRIVER = "milling.mill_rough.stages.0.milling.driver.tescan"


def test_a_driver_setting_not_saved_yet_patches_from_its_default(config):
    import fibsem.drivers.tescan.services  # noqa: F401 - registers the Tescan block

    milling = config.milling["mill_rough"].stages[0].milling
    assert "tescan" not in milling.driver
    changes = apply_patch(config, {f"{_DRIVER}.spot_size": 50e-9})
    assert changes == [(f"{_DRIVER}.spot_size", 1.0e-6, 50e-9)]
    assert milling.driver == {"tescan": {"spot_size": 50e-9}}
    # a saved one patches as any value does, and the driver's limits hold
    apply_patch(config, {f"{_DRIVER}.spot_size": 60e-9})
    assert milling.driver["tescan"]["spot_size"] == 60e-9
    with pytest.raises(PatchError, match="minimum"):
        apply_patch(config, {f"{_DRIVER}.spot_size": 0.0})


def test_a_driver_setting_must_be_declared(config):
    import fibsem.drivers.tescan.services  # noqa: F401 - registers the Tescan block

    with pytest.raises(PatchError, match="not a setting of the tescan driver"):
        apply_patch(config, {f"{_DRIVER}.aperture": 1})
    with pytest.raises(PatchError, match="not a driver"):
        apply_patch(config, {"milling.mill_rough.stages.0.milling.driver.x.y": 1})
    # shared fields are the recipe's own, never in a block
    with pytest.raises(PatchError, match="not a setting"):
        apply_patch(config, {f"{_DRIVER}.patterning_mode": "Serial"})
    assert "tescan" not in config.milling["mill_rough"].stages[0].milling.driver
