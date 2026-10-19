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


def test_a_field_nothing_reads_is_refused_with_why(config):
    """An edit to a field marked not_used would do nothing, so it is refused
    with the field's note rather than accepted silently (FIB-1297)."""
    path = "milling.mill_rough.alignment.interval"
    before = config.milling["mill_rough"].alignment.interval
    with pytest.raises(PatchError, match="not used"):
        apply_patch(config, {path: 5.0})
    assert config.milling["mill_rough"].alignment.interval == before


def test_the_milling_fields_are_explained_and_the_task_set_ones_marked():
    from fibsem.milling.tasks import (
        FibsemMillingTaskConfig,
        describe_config_fields,
        task_overridden_path,
    )

    fields = describe_config_fields(FibsemMillingTaskConfig())
    assert fields["task"]["alignment.interval"]["unit"] == "s"
    assert fields["task"]["alignment.interval_enabled"]["not_used"]
    # Set from the task's own value at each stage start, so a stage's edit
    # would do nothing.
    assert "field_of_view" in fields["stage"]["milling.hfw"]["not_used"]
    assert "set alignment instead" in fields["stage"]["alignment"]["not_used"]
    assert not any(path.startswith("alignment.") for path in fields["stage"])
    assert task_overridden_path("stages.0.alignment.enabled")
    assert not task_overridden_path("stages.0.milling.milling_current")


# -- a stage's strategy config, and fields that are None (FIB-1478) ---------


@pytest.fixture
def coincident():
    from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
        MillCoincidentTaskConfig,
    )

    return MillCoincidentTaskConfig()


STRATEGY = "milling.mill_coincident.stages.0.strategy.config"


def test_a_strategy_field_patches_by_the_path_the_document_shows(coincident):
    changes = apply_patch(
        coincident, {f"{STRATEGY}.supervised": False, f"{STRATEGY}.timeout": 600}
    )

    strategy = coincident.milling["mill_coincident"].stages[0].strategy
    assert (strategy.supervised, strategy.timeout) == (False, 600)
    assert [path for path, _, _ in changes] == [
        f"{STRATEGY}.supervised",
        f"{STRATEGY}.timeout",
    ]
    saved = coincident.milling["mill_coincident"].stages[0].to_dict()["strategy"]
    assert saved["config"]["timeout"] == 600


def test_a_strategy_field_keeps_its_limits(coincident):
    with pytest.raises(PatchError, match="below the minimum"):
        apply_patch(coincident, {f"{STRATEGY}.timeout": 5})


def test_a_field_that_is_none_takes_its_annotated_type(coincident):
    from fibsem.structures import FibsemRectangle

    rect = {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}
    apply_patch(coincident, {f"{STRATEGY}.bbox": rect})

    bbox = coincident.milling["mill_coincident"].stages[0].strategy.bbox
    assert bbox == FibsemRectangle(0.1, 0.2, 0.3, 0.4)
    apply_patch(coincident, {f"{STRATEGY}.bbox.width": 0.5})
    assert bbox.width == 0.5


def test_a_none_field_refuses_a_value_of_the_wrong_shape(coincident):
    with pytest.raises(PatchError, match="FibsemRectangle"):
        apply_patch(coincident, {f"{STRATEGY}.bbox": {"left": 0.1}})


def test_the_setup_tasks_none_fields_patch():
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTaskConfig,
    )

    config = SetupCoincidenceMillingTaskConfig()
    apply_patch(
        config,
        {
            "objective_position": 1.0e-3,
            "fm_roi": {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0},
        },
    )

    assert config.objective_position == 1.0e-3
    assert config.fm_roi.width == 1.0
