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


def test_bounds_are_checked_in_display_units(config):
    # depth metadata: minimum 0.01, maximum 1000 — µm (scale 1e6), not metres.
    # 2 µm (2e-6 m) is valid; 2 mm (2e-3 m -> 2000 µm) is above the maximum.
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
