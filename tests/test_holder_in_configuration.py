"""The sample holder lives in the microscope configuration.

It used to live in `sample-holder.yaml`, a second file beside the configuration with
its own loader and no version. `stage.holders` is a keyed map with an
`active_holder`, so a site that swaps a flat shuttle for a pre-tilted one selects the
other entry instead of re-entering its geometry.

The migration is what these tests are mostly about. A calibrated holder's slot
positions were captured by somebody standing at the microscope; they cannot be
regenerated, and losing them is worse than any failure mode this refactor was meant to
fix.
"""

import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes import _stage as stage_module
from fibsem.microscopes._stage import _resolve_configured_holder
from fibsem.structures import (
    FibsemStagePosition,
    GridSlot,
    SampleHolder,
    SlotCalibration,
    StageSystemSettings,
    SystemSettings,
)


def _calibrated_holder(name: str = "Pre-Tilted 35deg Shuttle") -> SampleHolder:
    """A holder as a calibrated site has it: a real position and the record proving it."""
    return SampleHolder(
        name=name,
        capacity=2,
        slots={
            "Slot-01": GridSlot(
                name="Slot-01",
                index=0,
                position=FibsemStagePosition(
                    name="Slot-01", x=1.5e-3, y=-2.5e-3, z=4.0e-3, r=0.5, t=0.61
                ),
                calibration=SlotCalibration(
                    orientation="SEM",
                    pre_tilt=35.0,
                    rotation_reference=110.0,
                    captured_at="2026-01-01T00:00:00",
                    fibsem_version="v0.5.2",
                ),
            ),
            "Slot-02": GridSlot(name="Slot-02", index=1),
        },
    )


def _stage_settings(**overrides) -> StageSystemSettings:
    fields = dict(rotation_reference=110.0, shuttle_pre_tilt=35.0)
    fields.update(overrides)
    return StageSystemSettings(**fields)


# ---------------------------------------------------------------------------
# It survives the configuration round trip
# ---------------------------------------------------------------------------


def test_a_holder_survives_the_configuration_round_trip():
    """Including the calibration record, which is the part that cannot be recreated."""
    holder = _calibrated_holder()
    stage = _stage_settings(holders={holder.name: holder}, active_holder=holder.name)

    restored = StageSystemSettings.from_dict(stage.to_dict())

    slot = restored.holders[holder.name].slots["Slot-01"]
    assert restored.active_holder == holder.name
    assert slot.position.x == pytest.approx(1.5e-3)
    assert slot.position.t == pytest.approx(0.61)
    assert slot.calibration.pre_tilt == 35.0
    assert slot.calibration.rotation_reference == 110.0
    assert slot.is_calibrated


def test_the_grids_in_the_slots_are_not_written_into_the_configuration():
    """Occupancy is session state and has its own file.

    Writing it here would make the configuration go stale every time someone swapped a
    grid, and make a saved configuration claim a grid is loaded that was taken out
    weeks ago.
    """
    from fibsem.structures import SampleGrid

    holder = _calibrated_holder()
    holder.slots["Slot-01"].loaded_grid = SampleGrid(name="grid-A")
    stage = _stage_settings(holders={holder.name: holder}, active_holder=holder.name)

    written = stage.to_dict()["holders"][holder.name]["slots"]["Slot-01"]

    assert written["loaded_grid"] is None
    assert written["position"] is not None  # the calibration still goes


def test_a_configuration_with_no_holders_loads():
    """Every configuration in the field is one of these."""
    stage = StageSystemSettings.from_dict({})
    assert stage.holders == {}
    assert stage.active_holder == ""


def test_the_holder_reaches_system_settings():
    holder = _calibrated_holder()
    system = SystemSettings.from_dict(
        {
            "stage": {
                "holders": {holder.name: holder.to_dict()},
                "active_holder": holder.name,
            }
        }
    )
    assert system.stage.holders[holder.name].slots["Slot-01"].is_calibrated


# ---------------------------------------------------------------------------
# Resolving which holder is on the stage
# ---------------------------------------------------------------------------


def test_a_configured_holder_is_used_and_no_file_is_read(monkeypatch, tmp_path):
    """The end state: the configuration answers, and `sample-holder.yaml` is irrelevant.

    The path is pointed at a file that does not exist, so a resolution that still
    reached for it would raise rather than quietly pass.
    """
    monkeypatch.setattr(
        stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(tmp_path / "absent.yaml")
    )
    holder = _calibrated_holder()
    stage = _stage_settings(holders={holder.name: holder}, active_holder=holder.name)

    resolved = _resolve_configured_holder(stage)

    assert resolved is holder


def test_an_existing_holder_file_is_imported_with_its_calibration(
    monkeypatch, tmp_path
):
    """The migration, and the only test here that really matters.

    A site upgrading has a calibrated `sample-holder.yaml` and no `holders:` block.
    Its slot positions must arrive intact, and be selected, without anyone being asked
    to choose from a list.
    """
    path = tmp_path / "sample-holder.yaml"
    path.write_text(yaml.dump(_calibrated_holder().to_dict(include_grids=False)))
    monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))

    stage = _stage_settings()
    resolved = _resolve_configured_holder(stage)

    slot = resolved.slots["Slot-01"]
    assert slot.position.x == pytest.approx(1.5e-3)
    assert slot.calibration.pre_tilt == 35.0
    assert slot.is_calibrated, "the calibration record did not survive the import"

    # imported into the configuration, and selected
    assert stage.active_holder == "Pre-Tilted 35deg Shuttle"
    assert stage.holders[stage.active_holder] is resolved


def test_the_migration_does_not_rewrite_either_file(monkeypatch, tmp_path):
    """Nothing is written back at connect time.

    The imported holder is on `stage.holders`, so saving the configuration keeps it --
    but a session that saves nothing must leave the user's files exactly as it found
    them. Rewriting a configuration on someone's behalf while they are connecting is
    how a good migration becomes a support ticket.
    """
    path = tmp_path / "sample-holder.yaml"
    original = yaml.dump(_calibrated_holder().to_dict(include_grids=False))
    path.write_text(original)
    monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))

    _resolve_configured_holder(_stage_settings())

    assert path.read_text() == original


def test_the_shipped_default_is_used_when_there_is_neither(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(tmp_path / "absent.yaml")
    )
    stage = _stage_settings()

    resolved = _resolve_configured_holder(stage)

    assert resolved.capacity == 2
    assert stage.active_holder == resolved.name


def test_resolving_twice_is_stable(monkeypatch, tmp_path):
    """The migration runs every session until the configuration is saved, so it has to
    be idempotent -- not accumulate an entry per connect."""
    path = tmp_path / "sample-holder.yaml"
    path.write_text(yaml.dump(_calibrated_holder().to_dict(include_grids=False)))
    monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))

    stage = _stage_settings()
    first = _resolve_configured_holder(stage)
    second = _resolve_configured_holder(stage)

    assert len(stage.holders) == 1
    assert second is first, "the second resolution re-imported instead of reusing"


def test_a_second_holder_can_sit_alongside_the_active_one():
    """The reason this is a map and not a field: a site with two shuttles keeps both
    calibrated and selects between them."""
    pre_tilted = _calibrated_holder("Pre-Tilted 35deg Shuttle")
    flat = _calibrated_holder("Flat Shuttle")
    stage = _stage_settings(
        holders={pre_tilted.name: pre_tilted, flat.name: flat},
        active_holder=flat.name,
    )

    assert _resolve_configured_holder(stage) is flat
    restored = StageSystemSettings.from_dict(stage.to_dict())
    assert set(restored.holders) == {"Pre-Tilted 35deg Shuttle", "Flat Shuttle"}
    assert restored.active_holder == "Flat Shuttle"


def test_an_active_holder_naming_nothing_falls_back_rather_than_raising(
    monkeypatch, tmp_path
):
    """A hand-edited configuration can name a holder that is not in the map.

    A KeyError here would stop the microscope from connecting over a typo, so it
    resolves the way a configuration with no selection does.
    """
    monkeypatch.setattr(
        stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(tmp_path / "absent.yaml")
    )
    holder = _calibrated_holder()
    stage = _stage_settings(holders={holder.name: holder}, active_holder="not-a-holder")

    resolved = _resolve_configured_holder(stage)

    assert resolved is not None
    assert stage.active_holder == resolved.name


def test_the_schema_knows_about_the_new_keys():
    from fibsem import utils

    assert (
        utils.unrecognised_configuration_keys(
            {"stage": {"holders": {}, "active_holder": "x"}}
        )
        == []
    )


def test_the_default_holder_file_still_ships():
    """The migration's third case reads it, so its absence would be a silent break."""
    assert os.path.exists(cfg.DEFAULT_SAMPLE_HOLDER_CONFIGURATION_PATH)


# ---------------------------------------------------------------------------
# Pressing Apply does not forget which holder is in the shuttle
# ---------------------------------------------------------------------------


def test_applying_a_configuration_keeps_the_holder_selection():
    """`apply_configuration` replaces `system.stage` wholesale.

    The same trap `rotation` fell into, one field over: a configuration written before
    the holder moved into it has no `holders:`, so Apply would empty the map and clear
    the selection while the running `Stage` carried on using a holder the configuration
    no longer knew about. Which holder is in the shuttle is a physical fact and
    pressing Apply did not change it.
    """
    path = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")

    holder = _calibrated_holder()
    microscope.system.stage.holders = {holder.name: holder}
    microscope.system.stage.active_holder = holder.name

    # a real load of the same file, which is what the Apply button does
    from_file = SystemSettings.from_dict(utils.load_yaml(path))
    assert not from_file.stage.holders, "fixture no longer exercises the trap"
    microscope.apply_configuration(from_file)

    assert microscope.system.stage.active_holder == holder.name
    assert microscope.system.stage.holders[holder.name] is holder


def test_a_configuration_that_names_holders_wins_over_the_live_one():
    """The other half: a file that does name holders is a user choosing one."""
    path = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")

    microscope.system.stage.holders = {"old": _calibrated_holder("old")}
    microscope.system.stage.active_holder = "old"

    chosen = _calibrated_holder("Flat Shuttle")
    incoming = SystemSettings.from_dict(utils.load_yaml(path))
    incoming.stage.holders = {chosen.name: chosen}
    incoming.stage.active_holder = chosen.name
    microscope.apply_configuration(incoming)

    assert microscope.system.stage.active_holder == "Flat Shuttle"
    assert set(microscope.system.stage.holders) == {"Flat Shuttle"}
    # And the running Stage moves with it: a configuration record naming one holder
    # while the Stage still drives to another's slots is the disagreement this exists
    # to rule out.
    assert microscope._stage.holder is microscope.system.stage.holders["Flat Shuttle"]
    assert microscope._stage.holder._parent is microscope


# ---------------------------------------------------------------------------
# Pre-tilt belongs to the holder
# ---------------------------------------------------------------------------


def test_the_stage_reads_its_pre_tilt_from_the_active_holder():
    """Physically correct: swap a 35 degree shuttle for a flat one and the pre-tilt
    changes with it, rather than needing the stage block edited by hand."""
    pre_tilted = SampleHolder(name="Pre-Tilted", pre_tilt=35.0)
    flat = SampleHolder(name="Flat", pre_tilt=0.0)
    stage = _stage_settings(
        holders={"Pre-Tilted": pre_tilted, "Flat": flat}, active_holder="Pre-Tilted"
    )

    assert stage.shuttle_pre_tilt == 35.0

    stage.active_holder = "Flat"
    assert stage.shuttle_pre_tilt == 0.0


def test_setting_the_stage_pre_tilt_sets_the_holders():
    """A setter, not a read-only property: around twenty-five test files use
    `stage.shuttle_pre_tilt = 35` as their setup idiom, and it reads correctly --
    the stage's pre-tilt *is* whatever holder is on it."""
    holder = SampleHolder(name="h", pre_tilt=35.0)
    stage = _stage_settings(holders={"h": holder}, active_holder="h")

    stage.shuttle_pre_tilt = 12.0

    assert holder.pre_tilt == 12.0
    assert stage.shuttle_pre_tilt == 12.0


def test_a_stage_with_no_holder_keeps_its_configured_pre_tilt():
    """Every `StageSystemSettings` built from a configuration is in this state until
    `_create_sample_stage` resolves a holder. Answering 0.0 here would turn a 35
    degree site flat for the whole of that window."""
    stage = _stage_settings(shuttle_pre_tilt=35.0)
    assert stage.shuttle_pre_tilt == 35.0


def test_a_holder_that_does_not_state_a_pre_tilt_does_not_flatten_the_stage():
    """`None` is not zero.

    Every holder file written before this change is silent on pre-tilt. Reading that
    silence as "flat" is the single most damaging thing this change could do, because
    nothing would report it -- the projections would simply come out wrong.
    """
    silent = SampleHolder(name="legacy")
    assert silent.pre_tilt is None
    stage = _stage_settings(
        shuttle_pre_tilt=35.0, holders={"legacy": silent}, active_holder="legacy"
    )

    assert stage.shuttle_pre_tilt == 35.0


def test_a_holder_imported_from_a_file_takes_the_configured_pre_tilt(
    monkeypatch, tmp_path
):
    """Even when the file states one of its own.

    Holder files carried a `pre_tilt` once, and it has been ignored ever since the
    value became derived from the stage. Honouring it now would silently resurrect a
    number that has not been in effect for however long that file has been sitting
    there -- in the term every projection is built on.
    """
    data = _calibrated_holder().to_dict(include_grids=False)
    data["pre_tilt"] = 15.0  # stale: ignored since it became derived
    path = tmp_path / "sample-holder.yaml"
    path.write_text(yaml.dump(data))
    monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))

    stage = _stage_settings(shuttle_pre_tilt=35.0)
    resolved = _resolve_configured_holder(stage)

    assert resolved.pre_tilt == 35.0, "a stale file pre-tilt was resurrected"
    assert stage.shuttle_pre_tilt == 35.0


def test_the_pre_tilt_survives_the_configuration_round_trip():
    holder = SampleHolder(name="Pre-Tilted", pre_tilt=35.0)
    stage = _stage_settings(
        shuttle_pre_tilt=0.0, holders={"Pre-Tilted": holder}, active_holder="Pre-Tilted"
    )

    restored = StageSystemSettings.from_dict(stage.to_dict())

    assert restored.holders["Pre-Tilted"].pre_tilt == 35.0
    assert restored.shuttle_pre_tilt == 35.0


def test_the_holder_no_longer_reads_the_stage():
    """The recursion this change had to remove.

    `SampleHolder.pre_tilt` was a property reading back from
    `_parent.system.stage.shuttle_pre_tilt`. With the stage now reading the holder,
    leaving that in place would have made the pair recurse until the interpreter gave
    up -- and it would have done so on the first real connect, not in a test.
    """
    assert not isinstance(
        type(SampleHolder(name="h")).__dict__.get("pre_tilt"), property
    )
