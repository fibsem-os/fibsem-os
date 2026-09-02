"""Slot positions are trusted only with a calibration record that matches the stage.

Every holder file in the field before this existed carries positions with no
record, captured by a button that took whatever the stage said at whatever
orientation. Those load as "not calibrated" so nothing moves to them; the wizard
is the only thing that writes a trusted position.
"""

import pytest

from fibsem import utils
from fibsem.microscopes._stage import (
    GridSlot,
    SampleHolder,
    SlotCalibration,
    _create_sample_stage,
)
from fibsem.structures import FibsemStagePosition


def _position(name="Slot-01", x=-5e-3):
    return FibsemStagePosition(name=name, x=x, y=0.0, z=4e-3, r=0.0, t=0.61)


def _record(pre_tilt=35.0, rotation=0.0):
    return SlotCalibration(
        orientation="SEM",
        pre_tilt=pre_tilt,
        rotation_reference=rotation,
        captured_at="2026-09-02T11:20:00",
        fibsem_version="0.5.2",
    )


class TestRecordRoundTrip:
    def test_slot_with_record_survives_the_file(self):
        slot = GridSlot(
            name="Slot-01", index=0, position=_position(), calibration=_record()
        )
        again = GridSlot.from_dict(slot.to_dict())
        assert again.is_calibrated
        assert again.calibration.pre_tilt == 35.0
        assert again.calibration.captured_at == "2026-09-02T11:20:00"

    def test_old_file_without_the_key_loads(self):
        slot = GridSlot.from_dict(
            {"name": "Slot-01", "index": 0, "position": {"x": 1e-3, "y": 0, "z": 0}}
        )
        assert slot.position is not None
        assert slot.calibration is None
        assert not slot.is_calibrated


class TestDiscardUntrustedPositions:
    def _holder(self, **slots):
        holder = SampleHolder(name="h", capacity=2)
        holder._ensure_slots()
        for name, (position, record) in slots.items():
            holder.slots[name].position = position
            holder.slots[name].calibration = record
        return holder

    def test_position_without_a_record_is_discarded(self):
        holder = self._holder(**{"Slot-01": (_position(), None)})
        notes = holder.discard_untrusted_positions(35.0, 0.0)
        assert holder.slots["Slot-01"].position is None
        assert notes == [
            "Slot-01: position discarded because it has no calibration record"
        ]

    def test_matching_record_is_kept(self):
        holder = self._holder(**{"Slot-01": (_position(), _record())})
        assert holder.discard_untrusted_positions(35.0, 0.0) == []
        assert holder.slots["Slot-01"].is_calibrated

    def test_record_against_other_geometry_is_discarded_with_the_numbers(self):
        holder = self._holder(**{"Slot-01": (_position(), _record(pre_tilt=35.0))})
        notes = holder.discard_untrusted_positions(45.0, 0.0)
        assert holder.slots["Slot-01"].position is None
        assert "pre-tilt 35°" in notes[0] and "45°" in notes[0]

    def test_rotation_reference_counts_too(self):
        holder = self._holder(**{"Slot-01": (_position(), _record(rotation=0.0))})
        assert holder.discard_untrusted_positions(35.0, 180.0)
        assert holder.slots["Slot-01"].position is None

    def test_uncalibrated_slots_are_left_alone(self):
        holder = self._holder(**{"Slot-01": (None, None)})
        assert holder.discard_untrusted_positions(35.0, 0.0) == []

    def test_calibrated_slots_property(self):
        holder = self._holder(
            **{
                "Slot-01": (_position(), _record()),
                "Slot-02": (_position("Slot-02", 5e-3), None),
            }
        )
        holder.discard_untrusted_positions(35.0, 0.0)
        assert [s.name for s in holder.calibrated_slots] == ["Slot-01"]


class TestStageRefusesUncalibratedSlots:
    def _fixed(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        microscope._stage = _create_sample_stage(microscope)
        return microscope

    def test_move_to_slot_refuses(self):
        microscope = self._fixed()
        with pytest.raises(ValueError, match="Calibrate slot positions"):
            microscope._stage.move_to_slot("Slot-01")

    def test_move_to_slot_works_once_calibrated(self):
        microscope = self._fixed()
        slot = microscope._stage.holder.slots["Slot-01"]
        slot.position = _position()
        slot.calibration = _record()
        microscope._stage.move_to_slot("Slot-01")
        assert abs(microscope.get_stage_position().x + 5e-3) < 1e-9

    def test_compustage_working_slot_is_the_origin_by_construction(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        stage = _create_sample_stage(microscope)
        slot = stage.holder.slots["Slot-01"]
        assert slot.is_calibrated and slot.calibration.is_builtin
        assert (slot.position.x, slot.position.y) == (0.0, 0.0)


class TestLoadingAnOldFile:
    def test_field_file_positions_are_dropped_on_load(self, tmp_path, monkeypatch):
        import fibsem.microscopes._stage as stage_module

        path = tmp_path / "holder.yaml"
        path.write_text(
            "name: old\ncapacity: 1\nslots:\n  Slot-01:\n    name: Slot-01\n"
            "    index: 0\n    position: {x: -5.0e-3, y: 0.0, z: 0.0}\n"
            "    loaded_grid: {name: grid-a, description: '', radius: 1.0e-3}\n"
        )
        monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        slot = stage.holder.slots["Slot-01"]
        assert slot.position is None  # the number is gone
        assert slot.loaded_grid.name == "grid-a"  # the name is not
        assert path.read_text().count("-5.0e-3") == 1  # the file is untouched
