import pytest

from fibsem import utils
from fibsem.microscopes._stage import (
    GridSlot,
    SampleGrid,
    SampleHolder,
    Stage,
    _create_sample_stage,
)
from fibsem.structures import FibsemStagePosition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(name: str = "Slot-01") -> FibsemStagePosition:
    return FibsemStagePosition(name=name, x=1e-3, y=2e-3, z=3e-3)


def _make_holder(capacity: int = 2, name: str = "Test Holder") -> SampleHolder:
    h = SampleHolder(name=name, capacity=capacity)
    h._ensure_slots()
    return h


# ---------------------------------------------------------------------------
# SampleGrid
# ---------------------------------------------------------------------------


class TestSampleGrid:
    def test_defaults(self):
        g = SampleGrid(name="Grid-A")
        assert g.name == "Grid-A"
        assert g.description == ""

    def test_roundtrip(self):
        g = SampleGrid(name="Grid-B", description="test desc", radius=0.5e-3)
        g2 = SampleGrid.from_dict(g.to_dict())
        assert g2.name == g.name
        assert g2.description == g.description
        assert g2.radius == g.radius


# ---------------------------------------------------------------------------
# GridSlot
# ---------------------------------------------------------------------------


class TestGridSlot:
    def test_roundtrip_empty(self):
        slot = GridSlot(name="Slot-01", index=0, position=_make_position())
        slot2 = GridSlot.from_dict(slot.to_dict())
        assert slot2.name == "Slot-01"
        assert slot2.index == 0
        assert slot2.loaded_grid is None

    def test_roundtrip_with_grid(self):
        grid = SampleGrid(name="Grid-A", description="desc")
        slot = GridSlot(
            name="Slot-01", index=0, position=_make_position(), loaded_grid=grid
        )
        slot2 = GridSlot.from_dict(slot.to_dict())
        assert slot2.loaded_grid is not None
        assert slot2.loaded_grid.name == "Grid-A"
        assert slot2.loaded_grid.description == "desc"


# ---------------------------------------------------------------------------
# SampleHolder construction and properties
# ---------------------------------------------------------------------------


class TestSampleHolderConstruction:
    def test_defaults(self):
        h = SampleHolder()
        assert h.name == "Sample Holder"
        assert h.description == ""
        assert h.capacity == 2
        assert h.slots == {}

    def test_pre_tilt_is_unstated_rather_than_zero(self):
        """`None`, not 0.0, and the distinction is load-bearing.

        A flat shuttle really is zero. A holder that has not been told its pre-tilt is
        a different thing, and the stage falls back to its own configured value for
        it -- reading "unstated" as "flat" would turn every 35 degree site flat at the
        moment this shipped.
        """
        assert SampleHolder().pre_tilt is None

    def test_reference_rotation_no_parent(self):
        h = SampleHolder()
        assert h.reference_rotation == 0.0

    def test_pre_tilt_is_the_holders_own_and_not_read_from_the_stage(self):
        """The direction reversed.

        It used to be a property reading back from
        `_parent.system.stage.shuttle_pre_tilt`. The stage now reads the holder, so a
        holder that read the stage would recurse until the interpreter gave up -- the
        two cannot both exist, and this is the one that stores.
        """
        h = SampleHolder(pre_tilt=35.0)

        class _FakeStage:
            shuttle_pre_tilt = 12.0
            rotation_reference = 0.0

        class _FakeSystem:
            stage = _FakeStage()

        class _FakeMicroscope:
            system = _FakeSystem()

        h._parent = _FakeMicroscope()
        assert h.pre_tilt == 35.0

    def test_reference_rotation_with_parent(self):
        h = SampleHolder()

        class _FakeStage:
            shuttle_pre_tilt = 0.0
            rotation_reference = 180.0

        class _FakeSystem:
            stage = _FakeStage()

        class _FakeMicroscope:
            system = _FakeSystem()

        h._parent = _FakeMicroscope()
        assert h.reference_rotation == 180.0

    def test_pre_tilt_is_serialised_and_reference_rotation_is_not(self):
        """Only the pre-tilt moved onto the holder.

        A reference rotation is the stage's: it describes where the stage's own zero
        is, and that does not change because a different shuttle was fitted.
        """
        h = SampleHolder(capacity=1, pre_tilt=35.0)
        h._ensure_slots()
        d = h.to_dict()
        assert d["pre_tilt"] == 35.0
        assert "reference_rotation" not in d


# ---------------------------------------------------------------------------
# _ensure_slots
# ---------------------------------------------------------------------------


class TestEnsureSlots:
    def test_creates_correct_count(self):
        h = _make_holder(capacity=3)
        assert len(h.slots) == 3
        assert "Slot-01" in h.slots
        assert "Slot-02" in h.slots
        assert "Slot-03" in h.slots

    def test_slot_names_are_zero_padded(self):
        h = _make_holder(capacity=10)
        assert "Slot-10" in h.slots

    def test_reduce_capacity_trims_slots(self):
        h = _make_holder(capacity=4)
        h.capacity = 2
        h._ensure_slots()
        assert len(h.slots) == 2
        assert "Slot-03" not in h.slots
        assert "Slot-04" not in h.slots

    def test_idempotent(self):
        h = _make_holder(capacity=2)
        h._ensure_slots()
        assert len(h.slots) == 2

    def test_new_slots_are_uncalibrated(self):
        h = _make_holder(capacity=2)
        for slot in h.slots.values():
            assert slot.position is None
            assert not slot.is_calibrated


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_keys(self):
        h = _make_holder()
        d = h.to_dict()
        assert "name" in d
        assert "capacity" in d
        assert "slots" in d
        assert "description" in d

    def test_roundtrip_empty_slots(self):
        h = SampleHolder(name="H1", description="desc", capacity=2)
        h._ensure_slots()
        h2 = SampleHolder.from_dict(h.to_dict())
        assert h2.name == "H1"
        assert h2.description == "desc"
        assert h2.capacity == 2
        assert len(h2.slots) == 2

    def test_roundtrip_with_loaded_grid(self):
        h = _make_holder(capacity=1)
        h.slots["Slot-01"].loaded_grid = SampleGrid(name="Grid-A")
        h2 = SampleHolder.from_dict(h.to_dict())
        assert h2.slots["Slot-01"].loaded_grid is not None
        assert h2.slots["Slot-01"].loaded_grid.name == "Grid-A"

    def test_from_dict_reads_pre_tilt_and_still_ignores_reference_rotation(self):
        """`pre_tilt` is the holder's again; `reference_rotation` never comes back.

        Note what this means for a *file* carrying a stale `pre_tilt` -- holder files
        did once, and it has been ignored since the value became derived.
        `_resolve_configured_holder` overwrites it with the configured one on the way
        in rather than resurrecting a number nobody has seen in months. See
        `tests/test_holder_in_configuration.py`.
        """
        h = SampleHolder.from_dict(
            {
                "name": "Old Holder",
                "capacity": 1,
                "description": "",
                "pre_tilt": 15.0,
                "reference_rotation": 90.0,
                "slots": {},
            }
        )
        assert h.name == "Old Holder"
        assert h.pre_tilt == 15.0
        assert not hasattr(h, "_reference_rotation")


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_keeps_geometry_not_grids(self, tmp_path):
        path = tmp_path / "holder.yaml"
        h = _make_holder(capacity=2, name="SavedHolder")
        h.slots["Slot-01"].loaded_grid = SampleGrid(name="Grid-A")
        h.save(path)

        h2 = SampleHolder.load(path)
        assert h2.name == "SavedHolder"
        assert len(h2.slots) == 2
        # what is in a slot is session state, kept in the occupancy file instead
        assert h2.slots["Slot-01"].loaded_grid is None
        assert h2.slots["Slot-02"].loaded_grid is None

    def test_occupancy_round_trips_separately(self, tmp_path):
        path = tmp_path / "occupancy.yaml"
        h = _make_holder(capacity=2)
        h.slots["Slot-02"].loaded_grid = SampleGrid(name="Grid-B", description="d")
        h.save_occupancy(path)

        h2 = _make_holder(capacity=2)
        h2.slots["Slot-01"].loaded_grid = SampleGrid(name="stale")
        assert h2.load_occupancy(path) is True
        assert h2.slots["Slot-01"].loaded_grid is None  # unlisted slots are emptied
        assert h2.slots["Slot-02"].loaded_grid.name == "Grid-B"
        assert h2.slots["Slot-02"].loaded_grid.description == "d"

    def test_load_occupancy_without_a_file_changes_nothing(self, tmp_path):
        h = _make_holder(capacity=1)
        h.slots["Slot-01"].loaded_grid = SampleGrid(name="keep")
        assert h.load_occupancy(tmp_path / "missing.yaml") is False
        assert h.slots["Slot-01"].loaded_grid.name == "keep"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SampleHolder.load(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Slot lookup helpers
# ---------------------------------------------------------------------------


class TestSlotLookup:
    def test_find_slot_for_grid(self):
        h = _make_holder(capacity=2)
        grid = SampleGrid(name="Grid-A")
        h.slots["Slot-01"].loaded_grid = grid
        assert h.find_slot_for_grid(grid) is h.slots["Slot-01"]

    def test_find_slot_for_grid_not_found(self):
        h = _make_holder(capacity=2)
        grid = SampleGrid(name="Grid-X")
        assert h.find_slot_for_grid(grid) is None

    def test_find_slot_by_grid_name(self):
        h = _make_holder(capacity=2)
        h.slots["Slot-02"].loaded_grid = SampleGrid(name="Grid-B")
        assert h.find_slot_by_grid_name("Grid-B") is h.slots["Slot-02"]

    def test_find_slot_by_grid_name_not_found(self):
        h = _make_holder(capacity=2)
        assert h.find_slot_by_grid_name("Missing") is None

    def test_find_slot_by_grid_name_empty_slots(self):
        h = _make_holder(capacity=2)
        assert h.find_slot_by_grid_name("Grid-A") is None


# ---------------------------------------------------------------------------
# _create_sample_stage
# ---------------------------------------------------------------------------


class TestCreateSampleStage:
    def test_compustage_returns_stage(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        stage = _create_sample_stage(microscope)
        assert isinstance(stage, Stage)

    def test_compustage_single_slot(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        stage = _create_sample_stage(microscope)
        assert stage.holder.capacity == 1
        assert len(stage.holder.slots) == 1
        assert "Slot-01" in stage.holder.slots

    def test_compustage_has_loader(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        stage = _create_sample_stage(microscope)
        assert stage.loader is not None

    def test_compustage_parent_set(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        stage = _create_sample_stage(microscope)
        assert stage.holder._parent is microscope

    def test_non_compustage_returns_stage(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        assert isinstance(stage, Stage)

    def test_non_compustage_no_loader(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        assert stage.loader is None

    def test_non_compustage_parent_set(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        assert stage.holder._parent is microscope

    def test_non_compustage_shipped_slots_are_uncalibrated(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        # the shipped file carries no calibrated positions, and nothing invents one
        for slot in stage.holder.slots.values():
            assert slot.position is None
            assert not slot.is_calibrated

    def test_non_compustage_falls_back_to_default(self, tmp_path, monkeypatch):
        import fibsem.microscopes._stage as stage_module

        monkeypatch.setattr(
            stage_module,
            "SAMPLE_HOLDER_CONFIGURATION_PATH",
            str(tmp_path / "missing.yaml"),
        )
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        assert isinstance(stage, Stage)
        assert len(stage.holder.slots) > 0

    def test_non_compustage_loads_user_config_when_present(self, tmp_path, monkeypatch):
        import fibsem.microscopes._stage as stage_module

        path = tmp_path / "holder.yaml"
        h = SampleHolder(name="UserHolder", capacity=3)
        h._ensure_slots()
        h.save(path)
        monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        stage = _create_sample_stage(microscope)
        assert stage.holder.name == "UserHolder"
        assert stage.holder.capacity == 3
