"""Metadata written by earlier versions, and by real instruments, must keep loading.

Synthetic dicts test the shape the current code believes in, which is the assumption
actually at risk when the schema changes. These fixtures are the real output of two
Thermo instruments and of three generations of this code -- see
`tests/fixtures/metadata/README.md` for their provenance and what was scrubbed.

This is the guard for FIB-481, which replaces the embedded `SystemSettings` with a
compact geometry record. Every value asserted here is one the reprojection path reads.
"""

import json
import os
from typing import Any, Dict

import pytest

from fibsem.structures import FibsemImageMetadata

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "metadata")

# (fixture, model, is_compustage, detected_via)
#
# `detected_via` records which arm of the check in reprojection.py fires:
#
#     "Arctis" in system.info.model or system.sim.get("is_compustage", False)
#
# The distinction matters because the simulator only ever exercises the `sim` arm --
# it reports model "DemoMicroscope" -- so without a real Arctis file the model-name
# arm, which is the one real hardware uses, is never covered.
FIXTURES = [
    ("thermo_arctis_compustage_v3.json", "Arctis", True, "model"),
    ("thermo_aquilos_v3.json", "Aquilos", False, "neither"),
    ("demo_simulator_v4.json", "DemoMicroscope", True, "sim"),
    ("demo_simulator_v5.json", "DemoMicroscope", True, "sim"),
]

ALL = [f[0] for f in FIXTURES]


def load(name: str) -> Dict[str, Any]:
    with open(os.path.join(FIXTURE_DIR, name)) as f:
        return json.load(f)


@pytest.mark.parametrize("name", ALL)
def test_fixture_metadata_still_loads(name: str) -> None:
    """The whole point: a file written by an older build must not raise."""
    metadata = FibsemImageMetadata.from_dict(load(name))

    assert metadata.system is not None
    assert metadata.microscope_state is not None


@pytest.mark.parametrize("name,model,_compustage,_via", FIXTURES)
def test_instrument_identity_survives_the_load(
    name: str, model: str, _compustage: bool, _via: str
) -> None:
    metadata = FibsemImageMetadata.from_dict(load(name))

    assert metadata.system.info.model == model
    # The serial is the key any per-instrument aggregation joins on (FIB-445), so it
    # has to survive even though the fixture's value is scrubbed to a placeholder.
    assert metadata.system.info.serial_number == "0000000"


@pytest.mark.parametrize("name", ALL)
def test_reprojection_geometry_is_readable(name: str) -> None:
    """The five angles `_inverse_y_corrected_stage_movement` needs.

    Asserted as `is not None` rather than by value: the Arctis records 0 for three of
    them, correctly -- a compustage tilts the sample directly instead of using a
    pre-tilted shuttle. Anything inferring "not recorded" from a zero here would break
    compustage instruments and nothing else.
    """
    system = FibsemImageMetadata.from_dict(load(name)).system

    assert system.electron.column_tilt is not None
    assert system.ion.column_tilt is not None
    assert system.stage.shuttle_pre_tilt is not None
    assert system.stage.rotation_reference is not None
    assert system.stage.rotation_180 is not None


@pytest.mark.parametrize("name,model,compustage,via", FIXTURES)
def test_compustage_detection(name: str, model: str, compustage: bool, via: str) -> None:
    """Pins the current behaviour of the detection in reprojection.py:299.

    FIB-481 replaces this expression with a recorded boolean. When it does, this test
    should keep passing against the same fixtures -- that is what makes it a
    migration guard rather than a description.
    """
    system = FibsemImageMetadata.from_dict(load(name)).system

    by_model = "Arctis" in system.info.model
    by_sim = system.sim.get("is_compustage", False)

    assert (by_model or by_sim) is compustage
    assert by_model is (via == "model")
    assert by_sim is (via == "sim")


def test_real_hardware_records_an_empty_sim_dict() -> None:
    """`system.sim` is `{}` on real instruments -- a dict, not None.

    Worth pinning: `reprojection.py` calls `system.sim.get(...)` unguarded, so a None
    here would be an AttributeError on the reprojection path of real acquired data.
    """
    for name in ("thermo_arctis_compustage_v3.json", "thermo_aquilos_v3.json"):
        system = FibsemImageMetadata.from_dict(load(name)).system
        assert system.sim == {}


def test_experiment_reference_spans_three_eras() -> None:
    """The fixtures cover every shape `FibsemExperimentRef` has had.

    v3 has no `name` and its `id` holds the experiment *name* (FIB-446); v4 added
    `name`; v5 dropped the duplicated version fields (FIB-448). All three must load,
    and the v3 case must not be silently backfilled -- a reader has to be able to tell
    that its `id` is a name rather than being handed one that claims to be an ID.
    """
    v3 = load("thermo_arctis_compustage_v3.json")["experiment"]
    v4 = load("demo_simulator_v4.json")["experiment"]
    v5 = load("demo_simulator_v5.json")["experiment"]

    assert "name" not in v3
    assert "name" in v4
    assert set(v5) == {"id", "name", "date"}

    # Keys v5 dropped are still in the older files, and must not stop them loading.
    assert "application_version" in v3
    assert "method" in v3

    from fibsem.structures import FibsemExperimentRef

    assert FibsemExperimentRef.from_dict(v3).name is None
    assert FibsemExperimentRef.from_dict(v3).id == "test-experiment"


def test_compustage_angles_are_recorded_as_zero_not_absent() -> None:
    """The Arctis records 0 for all three stage angles; the Aquilos records real ones.

    This is the trap in migrating the geometry off `SystemSettings`: a shim that reads
    "falsy" as "this file did not record it" and falls back to a default would corrupt
    every compustage image and no other kind, which is close to undetectable without
    a real Arctis file to test against.
    """
    arctis = FibsemImageMetadata.from_dict(load("thermo_arctis_compustage_v3.json")).system
    aquilos = FibsemImageMetadata.from_dict(load("thermo_aquilos_v3.json")).system

    assert (
        arctis.stage.shuttle_pre_tilt,
        arctis.stage.rotation_reference,
        arctis.stage.rotation_180,
    ) == (0, 0, 0)
    assert (
        aquilos.stage.shuttle_pre_tilt,
        aquilos.stage.rotation_reference,
        aquilos.stage.rotation_180,
    ) == (35.0, 110, 290)


@pytest.mark.parametrize("missing", ["stage", "electron", "ion", "manipulator", "gis", "info"])
def test_system_settings_from_dict_is_not_a_safe_migration_path(missing: str) -> None:
    """`SystemSettings.from_dict` is bracket-indexed throughout, so it raises rather
    than degrading when a key is absent.

    Recorded as a test because it constrains how the geometry record is derived: read
    the raw dict with `.get()` chains instead of constructing a full `SystemSettings`
    first, or the migration inherits this on exactly the old files it exists to
    protect. See FIB-481.
    """
    from fibsem.structures import SystemSettings

    full = load("thermo_arctis_compustage_v3.json")["system"]
    partial = {k: v for k, v in full.items() if k != missing}

    with pytest.raises(KeyError):
        SystemSettings.from_dict(partial)


def test_a_removed_system_info_field_does_not_break_the_load() -> None:
    """`demo_simulator_v5.json` carries `system.info.application_version`, which this
    build no longer declares. `SystemInfo.from_dict` builds field by field, so an
    unknown key is ignored rather than raising -- that is what made removing the field
    safe for files already on disk (FIB-448)."""
    raw = load("demo_simulator_v5.json")
    assert "application_version" in raw["system"]["info"]

    info = FibsemImageMetadata.from_dict(raw).system.info
    assert not hasattr(info, "application_version")
    assert info.model == "DemoMicroscope"
