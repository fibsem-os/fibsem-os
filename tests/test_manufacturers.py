"""Canonical manufacturer names (FIB-300, PR 1: the foundation).

Three spellings of ThermoFisher and two of TESCAN circulate through configs, image
headers and driver properties; every comparison used to cope locally (defensive
``.upper()``, hand-rolled alias lists) or silently took the wrong branch. These pin
the one alias map and the two normalisation boundaries: ``SystemInfo.from_dict``
(configs and old experiments read canonical) and the driver assignments/properties
(live values are born canonical).
"""

import pytest

from fibsem import manufacturers
from fibsem.manufacturers import (
    DEMO,
    ODEMIS,
    TESCAN,
    THERMOFISHER,
    ZEISS,
    is_tescan,
    is_thermo,
    normalize_manufacturer,
)
from fibsem.structures import SystemInfo

CANONICAL = {THERMOFISHER, TESCAN, DEMO, ODEMIS, ZEISS}


# ---------------------------------------------------------------------------
# the alias map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, canonical",
    [
        # ThermoFisher: the three spellings found in the codebase, plus case noise
        ("Thermo", THERMOFISHER),
        ("ThermoFisher", THERMOFISHER),
        ("Thermo Fisher Scientific", THERMOFISHER),
        ("thermo", THERMOFISHER),
        ("THERMOFISHER", THERMOFISHER),
        # TESCAN: config/property spelling and the live image-header spelling
        ("Tescan", TESCAN),
        ("TESCAN", TESCAN),
        ("tescan", TESCAN),
        ("  Tescan  ", TESCAN),
        ("Demo", DEMO),
        ("DEMO", DEMO),
        ("Odemis", ODEMIS),
        ("zeiss", ZEISS),
    ],
)
def test_known_aliases_normalise(raw, canonical):
    assert normalize_manufacturer(raw) == canonical


@pytest.mark.parametrize("raw", ["Hitachi", "", "Unknown"])
def test_unknown_values_pass_through_unchanged(raw):
    """Lenient by design: reading old or foreign data must never fail here.
    Validation sites keep their own membership checks."""
    assert normalize_manufacturer(raw) == raw


def test_non_strings_pass_through_unchanged():
    assert normalize_manufacturer(None) is None
    assert normalize_manufacturer(42) == 42


def test_predicates_accept_every_spelling():
    assert is_tescan("TESCAN") and is_tescan("Tescan") and is_tescan("tescan")
    assert not is_tescan("ThermoFisher") and not is_tescan(None)
    assert is_thermo("Thermo") and is_thermo("Thermo Fisher Scientific")
    assert not is_thermo("Tescan")


def test_every_canonical_name_normalises_to_itself():
    for name in CANONICAL:
        assert normalize_manufacturer(name) == name


# ---------------------------------------------------------------------------
# boundary: SystemInfo.from_dict normalises on read
# ---------------------------------------------------------------------------


def _info_dict(manufacturer):
    return {
        "name": "test",
        "ip_address": "localhost",
        "manufacturer": manufacturer,
        "model": "model",
        "serial_number": "sn",
        "hardware_version": "hw",
        "software_version": "sw",
    }


@pytest.mark.parametrize(
    "stored, expected",
    [
        ("TESCAN", TESCAN),  # live image header, pre-normalisation experiments
        ("Thermo", THERMOFISHER),  # config-file spelling
        ("Tescan", TESCAN),
        ("Demo", DEMO),
        ("Hitachi", "Hitachi"),  # unknown: kept verbatim, never dropped
    ],
)
def test_system_info_from_dict_normalises_manufacturer(stored, expected):
    info = SystemInfo.from_dict(_info_dict(stored))
    assert info.manufacturer == expected


def test_system_info_round_trip_stores_canonical():
    info = SystemInfo.from_dict(_info_dict("Thermo"))
    assert info.to_dict()["manufacturer"] == THERMOFISHER
    # and a second read is a fixed point
    assert SystemInfo.from_dict(info.to_dict()).manufacturer == THERMOFISHER


def test_system_info_missing_manufacturer_defaults_unknown():
    d = _info_dict("x")
    del d["manufacturer"]
    assert SystemInfo.from_dict(d).manufacturer == "Unknown"


# ---------------------------------------------------------------------------
# boundary: driver properties are born canonical
# ---------------------------------------------------------------------------


def test_thermo_property_is_canonical():
    from fibsem.microscope import FibsemMicroscope, ThermoMicroscope

    # the base-class property (which ThermoMicroscope inherits) -- evaluated
    # without an instance, since it returns a constant
    value = FibsemMicroscope.manufacturer.fget(object.__new__(ThermoMicroscope))
    assert value == THERMOFISHER


def test_tescan_property_is_canonical():
    from fibsem.microscopes.tescan import TescanMicroscope

    value = TescanMicroscope.manufacturer.fget(object.__new__(TescanMicroscope))
    assert value == TESCAN


def test_module_namespace_access():
    """Call sites use `manufacturers.TESCAN` -- keep the module-level names stable."""
    assert manufacturers.TESCAN == "Tescan"
    assert manufacturers.THERMOFISHER == "ThermoFisher"


# ---------------------------------------------------------------------------
# the ratchet: no new .upper() band-aids
# ---------------------------------------------------------------------------


def test_no_upper_tescan_band_aids_outside_the_allowlist():
    """`.upper() == "TESCAN"` was each call site's local coping mechanism for the
    casing split; new comparisons go through `manufacturers.is_tescan`. The two
    widget sites remain until the validation/config part lands -- the allowlist
    then empties and never grows."""
    import os
    from pathlib import Path

    import fibsem

    ALLOWED = {
        "fibsem/ui/widgets/milling_stages_widget.py",
        "fibsem/ui/widgets/beam_settings_widget.py",
    }
    root = Path(fibsem.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        rel = "fibsem/" + str(path.relative_to(root)).replace(os.sep, "/")
        if (
            'upper() == "TESCAN"' in path.read_text(encoding="utf-8")
            and rel not in ALLOWED
        ):
            offenders.append(rel)
    assert offenders == [], (
        'new `.upper() == "TESCAN"` band-aid(s) -- use manufacturers.is_tescan: '
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# boundary: setup_session dispatches on canonical, whatever the caller spelled
# ---------------------------------------------------------------------------


def test_setup_session_accepts_any_demo_spelling(tmp_path):
    from fibsem import utils

    microscope, settings = utils.setup_session(
        session_path=str(tmp_path), manufacturer="DEMO", setup_logging=False
    )
    assert type(microscope).__name__ == "DemoMicroscope"
    assert settings.system.info.manufacturer == DEMO
    microscope.disconnect()
