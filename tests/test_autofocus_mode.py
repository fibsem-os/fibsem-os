"""One AutoFocusMode, and everything that was ever written for it still loads.

There used to be two enums called `AutoFocusMode` -- `fibsem.structures` (NONE=0,
ONCE=1, EVERY_ROW=2, EVERY_TILE=3) for the beam tiler, and `fibsem.fm.structures`
(NONE="none" .. EACH_TILE="each_tile") for the fluorescence tiler. Same four
concepts, different member names, different value types, identical class name.
Two sibling widgets already imported different ones, and `is` comparisons across
them were silently False with no type error to catch it.

They are now one enum. The identity test below is the point of the exercise; the
rest pin the back-compat that made converging them safe.
"""

from pathlib import Path

import pytest
import yaml

import fibsem
from fibsem.structures import AutoFocusMode, OverviewAcquisitionSettings


def test_every_import_path_yields_the_same_enum():
    """The hazard, stated directly: these used to be four different objects."""
    import fibsem.fm
    import fibsem.fm.acquisition
    import fibsem.fm.structures
    import fibsem.fm.timing
    import fibsem.imaging.tiled

    paths = {
        "fibsem.fm.structures": fibsem.fm.structures.AutoFocusMode,
        "fibsem.fm": fibsem.fm.AutoFocusMode,
        "fibsem.fm.acquisition": fibsem.fm.acquisition.AutoFocusMode,
        "fibsem.fm.timing": fibsem.fm.timing.AutoFocusMode,
        "fibsem.imaging.tiled": fibsem.imaging.tiled.AutoFocusMode,
    }
    divergent = [name for name, enum in paths.items() if enum is not AutoFocusMode]
    assert not divergent, (
        f"{divergent} define or import a different AutoFocusMode than "
        "fibsem.structures. Re-export the canonical one; do not redefine it."
    )


def test_iteration_yields_exactly_the_four_modes():
    """Aliases must not leak into the mode combo boxes, which iterate the enum."""
    assert [m.name for m in AutoFocusMode] == ["NONE", "ONCE", "EACH_ROW", "EACH_TILE"]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("EVERY_ROW", AutoFocusMode.EACH_ROW), ("EVERY_TILE", AutoFocusMode.EACH_TILE)],
)
def test_the_beam_side_spellings_are_aliases_not_new_members(alias, canonical):
    assert getattr(AutoFocusMode, alias) is canonical
    assert AutoFocusMode[alias] is canonical   # settings persisted by name
    assert AutoFocusMode(alias) is canonical   # ... and read back by value


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        # canonical values, as written today
        ("none", AutoFocusMode.NONE),
        ("once", AutoFocusMode.ONCE),
        ("each_row", AutoFocusMode.EACH_ROW),
        ("each_tile", AutoFocusMode.EACH_TILE),
        # member names -- how the beam side persisted them
        ("NONE", AutoFocusMode.NONE),
        ("EVERY_ROW", AutoFocusMode.EACH_ROW),
        ("EVERY_TILE", AutoFocusMode.EACH_TILE),
        # the integers this enum's values used to be
        (0, AutoFocusMode.NONE),
        (1, AutoFocusMode.ONCE),
        (2, AutoFocusMode.EACH_ROW),
        (3, AutoFocusMode.EACH_TILE),
    ],
)
def test_older_spellings_still_resolve(stored, expected):
    assert AutoFocusMode(stored) is expected


@pytest.mark.parametrize("bogus", [True, False, 4, -1, "sometimes", ""])
def test_junk_still_raises(bogus):
    """The compatibility hatch must not turn into an accept-anything hatch.

    `True` is called out specifically: `True == 1`, so a dict lookup keyed on the
    legacy integers would have quietly resolved it to ONCE.
    """
    with pytest.raises(ValueError):
        AutoFocusMode(bogus)


@pytest.mark.parametrize("mode", list(AutoFocusMode))
def test_overview_settings_round_trip(mode):
    """The mode now rides on `OverviewAcquisitionSettings` itself.

    It used to live inside an `autofocus_settings` object that held nothing else; that
    class is gone (FIB-646), so this follows the mode to where it went rather than
    being deleted with it.
    """
    s = OverviewAcquisitionSettings(autofocus_mode=mode)
    assert OverviewAcquisitionSettings.from_dict(s.to_dict()).autofocus_mode is mode


def test_the_previously_persisted_name_form_still_loads_from_the_old_shape():
    """Two layers of compatibility have to hold at once here, and they are independent.

    Beam settings on disk say `{"autofocus_settings": {"mode": "EVERY_ROW"}}` -- the old
    *shape* (mode nested inside the settings object) carrying the old *member name*
    (`EVERY_ROW`, since renamed `EACH_ROW`). Both were already handled separately; this
    pins that the shape migration did not drop the name one on the way past.
    """
    def mode_of(raw):
        return OverviewAcquisitionSettings.from_dict(
            {"autofocus_settings": {"mode": raw}}
        ).autofocus_mode

    assert mode_of("EVERY_ROW") is AutoFocusMode.EACH_ROW
    assert mode_of("NONE") is AutoFocusMode.NONE


def test_the_new_shape_also_reads_the_previously_persisted_name_form():
    """And on the new key, for a file written after the split but before a rename."""
    s = OverviewAcquisitionSettings.from_dict({"autofocus_mode": "EVERY_ROW"})
    assert s.autofocus_mode is AutoFocusMode.EACH_ROW


def test_overview_parameters_round_trip():
    from fibsem.fm.structures import OverviewParameters

    for mode in AutoFocusMode:
        params = OverviewParameters(rows=2, cols=3, overlap=0.1, autofocus_mode=mode)
        assert OverviewParameters.from_dict(params.to_dict()).autofocus_mode is mode


def test_the_shipped_fm_configuration_still_parses():
    """The default config has `autofocus_mode: once` in it -- pin it against the enum."""
    from fibsem.fm.structures import OverviewParameters

    path = Path(fibsem.__file__).parent / "config" / "fm" / "fm-configuration-default.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    params = OverviewParameters.from_dict(config["overview_parameters"])
    assert params.autofocus_mode is AutoFocusMode.ONCE
