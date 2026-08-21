"""The RI widget's default correction factor must come from the lookup table.

FIB-593: the widget declared `_DEFAULT_FACTOR = 1.47` beside a lookup table that
returned 1.595 for the widget's own default parameters. Two sources of truth for
one number, with nothing holding them together, so the reset button handed back a
factor that described none of the settings displayed above it.

The factor is derived now. What still needs pinning is the offline fallback — a
literal, and therefore free to drift again the moment the defaults move.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.refractive_index import _LUT_PATH, lookup_zeta
from fibsem.ui.correlation.widgets.refractive_index_widget import (
    _DEFAULTS,
    _FALLBACK_FACTOR,
    _default_factor,
)

# Unlike most tests in this directory, these need the *real* table: they are
# about whether our constants agree with it, so a synthetic stand-in would
# assert nothing. See FIB-637 — once the LUT ships in the package this skip goes.
requires_lut = pytest.mark.skipif(
    not _LUT_PATH.exists(), reason="real refractive-index LUT not present"
)


@requires_lut
def test_the_fallback_matches_the_lookup_table():
    """The guard that was missing.

    `_FALLBACK_FACTOR` stands in for the LUT when it cannot be read, so it has to
    be the value the LUT would have given. Changing `_DEFAULTS` without changing
    it is exactly how 1.47 came to describe nothing — this test fails instead.
    """
    expected = lookup_zeta(
        tilt_deg=_DEFAULTS.tilt_deg,
        depth_um=_DEFAULTS.depth_um,
        NA=_DEFAULTS.NA,
        n2=_DEFAULTS.n2,
        wavelength_um=_DEFAULTS.wavelength_um,
    )
    assert _FALLBACK_FACTOR == pytest.approx(expected, abs=5e-4), (
        f"_FALLBACK_FACTOR is {_FALLBACK_FACTOR}, but the LUT gives {expected:.4f} "
        f"at _DEFAULTS. Update the constant to match, or revert the defaults."
    )


@requires_lut
def test_the_default_factor_is_read_from_the_table_not_the_constant():
    """Derivation must be live, not a copy of the literal.

    If `_default_factor()` ever returned `_FALLBACK_FACTOR` unconditionally the
    test above would still pass, and the drift it exists to catch would be back.
    """
    expected = lookup_zeta(
        tilt_deg=_DEFAULTS.tilt_deg,
        depth_um=_DEFAULTS.depth_um,
        NA=_DEFAULTS.NA,
        n2=_DEFAULTS.n2,
        wavelength_um=_DEFAULTS.wavelength_um,
    )
    assert _default_factor() == pytest.approx(expected, abs=1e-9)


@requires_lut
def test_the_defaults_reproduce_the_published_scaling_factor():
    """`_DEFAULTS` describes a real instrument, not an arbitrary point.

    Perez et al. (doi:10.64898/2026.05.11.724418) calculate 1.5 for vitrified
    cytoplasm (n = 1.35) through the Arctis iFLM's NA 0.75 objective and use it
    for depth correction throughout. Those are the widget's defaults, so the
    factor it offers should be the paper's. This is the reason the defaults are
    what they are; drift here means the widget has quietly stopped matching the
    method it implements.
    """
    assert _DEFAULTS.NA == 0.75
    assert _DEFAULTS.n2 == 1.35
    assert _default_factor() == pytest.approx(1.5, abs=0.01)


def test_an_unusable_lookup_table_yields_the_fallback(monkeypatch):
    """Offline is the case the literal exists for — it must not raise."""
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    def unusable(**kwargs):
        raise FileNotFoundError("no lookup table")

    monkeypatch.setattr(riw, "lookup_zeta", unusable)

    assert riw._default_factor() == _FALLBACK_FACTOR
