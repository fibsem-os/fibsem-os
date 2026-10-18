"""Slice scaling factor (zeta) lookup table for refractive-index depth correction.

The lookup table is a 5D grid over optical parameters:
    tilt_deg    : stage tilt angle (0–30 °)
    depth_um    : feature depth below the coverslip (0.5–14.5 µm)
    NA          : objective numerical aperture (0.2–0.9)
    n2          : refractive index of the immersion medium (1.22–1.46)
    wavelength_um: emission wavelength (0.42–0.72 µm)

The output ``zeta`` is the multiplicative correction applied to the apparent
FM depth to obtain the true depth, accounting for the mismatch between the
immersion medium and the sample.

Usage::

    from fibsem.correlation.refractive_index import lookup_zeta, SliceScalingFactorLUT

    # Convenience function (uses module-level lazy singleton)
    zeta = lookup_zeta(tilt_deg=15, depth_um=4.0, NA=0.8, n2=1.34, wavelength_um=0.570)

    # Or use the class directly for repeated queries
    lut = SliceScalingFactorLUT()
    zeta = lut.lookup(15, 4.0, 0.8, 1.34, 0.570)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

#: The table, shipped with the package: the five axes and zeta x 100 as uint16.
#: Every zeta in the source CSV has two decimals, so this holds it exactly, in
#: 66 KB where the CSV took 9 MB. :func:`pack_lut` writes it from that CSV.
_LUT_PATH = Path(__file__).parent / "data" / "zeta_lut.npz"

#: The table's axes, in the order of the zeta grid's dimensions.
_AXES = ("tilt_deg", "depth_um", "n2", "wavelength_um", "NA")


@dataclass
class ZetaParams:
    """Optical parameters used to look up the slice scaling factor."""

    tilt_deg: float  # stage tilt angle (°), 0–30
    depth_um: float  # feature depth below coverslip (µm), 0.5–14.5
    NA: float  # objective numerical aperture, 0.2–0.9
    n2: float  # immersion medium refractive index, 1.22–1.46
    wavelength_um: float  # emission wavelength (µm), 0.42–0.72


def pack_lut(
    csv_path: Union[str, Path], npz_path: Union[str, Path] = _LUT_PATH
) -> None:
    """Write the shipped table from the source CSV (columns ``tilt_deg, depth_um,
    NA, n2, wavelength_um, zeta``): how ``zeta_lut.npz`` was made. Refuses a CSV
    that is not one full grid, or whose zeta has more than two decimals."""
    import pandas as pd

    lut = pd.read_csv(csv_path).sort_values(by=list(_AXES))
    axes = {name: np.sort(lut[name].unique()) for name in _AXES}
    shape = tuple(len(axes[name]) for name in _AXES)
    if len(lut) != int(np.prod(shape)) or lut.duplicated(list(_AXES)).any():
        raise ValueError(
            f"LUT grid is incomplete: expected {int(np.prod(shape))} rows, "
            f"found {len(lut)}"
        )
    zeta_x100 = np.round(lut["zeta"].to_numpy() * 100)
    if not np.allclose(zeta_x100 / 100, lut["zeta"].to_numpy()):
        raise ValueError("zeta has more than two decimals; uint16 x 100 would lose it")
    np.savez_compressed(
        npz_path, zeta_x100=zeta_x100.astype(np.uint16).reshape(shape), **axes
    )


class SliceScalingFactorLUT:
    """5-D linear interpolator over the slice scaling factor lookup table.

    Parameters
    ----------
    path:
        The packed table (see :func:`pack_lut`). Defaults to the one shipped
        with the package.
    """

    def __init__(self, path: Path = _LUT_PATH) -> None:
        with np.load(path) as lut:
            self.tilts: np.ndarray = lut["tilt_deg"]
            self.depths: np.ndarray = lut["depth_um"]
            self.n2s: np.ndarray = lut["n2"]
            self.wavelengths: np.ndarray = lut["wavelength_um"]
            self.nas: np.ndarray = lut["NA"]
            zeta_grid = lut["zeta_x100"] / 100.0

        self._interp = RegularGridInterpolator(
            points=(self.tilts, self.depths, self.n2s, self.wavelengths, self.nas),
            values=zeta_grid,
            method="linear",
            bounds_error=True,
        )

    def lookup(
        self,
        tilt_deg: float,
        depth_um: float,
        NA: float,
        n2: float,
        wavelength_um: float,
    ) -> float:
        """Return the interpolated scaling factor zeta.

        Parameters are clamped to valid grid ranges by the interpolator
        (``bounds_error=True`` will raise if any value is out of range).
        """
        return float(self._interp((tilt_deg, depth_um, n2, wavelength_um, NA)))

    def lookup_params(self, params: ZetaParams) -> float:
        """Return zeta for a :class:`ZetaParams` instance."""
        return self.lookup(
            params.tilt_deg,
            params.depth_um,
            params.NA,
            params.n2,
            params.wavelength_um,
        )


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_lut: Optional[SliceScalingFactorLUT] = None


def get_lut() -> SliceScalingFactorLUT:
    """Return the module-level :class:`SliceScalingFactorLUT` singleton.

    The LUT is loaded from disk on the first call and cached for subsequent
    calls.
    """
    global _lut
    if _lut is None:
        _lut = SliceScalingFactorLUT()
    return _lut


def lookup_zeta(
    tilt_deg: float,
    depth_um: float,
    NA: float,
    n2: float,
    wavelength_um: float,
) -> float:
    """Convenience wrapper: look up zeta using the module-level singleton LUT.

    Parameters
    ----------
    tilt_deg:
        Stage tilt angle in degrees (0–30).
    depth_um:
        Feature depth below the coverslip in micrometres (0.5–14.5).
    NA:
        Objective numerical aperture (0.2–0.9).
    n2:
        Refractive index of the immersion medium (1.22–1.46).
    wavelength_um:
        Emission wavelength in micrometres (0.42–0.72).

    Returns
    -------
    float
        The depth scaling factor zeta (typically 1.2–1.6 for cryo-CLEM).
    """
    return get_lut().lookup(tilt_deg, depth_um, NA, n2, wavelength_um)
