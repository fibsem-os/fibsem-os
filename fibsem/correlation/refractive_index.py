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
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

_LUT_PATH = Path(__file__).parent / "scaling_factor_lookup_table.csv"


@dataclass
class ZetaParams:
    """Optical parameters used to look up the slice scaling factor."""

    tilt_deg: float       # stage tilt angle (°), 0–30
    depth_um: float       # feature depth below coverslip (µm), 0.5–14.5
    NA: float             # objective numerical aperture, 0.2–0.9
    n2: float             # immersion medium refractive index, 1.22–1.46
    wavelength_um: float  # emission wavelength (µm), 0.42–0.72


class SliceScalingFactorLUT:
    """5-D linear interpolator over the slice scaling factor lookup table.

    Parameters
    ----------
    csv_path:
        Path to ``scaling_factor_lookup_table.csv``.  Defaults to the file
        bundled with the package.
    """

    def __init__(self, csv_path: Path = _LUT_PATH) -> None:
        lut = pd.read_csv(csv_path)

        # Sort in the order that defines the tensor axes
        sort_cols = ["tilt_deg", "depth_um", "n2", "wavelength_um", "NA"]
        lut_sorted = lut.sort_values(by=sort_cols)

        self.tilts: np.ndarray = np.sort(lut["tilt_deg"].unique())
        self.depths: np.ndarray = np.sort(lut["depth_um"].unique())
        self.n2s: np.ndarray = np.sort(lut["n2"].unique())
        self.wavelengths: np.ndarray = np.sort(lut["wavelength_um"].unique())
        self.nas: np.ndarray = np.sort(lut["NA"].unique())

        grid_shape = (
            len(self.tilts),
            len(self.depths),
            len(self.n2s),
            len(self.wavelengths),
            len(self.nas),
        )
        expected = np.prod(grid_shape)
        if len(lut_sorted) != expected:
            raise ValueError(
                f"LUT grid is incomplete: expected {expected} rows, found {len(lut_sorted)}. "
                "The CSV may be corrupt or missing entries."
            )

        zeta_grid = lut_sorted["zeta"].values.reshape(grid_shape)

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
