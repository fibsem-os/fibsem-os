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

import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

_LUT_FILENAME = "table_refractive_index_lookup.csv"
_LUT_PATH = Path(__file__).parent / "data" / _LUT_FILENAME
_LUT_URL = (
    "https://github.com/fibsem-os/fibsem-os/releases/download/data-0.1.0/"
    + _LUT_FILENAME
)
_LUT_COLUMNS = ("tilt_deg", "depth_um", "n2", "wavelength_um", "NA", "zeta")

# The download runs on the GUI thread when the correlation window opens, and
# microscope PCs are commonly offline or behind a proxy that blackholes the
# connection rather than refusing it — without this the window hangs until the
# OS gives up, which on some platforms is over a minute.
_LUT_DOWNLOAD_TIMEOUT_S = 10.0

# One attempt per process. A machine that cannot reach the release will not
# start being able to mid-session, and the cost of pretending otherwise is the
# stall above on every single open of the window.
_download_attempted = False


def _download_lut() -> None:
    """Fetch the LUT CSV from the data release, atomically.

    Writes to a temporary file beside the destination and renames it into place.
    A transfer that dies half way must never leave a truncated CSV at the real
    path: that file *exists*, so nothing re-fetches it, and every lookup fails
    against it for the life of the install (FIB-592).
    """
    _LUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading the refractive-index LUT from {_LUT_URL}")
    tmp_path: Optional[Path] = None
    try:
        with urllib.request.urlopen(
            _LUT_URL, timeout=_LUT_DOWNLOAD_TIMEOUT_S
        ) as response:
            with tempfile.NamedTemporaryFile(
                dir=str(_LUT_PATH.parent),
                prefix=_LUT_PATH.name + ".",
                suffix=".part",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                shutil.copyfileobj(response, tmp)
        os.replace(str(tmp_path), str(_LUT_PATH))  # atomic within the directory
        tmp_path = None
        logging.info(f"Refractive-index LUT saved to {_LUT_PATH}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _ensure_lut() -> None:
    """Make sure a *usable* LUT is on disk, downloading one if it is not.

    Also the re-fetch path for a file that is present but will not parse — an
    install that picked up a truncated CSV before the atomic write above would
    otherwise stay broken forever, with no way out but deleting the file by hand.

    Raises whatever the download raises, plus RuntimeError if what arrived still
    does not parse. Callers on the GUI thread are expected to catch and report.
    """
    global _download_attempted
    if lut_error() is None:
        return
    if _download_attempted:
        return
    _download_attempted = True
    _download_lut()
    reset_lut_cache()  # re-validate what we just wrote rather than assuming
    error = lut_error()
    if error is not None:
        raise RuntimeError(f"downloaded refractive-index LUT is unusable: {error}")


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

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        # Resolved here, not as a default argument: a default binds _LUT_PATH at
        # import, so this class would go on reading the original file after
        # anything repointed the module — which is every test, and would be a
        # silent disagreement with lut_error() rather than a failure.
        csv_path = Path(csv_path) if csv_path is not None else _LUT_PATH
        # Deliberately does not download: fetching is _ensure_lut's job, so that
        # merely asking whether the LUT works cannot reach the network.
        lut = pd.read_csv(csv_path)

        missing = [c for c in _LUT_COLUMNS if c not in lut.columns]
        if missing:
            # A proxy error page or an HTML 404 saved under the CSV's name parses
            # as *something*; say what it is rather than dying on a KeyError.
            raise ValueError(
                f"{csv_path} is not the LUT: missing column(s) {', '.join(missing)}"
            )

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
_lut_error: Optional[str] = None
_lut_checked = False


def _load_lut_once() -> None:
    """Fill the cache with either the LUT or the reason there isn't one."""
    global _lut, _lut_error, _lut_checked
    if _lut_checked:
        return
    _lut_checked = True
    if not _LUT_PATH.exists():
        _lut_error = f"file not found at {_LUT_PATH}"
        return
    try:
        _lut = SliceScalingFactorLUT(_LUT_PATH)
    except Exception as exc:
        _lut_error = f"file at {_LUT_PATH} could not be read ({exc})"
        logging.warning(f"Refractive-index LUT unusable: {_lut_error}")


def lut_error() -> Optional[str]:
    """Why the LUT cannot be used, or None when it can. Never downloads.

    "The file exists" is not "the file works". A transfer that died half way, or
    a proxy error page saved under the CSV's name, leaves a file that every
    lookup raises against — and testing existence let exactly that state present
    as a working calculator: five live spinboxes, no warning, and a correction
    factor that silently never moved (FIB-592).

    The verdict is cached. Neither failure mode fixes itself within a session,
    and the alternative is re-parsing a 9 MB CSV on every spinbox keystroke.
    """
    _load_lut_once()
    return _lut_error


def reset_lut_cache() -> None:
    """Forget the cached LUT and its verdict — after a download, or in tests."""
    global _lut, _lut_error, _lut_checked
    _lut, _lut_error, _lut_checked = None, None, False


def get_lut() -> SliceScalingFactorLUT:
    """Return the module-level :class:`SliceScalingFactorLUT` singleton.

    Loaded from disk on the first call and cached. Raises RuntimeError when the
    LUT is missing or unreadable — the failure is cached too, so a caller in a
    spinbox callback is not re-parsing a broken 9 MB CSV on every keystroke.
    """
    _load_lut_once()
    if _lut is None:
        raise RuntimeError(f"refractive-index LUT unavailable: {_lut_error}")
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
