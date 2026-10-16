"""Experiment-global correlation configuration (Tier-1 of the persistence design).

``CorrelationConfig`` is the reusable home for how a correlation is set up: fit
settings, RI-correction parameters, and the spot-burn seeding toggle. It is held
as a peer field on
``AutoLamellaTaskProtocol`` — not an ``AutoLamellaTaskConfig``, since correlation
is a user step, not an automated microscope task. Defined here so the autolamella
model only holds an instance; imports no UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fibsem.structures import Configurable, Display, RangeLimit, param

# Mirror the UI constants as plain strings so this module never imports the UI.
FIT_METHODS = ("None", "Hole", "Gaussian")

# What deleting a FIB or FM fiducial does to its partner on the other side
# (FIB-1243): ask each time, remove the pair, or remove only that point.
DELETE_FIDUCIAL_ASK = "ask"
DELETE_FIDUCIAL_PAIR = "pair"
DELETE_FIDUCIAL_ONE = "one"
DELETE_FIDUCIAL_CHOICES = (
    DELETE_FIDUCIAL_ASK,
    DELETE_FIDUCIAL_PAIR,
    DELETE_FIDUCIAL_ONE,
)


# The values FitSettings.fm_fiducial_polarity takes; anything else reads as dark.
FM_FIDUCIAL_POLARITIES = ("dark", "bright")


@dataclass
class FitSettings:
    """Fit method per point-role, and channel *by name* (not index, so it can't
    pin the wrong dye across images). ``cutout`` is the ROI half-size — currently
    hardcoded per method (no UI), kept here so a future control has a home."""

    fib_method: str = "Hole"
    fm_fiducial_method: str = "None"
    fm_poi_method: str = "Gaussian"
    fm_fiducial_channel: Optional[str] = None
    fm_poi_channel: Optional[str] = None
    reflection_cutout: int = 2
    fluorescence_cutout: int = 5
    # How a burn looks in the FM fiducial channel: dark on Arctis reflection,
    # bright on METEOR reflection (FIB-1218). Dark is what the fit always
    # assumed, so a protocol without the key fits as before.
    fm_fiducial_polarity: str = "dark"
    # The FIB hole fit's window half-width (px), and the planes below and above
    # the clicked one the reflection fit searches. Both were fixed in the fits;
    # the fit panel's "Save as default" sets them (FIB-1218).
    fib_window: int = 15
    reflection_planes: Tuple[int, int] = (10, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fib_method": self.fib_method,
            "fm_fiducial_method": self.fm_fiducial_method,
            "fm_poi_method": self.fm_poi_method,
            "fm_fiducial_channel": self.fm_fiducial_channel,
            "fm_poi_channel": self.fm_poi_channel,
            "reflection_cutout": self.reflection_cutout,
            "fluorescence_cutout": self.fluorescence_cutout,
            "fm_fiducial_polarity": self.fm_fiducial_polarity,
            "fib_window": self.fib_window,
            "reflection_planes": list(self.reflection_planes),
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "FitSettings":
        d = data or {}
        default = FitSettings()
        return FitSettings(
            fib_method=d.get("fib_method", default.fib_method),
            fm_fiducial_method=d.get("fm_fiducial_method", default.fm_fiducial_method),
            fm_poi_method=d.get("fm_poi_method", default.fm_poi_method),
            fm_fiducial_channel=d.get("fm_fiducial_channel"),
            fm_poi_channel=d.get("fm_poi_channel"),
            reflection_cutout=d.get("reflection_cutout", default.reflection_cutout),
            fluorescence_cutout=d.get(
                "fluorescence_cutout", default.fluorescence_cutout
            ),
            fm_fiducial_polarity=(
                d.get("fm_fiducial_polarity")
                if d.get("fm_fiducial_polarity") in FM_FIDUCIAL_POLARITIES
                else default.fm_fiducial_polarity
            ),
            fib_window=_positive_int(d.get("fib_window"), default.fib_window),
            reflection_planes=_plane_pair(
                d.get("reflection_planes"), default.reflection_planes
            ),
        )


def _positive_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _plane_pair(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """Two positive plane counts, (below, above); anything else, the default."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and v > 0 for v in value)
    ):
        return (value[0], value[1])
    return default


@dataclass
class RISettings(Configurable):
    """RI depth-correction ζ inputs + mode. Mirrors ``ZetaParams`` (tilt / depth /
    NA / n2 / wavelength). ``wavelength_um`` and ``na`` may be seeded from channel
    metadata on FM load (FIB-277).

    The defaults are the one default set: the RI widget starts from them too.
    They are the configuration Perez et al. use for depth correction throughout:
    vitrified cytoplasm (n = 1.35) through the Arctis iFLM's NA 0.75 objective,
    for which the paper reports a scaling factor of 1.5
    (doi:10.64898/2026.05.11.724418, the source of the lookup table). Only NA and
    n2 move ζ appreciably; it is near-flat in tilt, depth and wavelength. These
    were NA 0.8 / n 1.4 here while the widget used 0.75 / 1.35, so a new protocol
    showed one set and computed ζ from the other (FIB-1234).
    """

    # The bounds are the lookup table's: zeta is only known inside them.
    tilt_deg: float = param(
        15.0,
        limits=RangeLimit(0.0, 30.0),
        doc="The angle between the FIB column and the sample surface.",
        display=Display("Milling Angle", unit="°", step=1.0, decimals=0),
    )
    depth_um: float = param(
        4.0,
        limits=RangeLimit(0.5, 14.5),
        doc="Estimated depth of the feature below the coverslip (has little effect below ~20 µm).",
        display=Display("Depth", unit="µm", step=0.5, decimals=1),
    )
    na: float = param(
        0.75,
        limits=RangeLimit(0.2, 0.9),
        doc="Numerical aperture of the objective lens.",
        display=Display("Numerical Aperture", step=0.1, decimals=2),
    )
    n2: float = param(
        1.35,
        limits=RangeLimit(1.22, 1.46),
        doc="Refractive index of the sample medium (n=1.0 is vacuum/air).",
        display=Display("Refractive Index", step=0.04, decimals=2),
    )
    wavelength_um: float = param(
        0.515,
        limits=RangeLimit(0.42, 0.72),
        doc="Excitation wavelength of the fluorescence channel.",
        display=Display("Wavelength (λ)", scale=1e3, unit="nm", step=10.0, decimals=0),
    )
    mode: str = "pre"  # "pre" (correct POI z before correlation) | "post"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tilt_deg": self.tilt_deg,
            "depth_um": self.depth_um,
            "na": self.na,
            "n2": self.n2,
            "wavelength_um": self.wavelength_um,
            "mode": self.mode,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "RISettings":
        d = data or {}
        default = RISettings()
        return RISettings(
            tilt_deg=d.get("tilt_deg", default.tilt_deg),
            depth_um=d.get("depth_um", default.depth_um),
            na=d.get("na", default.na),
            n2=d.get("n2", default.n2),
            wavelength_um=d.get("wavelength_um", default.wavelength_um),
            mode=d.get("mode", default.mode),
        )


@dataclass
class CorrelationConfig:
    """Experiment-global correlation config; inherited by every lamella in the run.

    Interpolation is deliberately absent. It was persisted here for the setup
    pre-dialog, which FIB-302 replaced with the in-window Interpolate action, and
    a stored preference nothing reads is how ``load_spot_burns`` came to be inert
    (FIB-319). ``from_dict`` reads named keys only, so the leftover
    ``"interpolation"`` in an existing protocol is ignored rather than fatal.
    """

    fit: FitSettings = field(default_factory=FitSettings)
    ri: RISettings = field(default_factory=RISettings)
    load_spot_burns: bool = True
    # Run the correlation again when the points settle, rather than on a press
    # (FIB-1020). Persisted because the alternative is a per-dialog tick: the
    # protocol editor builds a new correlation dialog for every lamella, so a
    # session-only preference is re-set once per lamella and reaches nobody.
    auto_rerun: bool = False
    # Resample an anisotropic FM stack to isotropic as it loads, off the GUI
    # thread, instead of waiting for the Interpolate action (FIB-1023). The fit
    # does not need it -- it converts z to xy pixels itself (FIB-881) -- so this
    # is for the display and for anything downstream that assumes isotropic
    # voxels. Off by default: it costs a rewrite of the volume.
    auto_interpolate: bool = False
    # Fiducials pair by their place in the FIB and FM lists, so deleting one
    # side shifts every later pair (FIB-1243). "ask" until someone ticks "Don't
    # ask again", which stores their choice here; the Method panel sets it back.
    delete_fiducial: str = DELETE_FIDUCIAL_ASK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit": self.fit.to_dict(),
            "ri": self.ri.to_dict(),
            "load_spot_burns": self.load_spot_burns,
            "auto_rerun": self.auto_rerun,
            "auto_interpolate": self.auto_interpolate,
            "delete_fiducial": self.delete_fiducial,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "CorrelationConfig":
        d = data or {}
        return CorrelationConfig(
            fit=FitSettings.from_dict(d.get("fit")),
            ri=RISettings.from_dict(d.get("ri")),
            load_spot_burns=d.get("load_spot_burns", True),
            auto_rerun=d.get("auto_rerun", False),
            auto_interpolate=d.get("auto_interpolate", False),
            delete_fiducial=(
                d.get("delete_fiducial")
                if d.get("delete_fiducial") in DELETE_FIDUCIAL_CHOICES
                else DELETE_FIDUCIAL_ASK
            ),
        )
