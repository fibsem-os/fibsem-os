"""Experiment-global correlation configuration (Tier-1 of the persistence design).

``CorrelationConfig`` is the reusable home for how a correlation is set up: fit
settings, RI-correction parameters, an interpolation preference, and the
spot-burn seeding toggle. It is held as a peer field on
``AutoLamellaTaskProtocol`` — not an ``AutoLamellaTaskConfig``, since correlation
is a user step, not an automated microscope task. Defined here so the autolamella
model only holds an instance; imports no UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Mirror the UI constants as plain strings so this module never imports the UI.
FIT_METHODS = ("None", "Hole", "Gaussian")
INTERPOLATION_METHODS = ("linear", "cubic")


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fib_method": self.fib_method,
            "fm_fiducial_method": self.fm_fiducial_method,
            "fm_poi_method": self.fm_poi_method,
            "fm_fiducial_channel": self.fm_fiducial_channel,
            "fm_poi_channel": self.fm_poi_channel,
            "reflection_cutout": self.reflection_cutout,
            "fluorescence_cutout": self.fluorescence_cutout,
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
            fluorescence_cutout=d.get("fluorescence_cutout", default.fluorescence_cutout),
        )


@dataclass
class RISettings:
    """RI depth-correction ζ inputs + mode. Mirrors ``ZetaParams`` (tilt / depth /
    NA / n2 / wavelength); defaults match ``DEFAULT_ZETA_PARAMS``. ``wavelength_um``
    and ``na`` may be seeded from channel metadata on FM load (FIB-277)."""

    tilt_deg: float = 15.0
    depth_um: float = 4.0
    na: float = 0.8
    n2: float = 1.4
    wavelength_um: float = 0.515
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
class InterpolationSettings:
    """FM z-interpolation preference. ``isotropic`` targets XY pixel size, else
    ``target_z_nm``."""

    enabled: bool = False
    isotropic: bool = True
    target_z_nm: Optional[float] = None
    method: str = "linear"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "isotropic": self.isotropic,
            "target_z_nm": self.target_z_nm,
            "method": self.method,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "InterpolationSettings":
        d = data or {}
        default = InterpolationSettings()
        return InterpolationSettings(
            enabled=d.get("enabled", default.enabled),
            isotropic=d.get("isotropic", default.isotropic),
            target_z_nm=d.get("target_z_nm"),
            method=d.get("method", default.method),
        )


@dataclass
class CorrelationConfig:
    """Experiment-global correlation config; inherited by every lamella in the run."""

    fit: FitSettings = field(default_factory=FitSettings)
    ri: RISettings = field(default_factory=RISettings)
    interpolation: InterpolationSettings = field(default_factory=InterpolationSettings)
    load_spot_burns: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit": self.fit.to_dict(),
            "ri": self.ri.to_dict(),
            "interpolation": self.interpolation.to_dict(),
            "load_spot_burns": self.load_spot_burns,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "CorrelationConfig":
        d = data or {}
        return CorrelationConfig(
            fit=FitSettings.from_dict(d.get("fit")),
            ri=RISettings.from_dict(d.get("ri")),
            interpolation=InterpolationSettings.from_dict(d.get("interpolation")),
            load_spot_burns=d.get("load_spot_burns", True),
        )
