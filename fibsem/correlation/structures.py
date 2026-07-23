from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Optional
import json
import logging
import time

import numpy as np
from fibsem.structures import FibsemImage, Point
from fibsem.fm.structures import FluorescenceImage


class PointType(Enum):
    FIB = "FIB"
    FM = "FM"
    POI = "POI"
    SURFACE = "SURFACE"          # sample surface in the FIB image (post-correlation RI correction)
    SURFACE_FM = "FM-SURFACE"    # sample surface in the FM volume (pre-correlation RI correction)


@dataclass
class PointXYZ:
    x: float = field(default=0.0)
    y: float = field(default=0.0)
    z: float = field(default=0.0)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data: dict) -> PointXYZ:
        return PointXYZ(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Coordinate:
    point: PointXYZ = field(default_factory=PointXYZ)
    point_type: PointType = field(default=PointType.FIB)
    fitted: bool = False  # True when this position came from an accepted auto-fit

    def to_dict(self):
        return {
            "point": self.point.to_dict(),
            "point_type": self.point_type.value,
            "fitted": self.fitted,
        }

    @staticmethod
    def from_dict(data: dict) -> Coordinate:
        point = PointXYZ.from_dict(data["point"])
        point_type = PointType(data["point_type"])
        return Coordinate(
            point=point, point_type=point_type, fitted=data.get("fitted", False)
        )


def scale_about_surface(value, surface, correction_factor):
    """Scale a depth-like value about a surface reference.

    ``surface + (value - surface) * correction_factor`` — the single source of
    the refractive-index depth-scaling formula. Signed (works regardless of
    axis direction) and unit-free; accepts scalars or numpy arrays.
    """
    return surface + (value - surface) * correction_factor


def apply_z_surface_correction(
    poi_coords: np.ndarray, surface_z: float, correction_factor: float
) -> np.ndarray:
    """Scale POI z about the FM surface z (pre-correlation RI correction).

    corrected_z = surface_z + (z - surface_z) * correction_factor

    The scaling is signed, so it is independent of the stack z direction, and
    unit-free, so it is valid in voxel indices for a uniform z-step.

    Args:
        poi_coords: (N, 3) array of POI coordinates (x, y, z) in FM voxels.
        surface_z: z of the surface point in the FM volume (voxels).
        correction_factor: depth scaling factor (zeta).

    Returns:
        A corrected copy; the input array is not modified.
    """
    corrected = np.array(poi_coords, dtype=np.float32, copy=True)
    if corrected.size:
        corrected[:, 2] = scale_about_surface(
            corrected[:, 2], surface_z, correction_factor
        )
    return corrected


@dataclass
class CorrelationInputData:
    fib_image: Optional[FibsemImage] = None
    fm_image: Optional[FluorescenceImage] = None
    fib_coordinates: list[Coordinate] = field(default_factory=list)
    fm_coordinates: list[Coordinate] = field(default_factory=list)
    poi_coordinates: list[Coordinate] = field(default_factory=list)
    surface_coordinate: Optional[Coordinate] = None
    # Surface point in the FM volume; mutually exclusive with surface_coordinate.
    # When set together with ri_pre_correction_factor, the POI z is corrected
    # before the correlation transform is applied (see run_correlation_from_data).
    fm_surface_coordinate: Optional[Coordinate] = None
    ri_pre_correction_factor: Optional[float] = None
    method: str = "multi-point"
    # Fallbacks restored from serialized files, so a result loaded from JSON
    # (images absent) can still convert corrected pixels to px / px_m
    stored_fib_image_shape: Optional[tuple] = None
    stored_fib_image_pixel_size: Optional[float] = None

    def to_dict(self):
        return {
            "fib_coordinates": [coord.to_dict() for coord in self.fib_coordinates],
            "fm_coordinates": [coord.to_dict() for coord in self.fm_coordinates],
            "poi_coordinates": [coord.to_dict() for coord in self.poi_coordinates],
            "surface_coordinate": self.surface_coordinate.to_dict()
            if self.surface_coordinate
            else None,
            "fm_surface_coordinate": self.fm_surface_coordinate.to_dict()
            if self.fm_surface_coordinate
            else None,
            "ri_pre_correction_factor": self.ri_pre_correction_factor,
            "fm_image_shape": self.fm_image_shape,
            "fib_image_shape": self.fib_image_shape,
            "fib_image_pixel_size": self.fib_image_pixel_size,
            "fib_image_filename": self.fib_image_filename,
            "fm_image_filename": self.fm_image_filename,
            "method": self.method,
        }

    @property
    def fib_image_pixel_size(self) -> Optional[float]:
        if self.fib_image is None:
            return self.stored_fib_image_pixel_size
        return self.fib_image.metadata.pixel_size.x  # type: ignore[union-attr]

    @property
    def fib_image_shape(self) -> Optional[tuple[int, int]]:
        if self.fib_image is None or self.fib_image.data is None:
            return self.stored_fib_image_shape
        return self.fib_image.data.shape

    @property
    def fm_image_shape(self) -> Optional[tuple[int, int, int, int]]:
        if self.fm_image is None or self.fm_image.data is None:
            return None
        return self.fm_image.data.shape

    @property
    def fib_image_filename(self) -> Optional[str]:
        return (
            self.fib_image.metadata.image_settings.filename if self.fib_image else None
        )

    @property
    def fm_image_filename(self) -> Optional[str]:
        return self.fm_image.metadata.filename if self.fm_image else None

    @staticmethod
    def from_dict(data: dict) -> CorrelationInputData:
        # Reject rather than default. to_dict always writes "fib_coordinates",
        # even when empty, so its absence means this is not a coordinates file —
        # and defaulting to [] would quietly return an empty CorrelationInputData
        # that the caller then applies, wiping the coordinates it replaced.
        if "fib_coordinates" not in data:
            hint = (
                " This looks like a correlation result — load it with"
                " Load Correlation Result instead."
                if ("poi" in data or "input_data" in data)
                else ""
            )
            raise ValueError(
                "Not a correlation coordinates file: no 'fib_coordinates' key." + hint
            )

        fib_coordinates = [
            Coordinate.from_dict(coord) for coord in data["fib_coordinates"]
        ]
        # Sibling keys are written by the same writer, so tolerate their absence
        # in a hand-edited file now that the file type itself is established.
        fm_coordinates = [
            Coordinate.from_dict(coord) for coord in data.get("fm_coordinates", [])
        ]
        poi_coordinates = [
            Coordinate.from_dict(coord) for coord in data.get("poi_coordinates", [])
        ]
        surface_coordinate = (
            Coordinate.from_dict(data["surface_coordinate"])
            if data.get("surface_coordinate")
            else None
        )
        fm_surface_coordinate = (
            Coordinate.from_dict(data["fm_surface_coordinate"])
            if data.get("fm_surface_coordinate")
            else None
        )
        stored_shape = data.get("fib_image_shape")
        return CorrelationInputData(
            fib_coordinates=fib_coordinates,
            fm_coordinates=fm_coordinates,
            poi_coordinates=poi_coordinates,
            surface_coordinate=surface_coordinate,
            fm_surface_coordinate=fm_surface_coordinate,
            ri_pre_correction_factor=data.get("ri_pre_correction_factor"),
            # Has a dataclass default and sits between two .get() calls; the hard
            # subscript was the second key that could throw out of this function.
            method=data.get("method", "multi-point"),
            stored_fib_image_shape=tuple(stored_shape) if stored_shape else None,
            stored_fib_image_pixel_size=data.get("fib_image_pixel_size"),
        )

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(filename: str) -> CorrelationInputData:

        with open(filename, "r") as f:
            data = json.load(f)
            return CorrelationInputData.from_dict(data)


def _transform_inputs(data: CorrelationInputData) -> dict:
    """The subset of the inputs that can change the fitted transform.

    Used to decide whether a result still describes the current points. Scoping
    matters both ways: too narrow and a real change slips through; too broad and
    an edit that cannot move the answer marks a good result stale, at which point
    the signal stops meaning anything and users learn to ignore it.
    """

    def positions(coords: list[Coordinate]) -> list:
        return [c.point.to_dict() for c in coords]

    def position(coord: Optional[Coordinate]) -> Optional[dict]:
        return coord.point.to_dict() if coord is not None else None

    return {
        "fib": positions(data.fib_coordinates),
        "fm": positions(data.fm_coordinates),
        "poi": positions(data.poi_coordinates),
        "surface": position(data.surface_coordinate),
        "fm_surface": position(data.fm_surface_coordinate),
        # scales POI z about the FM surface *before* the fit, so a change here
        # genuinely changes the answer (see run_correlation_from_data).
        "ri_pre_correction_factor": data.ri_pre_correction_factor,
    }


@dataclass
class CorrelationPointOfInterest:
    """A single POI reprojected into the FIB image, in multiple coordinate systems."""

    image_px: Point = field(default_factory=Point)  # pixel coordinates in FIB image
    px: Point = field(
        default_factory=Point
    )  # microscope image coords (pixels, centred)
    px_m: Point = field(default_factory=Point)  # microscope image coords (metres)

    def to_dict(self) -> dict:
        return {
            "image_px": self.image_px.to_dict(),
            "px": self.px.to_dict(),
            "px_m": self.px_m.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> CorrelationPointOfInterest:
        return CorrelationPointOfInterest(
            image_px=Point.from_dict(data["image_px"]),
            px=Point.from_dict(data["px"]),
            px_m=Point.from_dict(data["px_m"]),
        )


@dataclass
class CorrelationResult:
    poi: list[CorrelationPointOfInterest] = field(default_factory=list)
    # Where the POIs would land WITHOUT the pre-correlation RI correction,
    # reprojected through the same fitted transform. Only populated when the
    # pre-correction was applied; used as a "ghost" overlay for comparison.
    poi_uncorrected: list[CorrelationPointOfInterest] = field(default_factory=list)

    # Transformation
    scale: float = 0.0
    rotation_eulers: list = field(default_factory=list)  # degrees, x-convention
    rotation_quaternion: list = field(default_factory=list)
    translation: list = field(default_factory=list)  # around rotation center (zero)
    translation_custom: list = field(
        default_factory=list
    )  # around rotation center (custom)

    # Error
    rms_error: float = 0.0
    mean_absolute_error: list = field(default_factory=list)  # [err_x, err_y]
    delta_2d: list[Point] = field(
        default_factory=list
    )  # per-marker reprojection error (pixels)
    reprojected_3d: list[PointXYZ] = field(
        default_factory=list
    )  # 3D markers reprojected into 2D image (pixels)

    # Provenance
    input_data: Optional[CorrelationInputData] = None
    refractive_index_correction_factor: Optional[float] = None
    # "post": factor applied to the correlated POI in FIB image space
    # "pre":  factor applied to the input POI z (FM space) before correlation
    refractive_index_correction_mode: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "poi": [p.to_dict() for p in self.poi],
            "poi_uncorrected": [p.to_dict() for p in self.poi_uncorrected],
            "scale": self.scale,
            "rotation_eulers": self.rotation_eulers,
            "rotation_quaternion": self.rotation_quaternion,
            "translation": self.translation,
            "translation_custom": self.translation_custom,
            "rms_error": self.rms_error,
            "mean_absolute_error": self.mean_absolute_error,
            "delta_2d": [p.to_dict() for p in self.delta_2d],
            "reprojected_3d": [p.to_dict() for p in self.reprojected_3d],
            "input_data": self.input_data.to_dict() if self.input_data else None,
            "refractive_index_correction_factor": self.refractive_index_correction_factor,
            "refractive_index_correction_mode": self.refractive_index_correction_mode,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict) -> CorrelationResult:
        # The mirror of CorrelationInputData.from_dict: the two auto-saved files
        # sit side by side with near-identical names, so guard both directions.
        # Loading coordinates here would otherwise yield a result with no POI and
        # rms 0 — not a crash, but a convincing-looking empty one.
        if "poi" not in data and "fib_coordinates" in data:
            raise ValueError(
                "Not a correlation result file: no 'poi' key. This looks like a"
                " coordinates file — load it with Load Coordinates instead."
            )
        return CorrelationResult(
            poi=[CorrelationPointOfInterest.from_dict(p) for p in data.get("poi", [])],
            poi_uncorrected=[
                CorrelationPointOfInterest.from_dict(p)
                for p in data.get("poi_uncorrected", [])
            ],
            scale=data.get("scale", 0.0),
            rotation_eulers=data.get("rotation_eulers", []),
            rotation_quaternion=data.get("rotation_quaternion", []),
            translation=data.get("translation", []),
            translation_custom=data.get("translation_custom", []),
            rms_error=data.get("rms_error", 0.0),
            mean_absolute_error=data.get("mean_absolute_error", []),
            delta_2d=[Point.from_dict(p) for p in data.get("delta_2d", [])],
            reprojected_3d=[
                PointXYZ.from_dict(p) for p in data.get("reprojected_3d", [])
            ],
            input_data=CorrelationInputData.from_dict(data["input_data"])
            if data.get("input_data")
            else None,
            refractive_index_correction_factor=data.get(
                "refractive_index_correction_factor"
            ),
            refractive_index_correction_mode=data.get(
                "refractive_index_correction_mode"
            ),
            updated_at=data.get("updated_at", time.time()),
        )

    def apply_refractive_index_correction(
        self,
        correction_factor: float,
    ) -> None:
        """Apply depth correction to the first POI (index 0) in-place.

        Reads surface_y / fib_shape / pixel_size from self.input_data. The
        uncorrected position of POI 1 is snapshotted into ``poi_uncorrected``
        (ghost overlay) before the correction is applied.

        Raises ValueError if a correction (pre or post) has already been
        applied — re-applying would compound the depth scaling.
        """
        if self.refractive_index_correction_factor is not None:
            raise ValueError(
                "A refractive-index correction has already been applied to this "
                f"result ({self.refractive_index_correction_mode or 'post'}, factor "
                f"{self.refractive_index_correction_factor}); re-applying would "
                "compound it. Run the correlation again first."
            )
        if self.input_data is None:
            raise ValueError(
                "input_data is required to apply refractive index correction"
            )

        if not self.poi:
            raise ValueError("No POI available to apply refractive index correction")

        if self.input_data.surface_coordinate is None:
            raise ValueError(
                "surface_coordinate is required in input_data to apply refractive index correction"
            )
        if self.input_data.surface_coordinate.point.y is None:
            raise ValueError(
                "surface_coordinate.point.y is required in input_data to apply refractive index correction"
            )

        surface_y = self.input_data.surface_coordinate.point.y
        fib_shape = self.input_data.fib_image_shape
        pixel_size = self.input_data.fib_image_pixel_size
        poi0 = self.poi[0]
        # Snapshot the uncorrected position for the ghost overlay
        self.poi_uncorrected = [
            CorrelationPointOfInterest(
                image_px=Point(poi0.image_px.x, poi0.image_px.y),
                px=Point(poi0.px.x, poi0.px.y),
                px_m=Point(poi0.px_m.x, poi0.px_m.y),
            )
        ]
        corrected_y = scale_about_surface(poi0.image_px.y, surface_y, correction_factor)
        poi0.image_px.y = corrected_y
        if fib_shape is not None and pixel_size is not None:
            cy = fib_shape[0] / 2.0
            poi0.px.y = -(corrected_y - cy)
            poi0.px_m.y = poi0.px.y * pixel_size
        else:
            logging.warning(
                "RI correction: fib image shape/pixel size unavailable — "
                "poi.px / poi.px_m were NOT updated (only image_px)."
            )
        self.refractive_index_correction_factor = correction_factor
        self.refractive_index_correction_mode = "post"
        self.updated_at = time.time()

    def to_input_dataframe(self):
        """Return input coordinates (FIB, FM, POI, SURFACE) as a DataFrame."""
        import pandas as pd

        if self.input_data is None:
            return pd.DataFrame(columns=["type", "idx", "x_px", "y_px", "z_px"])
        rows = []
        for lst, label in [
            (self.input_data.fib_coordinates, "FIB"),
            (self.input_data.fm_coordinates, "FM"),
            (self.input_data.poi_coordinates, "POI"),
        ]:
            for i, c in enumerate(lst):
                rows.append(
                    {
                        "type": label,
                        "idx": i,
                        "x_px": c.point.x,
                        "y_px": c.point.y,
                        "z_px": c.point.z,
                    }
                )
        if self.input_data.surface_coordinate is not None:
            sc = self.input_data.surface_coordinate
            rows.append(
                {
                    "type": "SURFACE",
                    "idx": 0,
                    "x_px": sc.point.x,
                    "y_px": sc.point.y,
                    "z_px": sc.point.z,
                }
            )
        if self.input_data.fm_surface_coordinate is not None:
            sc = self.input_data.fm_surface_coordinate
            rows.append(
                {
                    "type": PointType.SURFACE_FM.value,
                    "idx": 0,
                    "x_px": sc.point.x,
                    "y_px": sc.point.y,
                    "z_px": sc.point.z,
                }
            )
        return pd.DataFrame(rows)

    def to_result_dataframe(self):
        """Return per-marker reprojection error and 3-D reprojected positions as a DataFrame."""
        import pandas as pd

        n = max(len(self.delta_2d), len(self.reprojected_3d))
        rows = []
        for i in range(n):
            row: dict = {"marker": i}
            if i < len(self.delta_2d):
                row["delta_x_px"] = self.delta_2d[i].x
                row["delta_y_px"] = self.delta_2d[i].y
            if i < len(self.reprojected_3d):
                row["reprojected_x_px"] = self.reprojected_3d[i].x
                row["reprojected_y_px"] = self.reprojected_3d[i].y
                row["reprojected_z_px"] = self.reprojected_3d[i].z
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, filename: str) -> None:
        """Export input coordinates and reprojection error to a CSV file."""
        df_input = self.to_input_dataframe()
        df_result = self.to_result_dataframe()
        with open(filename, "w") as f:
            f.write("# Input Coordinates\n")
            df_input.to_csv(f, index=False)
            f.write("\n# Marker Reprojection Error\n")
            df_result.to_csv(f, index=False)

    def matches_inputs(self, current: CorrelationInputData) -> bool:
        """Whether this result was computed from ``current``'s coordinates.

        ``input_data`` is a snapshot of the points the transform was fitted to,
        which makes staleness *derivable* — no flag anyone has to remember to
        set. Compares only what the transform consumes (see
        :func:`_transform_inputs`); notably **excluded**:

          * ``fitted`` — provenance on a Coordinate, not a position;
          * ``method`` — not read by ``run_correlation_from_data``;
          * anything image-derived — a deserialized result carries no images, so
            e.g. ``fm_image_shape`` is None on this side and populated on the
            live side, which would report every reloaded result as stale.

        A result with no snapshot can't be shown to match, so it reports False.
        """
        if self.input_data is None:
            return False
        return _transform_inputs(self.input_data) == _transform_inputs(current)

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(filename: str) -> CorrelationResult:
        with open(filename, "r") as f:
            return CorrelationResult.from_dict(json.load(f))
