from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional
import json
import time

import numpy as np
from fibsem.structures import FibsemImage, Point
from fibsem.fm.structures import FluorescenceImage


class PointType(StrEnum):
    FIB = "FIB"
    FM = "FM"
    POI = "POI"
    SURFACE = "SURFACE"


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

    def to_dict(self):
        return {"point": self.point.to_dict(), "point_type": self.point_type.value}

    @staticmethod
    def from_dict(data: dict) -> Coordinate:
        point = PointXYZ.from_dict(data["point"])
        point_type = PointType(data["point_type"])
        return Coordinate(point=point, point_type=point_type)


@dataclass
class CorrelationInputData:
    fib_image: Optional[FibsemImage] = None
    fm_image: Optional[FluorescenceImage] = None
    fib_coordinates: list[Coordinate] = field(default_factory=list)
    fm_coordinates: list[Coordinate] = field(default_factory=list)
    poi_coordinates: list[Coordinate] = field(default_factory=list)
    surface_coordinate: Optional[Coordinate] = None
    method: str = "multi-point"

    def to_dict(self):
        return {
            "fib_coordinates": [coord.to_dict() for coord in self.fib_coordinates],
            "fm_coordinates": [coord.to_dict() for coord in self.fm_coordinates],
            "poi_coordinates": [coord.to_dict() for coord in self.poi_coordinates],
            "surface_coordinate": self.surface_coordinate.to_dict()
            if self.surface_coordinate
            else None,
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
            return None
        return self.fib_image.metadata.pixel_size.x  # type: ignore[union-attr]

    @property
    def fib_image_shape(self) -> Optional[tuple[int, int]]:
        if self.fib_image is None or self.fib_image.data is None:
            return None
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
        fib_coordinates = [
            Coordinate.from_dict(coord) for coord in data["fib_coordinates"]
        ]
        fm_coordinates = [
            Coordinate.from_dict(coord) for coord in data["fm_coordinates"]
        ]
        poi_coordinates = [
            Coordinate.from_dict(coord) for coord in data["poi_coordinates"]
        ]
        surface_coordinate = (
            Coordinate.from_dict(data["surface_coordinate"])
            if data["surface_coordinate"]
            else None
        )
        return CorrelationInputData(
            fib_coordinates=fib_coordinates,
            fm_coordinates=fm_coordinates,
            poi_coordinates=poi_coordinates,
            surface_coordinate=surface_coordinate,
            method=data["method"],
        )

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(filename: str) -> CorrelationInputData:

        with open(filename, "r") as f:
            data = json.load(f)
            return CorrelationInputData.from_dict(data)


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
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "poi": [p.to_dict() for p in self.poi],
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
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict) -> CorrelationResult:
        return CorrelationResult(
            poi=[CorrelationPointOfInterest.from_dict(p) for p in data.get("poi", [])],
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
            updated_at=data.get("updated_at", time.time()),
        )

    def apply_refractive_index_correction(
        self,
        correction_factor: float,
    ) -> None:
        """Apply depth correction to the first POI (index 0) in-place.

        Falls back to self.input_data for surface_y / fib_shape / pixel_size
        when not provided explicitly.
        """
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
        corrected_y = surface_y + (poi0.image_px.y - surface_y) * correction_factor
        poi0.image_px.y = corrected_y
        if fib_shape is not None and pixel_size is not None:
            cy = fib_shape[0] / 2.0
            poi0.px.y = -(corrected_y - cy)
            poi0.px_m.y = poi0.px.y * pixel_size
        self.refractive_index_correction_factor = correction_factor
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

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(filename: str) -> CorrelationResult:
        with open(filename, "r") as f:
            return CorrelationResult.from_dict(json.load(f))
