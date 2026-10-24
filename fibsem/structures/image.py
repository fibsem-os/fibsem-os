"""FibsemImage and its metadata."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import tifffile as tff
from numpy.typing import NDArray

from fibsem import image_format
from fibsem.codec import AwareTimestamp, decode, encode, saved
from fibsem.codec.upgrades import image_metadata as metadata_upgrades
from fibsem.config import METADATA_VERSION
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.structures.system import (
    FibsemExperimentRef,
    FibsemHardwareGeometry,
    FibsemUser,
    SystemInfo,
)
from fibsem.util.timestamps import acquisition_datetime_of


@saved(keys={"image_settings": "image"})
@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage.

    Three kinds of claim live here, and they are easy to mistake for each other.
    Anything added should be placed deliberately in one of them:

    **Provenance** -- what produced this image. ``system_info`` (which instrument),
    ``user`` (who), ``experiment`` (which run). Constant for a run. The same
    question at a finer grain -- which item (a lamella, or a grid), which task --
    is answered by ``experiment.item_name`` and ``experiment.task_name``.
    That varies *within* a run, which changes the mechanism that writes it, but not
    the kind of fact it is.

    **Configuration** -- what the instrument *is*. ``hardware_geometry``: the fixed
    physical arrangement a projection needs.

    **Observation** -- what the instrument was doing. ``microscope_state``,
    ``pixel_size``. Measured at acquisition.

    **Request** -- what was asked for. ``image_settings``. Note this is *intent*,
    not outcome: if autocontrast ran, or the requested hfw was clamped, nothing
    here records that the request was not honoured.

    The rule for provenance is one writer per fact, denormalised deliberately at
    this boundary: an image is embedded in a file that may be copied, emailed or
    read years later, so it carries enough to identify its source without the
    surrounding directory. Whether an embedded copy *defers* to a richer record is
    a separate question, answered per-structure -- see ``FibsemExperimentRef``
    (defers) and ``FibsemUser`` (does not, because there is nothing to defer to).
    """

    image_settings: ImageSettings
    pixel_size: Point
    microscope_state: MicroscopeState
    # Both replaced `system: Optional[SystemSettings]` in v6. Optional
    # because a file written before v6 may carry neither in a recoverable form, and
    # because a FibsemImage can be constructed without a microscope at all.
    system_info: Optional[SystemInfo] = None
    hardware_geometry: Optional[FibsemHardwareGeometry] = None
    version: str = METADATA_VERSION
    user: FibsemUser = field(default_factory=lambda: FibsemUser())
    experiment: FibsemExperimentRef = field(
        default_factory=lambda: FibsemExperimentRef()
    )
    # When the image was acquired: an aware datetime, written as ISO 8601 with its
    # UTC offset, `2026-09-13T21:19:40.974286-06:00`. An observation, like
    # microscope_state, but a separate field: the state's timestamp says when the
    # state was read, which is not the same moment. `Beam.acquire` stamps it just
    # before the driver runs; a driver with a vendor time sets its own. None on
    # files from before v11 and on images built rather than acquired -- read the
    # time through `acquisition_datetime_of`, which falls back for those.
    acquisition_datetime: Optional[AwareTimestamp] = None
    # Provenance: the device in each role the acquiring beam had filled (v12), by
    # role name, such as the external scanner that made the frame: its ``name``,
    # ``class``, ``driver``, its parameters' last known ``values`` and its
    # ``facts``. Empty when the beam imaged through the vendor's own scan;
    # ``microscope_state`` still carries the vendor detector either way.
    roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def beam_type(self) -> BeamType:
        return self.image_settings.beam_type

    @property
    def stage_position(self) -> FibsemStagePosition:
        return self.microscope_state.stage_position

    @property
    def acquisition_date(self) -> Optional[datetime]:
        """When the image was acquired: aware, with the offset it was recorded with,
        when the file says which zone.

        Naive when it does not: a ThermoFisher image from before v11 records the
        instrument PC's clock time alone. See `acquisition_datetime_of`.
        """
        return acquisition_datetime_of(self)

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(settings: dict) -> "FibsemImageMetadata":
        """Read metadata written by any version (`fibsem.codec.upgrades.image_metadata`)."""
        return decode(FibsemImageMetadata, metadata_upgrades.upgrade(settings))

    @staticmethod
    def read(path: str) -> Optional["FibsemImageMetadata"]:
        """A saved beam image's metadata, without reading its pixels; None for
        a file that records none (an FM stack's OME, a plain TIFF)."""
        try:
            with tff.TiffFile(path) as tiff:
                raw = image_format.read_metadata_dict(tiff)
            return FibsemImageMetadata.from_dict(raw) if raw is not None else None
        except Exception:
            logging.debug(f"Could not read the metadata of {path}", exc_info=True)
            return None


@dataclass
class ImageStats:
    """Histogram statistics for a FibsemImage.

    All intensity values are normalised to [0, 1] relative to the dtype maximum.
    """

    mean: float
    std: float
    p01: float  # 1st percentile
    p99: float  # 99th percentile
    saturation_lo: float  # fraction of pixels at dtype min
    saturation_hi: float  # fraction of pixels at dtype max
    contrast_ratio: float  # coefficient of variation: std / mean
    range_utilisation: float  # p99 - p01
    median: float  # normalised median (robust alternative to mean)
    snr: float  # mean / std
    entropy: float  # Shannon entropy of the normalised histogram (bits)

    def __str__(self) -> str:
        return (
            f"mean={self.mean:.3f}, median={self.median:.3f}, std={self.std:.3f}, "
            f"p01={self.p01:.3f}, p99={self.p99:.3f}, "
            f"sat_lo={self.saturation_lo:.4f}, sat_hi={self.saturation_hi:.4f}, "
            f"CV={self.contrast_ratio:.3f}, SNR={self.snr:.2f}, "
            f"range={self.range_utilisation:.3f}, entropy={self.entropy:.2f}b"
        )

    def converged(
        self, mean_target: float, mean_tolerance: float, saturation_limit: float
    ) -> bool:
        """Return True when mean and saturation hard criteria are both satisfied."""
        return (
            abs(self.mean - mean_target) <= mean_tolerance
            and self.saturation_hi <= saturation_limit
        )


class FibsemImage:
    """
    Class representing a FibsemImage and its associated metadata.

    Args:
        data (np.ndarray): The image data stored in a numpy array.
        metadata (FibsemImageMetadata, optional): The metadata associated with the image. Defaults to None.

    Methods:
        load(cls, tiff_path: str) -> "FibsemImage":
            Loads a FibsemImage from a tiff file.

            Args:
                tiff_path (path): path to the tif* file

            Returns:
                FibsemImage: instance of FibsemImage

        save(self, path: Path) -> str:
            Saves a FibsemImage to a tiff file.

            Inputs:
                path (path): path to save directory and filename

            Returns:
                str: the resolved path written to

    Attributes:
        filepath (Optional[str]): the file this image is associated with on disk, set by
            save() and load(). None for an image that has never been written or read.
    """

    def __init__(
        self, data: np.ndarray, metadata: Optional[FibsemImageMetadata] = None
    ):
        if check_data_format(data):
            if data.ndim == 3 and data.shape[2] == 1:
                data = data[:, :, 0]
            self.data = data  # setter also populates _filtered_data
        else:
            raise Exception("Invalid Data format for Fibsem Image")
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None
        # the file this image is associated with on disk, set by save() and load().
        # not serialised: it describes where the image lives, not what it contains.
        self.filepath: Optional[str] = None

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the image data."""
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the image data."""
        return self.data.dtype

    @property
    def data(self) -> NDArray:
        """Returns the image data as a numpy array."""
        return self._data

    @data.setter
    def data(self, value: NDArray) -> None:
        if check_data_format(value):
            self._data = value
            self._filtered_data = self._filter_data(value)
        else:
            raise Exception("Invalid Data format for Fibsem Image")

    @property
    def filtered_data(self) -> NDArray:
        """Returns a median filtered version of the image data. Typically used for display purposes."""
        return self._filtered_data

    def _filter_data(self, data, size: int = 3, sigma: float = 1) -> NDArray:
        """Returns a filtered version of the image data using a median filter followed by a gaussian filter. Can be used for display or processing purposes."""
        # opencv rather than scipy: the same two filters, ~300x faster at 4096x4096. the
        # setter runs this on every assignment, including the one inside load(), so a
        # 385 MB overview took ~30 s to load on scipy and ~0.2 s here. data is 2D uint8 or
        # uint16 (check_data_format), exactly what medianBlur(ksize=3) accepts.
        # the kernel size and border mode match scipy, not opencv's defaults, which would be
        # a 7x7 kernel and reflect-101 -- 16x further from the output this replaces.
        radius = int(4.0 * sigma + 0.5)  # scipy's gaussian_filter default truncate=4.0
        ksize = 2 * radius + 1
        filtered = cv2.GaussianBlur(
            cv2.medianBlur(data, size),
            (ksize, ksize),
            sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        # opencv drops a trailing length-1 axis. (H, W, 1) can reach the setter unsqueezed
        # -- only __init__ squeezes it -- and filtered_data has always matched data's shape.
        return filtered.reshape(data.shape)

    @classmethod
    def load(cls, tiff_path: str) -> "FibsemImage":
        """Loads a FibsemImage from a tiff file.

        Args:
            tiff_path (path): path to the tif* file

        Returns:
            FibsemImage: instance of FibsemImage
        """
        with tff.TiffFile(tiff_path) as tiff_image:
            data = tiff_image.asarray()
            try:
                metadata = FibsemImageMetadata.from_dict(
                    image_format.read_metadata_dict(tiff_image)
                )
            except Exception:
                metadata = None
        image = cls(data=data, metadata=metadata)
        image.filepath = str(tiff_path)
        return image

    def save(self, path: Optional[Union[Path, str]] = None) -> str:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            path (path): path to save directory and filename

        Returns:
            str: the resolved path the image was written to (also set on self.filepath)
        """

        if path is None:
            if self.metadata is None:
                raise ValueError(
                    "No metadata provided, cannot determine save path. Please provide a path."
                )
            filename = self.metadata.image_settings.filename
            directory = self.metadata.image_settings.path
            if filename is None:
                raise ValueError(
                    "No filename provided in metadata, cannot determine save path. Please provide a path."
                )
            if directory is None:
                raise ValueError(
                    "No path provided in metadata, cannot determine save path. Please provide a path."
                )
            # The recorded path is an absolute directory on whichever machine acquired
            # the image, and it travels inside the file. Creating it would mean loading
            # a colleague's image and re-saving it silently reconstructs their directory
            # tree here -- `D:\SharedData\<their name>\...` and all. A path the caller
            # passes in is theirs to create; one that arrived in a file is not.
            if not os.path.isdir(directory):
                raise ValueError(
                    f"The directory recorded in this image's metadata does not exist: "
                    f"{directory}. It is from the machine that acquired the image. "
                    f"Please provide a path."
                )
            path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = Path(path).with_suffix(".tif")

        image_format.write(
            str(path),
            self.data,
            description=image_format.beam_image_ome(self.data, self.metadata),
            levels=image_format.has_levels(self.data.shape),
        )
        # set only after a successful write, so a recorded path is always a path that exists
        self.filepath = str(path)
        return self.filepath

    def extract_region(self, rect: "FibsemRectangle") -> "FibsemImage":
        """Extract a sub-region of the image and return a new FibsemImage with valid metadata.

        The returned image has the same resolution/hfw as the original (metadata describes
        the full scan), with reduced_area updated to reflect the extracted region.

        Args:
            rect (FibsemRectangle): Normalized rectangle (0–1 coordinates) defining the region to extract.

        Returns:
            FibsemImage: A new FibsemImage containing the cropped data and updated metadata.

        Raises:
            ValueError: If metadata is None or if rect coordinates are invalid / out of bounds.
        """
        if self.metadata is None:
            raise ValueError("Cannot extract region from FibsemImage without metadata.")

        if not rect.is_valid_reduced_area:
            raise ValueError(
                f"Invalid rectangle: {rect.pretty_string}. "
                "left/top must be >= 0, width/height > 0, and region must not exceed image bounds."
            )

        # Convert normalized coords to pixel indices using existing helper
        x, y, pw, ph = rect.to_pixel_coordinates(
            self.data.shape
        )  # (x, y, width, height)
        cropped = self.data[y : y + ph, x : x + pw].copy()

        # Clone metadata; only update reduced_area — resolution/hfw/pixel_size unchanged

        new_metadata = deepcopy(self.metadata)
        new_metadata.image_settings.reduced_area = rect

        return FibsemImage(data=cropped, metadata=new_metadata)

    def resize(self, resolution: Tuple[int, int]) -> "FibsemImage":
        """Resize the image to the given resolution and return a new FibsemImage with updated metadata.

        HFW is preserved; pixel_size is recalculated to match the new pixel dimensions.

        Args:
            resolution (Tuple[int, int]): Target resolution as (width, height) in pixels.

        Returns:
            FibsemImage: A new FibsemImage with resized data and updated metadata.

        Raises:
            ValueError: If metadata is None.
        """
        if self.metadata is None:
            raise ValueError("Cannot resize FibsemImage without metadata.")

        from skimage.transform import resize as skimage_resize

        new_width, new_height = resolution
        resized = skimage_resize(
            self.data,
            output_shape=(new_height, new_width),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(self.data.dtype)

        new_metadata = deepcopy(self.metadata)
        new_metadata.image_settings.resolution = resolution
        # pixel size scales inversely with resolution at fixed HFW
        orig_height, orig_width = self.data.shape
        new_metadata.pixel_size = Point(
            x=self.metadata.pixel_size.x * (orig_width / new_width),
            y=self.metadata.pixel_size.y * (orig_height / new_height),
        )

        return FibsemImage(data=resized, metadata=new_metadata)

    def apply_gamma(self, gamma: float) -> "FibsemImage":
        """Return a copy of the image with the given gamma correction applied.

        Args:
            gamma (float): Gamma value to apply. Must be > 0.
                Values < 1 brighten the image; values > 1 darken it.

        Returns:
            FibsemImage: New image with gamma-corrected data and the same metadata.

        Raises:
            ValueError: If gamma is not positive.
        """

        from fibsem.autofunctions.gamma import apply_gamma as _apply_gamma

        return FibsemImage(
            data=_apply_gamma(self.data, gamma), metadata=deepcopy(self.metadata)
        )

    def compute_stats(self) -> "ImageStats":
        """Compute histogram statistics for this image.

        Returns:
            ImageStats with all metrics normalised to [0, 1].
        """
        data = self.filtered_data.astype(np.float64)
        if np.issubdtype(self.data.dtype, np.floating):
            dtype_max = 1.0
        else:
            dtype_max = float(np.iinfo(self.data.dtype).max)

        norm = data / dtype_max
        mean = float(np.mean(norm))
        std = float(np.std(norm))
        p01 = float(np.percentile(norm, 1))
        p99 = float(np.percentile(norm, 99))
        sat_lo = float(np.mean(norm <= 0.0))
        sat_hi = float(np.mean(norm >= 1.0))
        cv = std / mean if mean > 0 else 0.0
        median = float(np.median(norm))
        snr = mean / std if std > 0 else 0.0

        counts, _ = np.histogram(norm, bins=256, range=(0.0, 1.0))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

        return ImageStats(
            mean=mean,
            std=std,
            p01=p01,
            p99=p99,
            saturation_lo=sat_lo,
            saturation_hi=sat_hi,
            contrast_ratio=cv,
            range_utilisation=p99 - p01,
            median=median,
            snr=snr,
            entropy=entropy,
        )

    @property
    def brightness(self) -> float:
        """Mean pixel intensity of the image."""
        return float(np.mean(self.data))

    @staticmethod
    def generate_blank_image(
        resolution: Tuple[int, int] = (1536, 1024),
        hfw: float = 100e-6,
        pixel_size: Optional[Point] = None,
        random: bool = False,
        dtype: np.dtype = np.uint8,
    ) -> "FibsemImage":
        """Generate a blank image with a given resolution and field of view.
        Args:
            resolution: List[int]: Resolution of the image.
            hfw: float: Horizontal field width of the image.
            pixel_size: Point: Pixel size of the image.
            random: bool: If True, generate a random (noise) image.
            dtype: np.dtype: Data type of the image. Defaults to np.uint8.
        Returns:
            FibsemImage: Blank image with valid metadata from display.
        """
        # need at least one of hfw, pixelsize
        if pixel_size is None and hfw is None:
            raise ValueError("Need to specify either hfw or pixelsize")

        if pixel_size is None:
            vfw = hfw * resolution[1] / resolution[0]
            pixel_size = Point(hfw / resolution[0], vfw / resolution[1])

        shape = (resolution[1], resolution[0])
        if random:
            arr = np.random.randint(0, 255, size=shape, dtype=dtype)
        else:
            arr = np.zeros(shape=shape, dtype=dtype)

        image = FibsemImage(
            data=arr,
            metadata=FibsemImageMetadata(
                image_settings=ImageSettings(hfw=hfw, resolution=resolution),
                microscope_state=None,
                pixel_size=pixel_size,
            ),
        )
        return image


def check_data_format(data: np.ndarray) -> bool:
    """Checks that data is in the correct format."""
    # assert data.ndim == 2  # or data.ndim == 3
    # assert data.dtype in [np.uint8, np.uint16]
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    return data.ndim == 2 and data.dtype in [np.uint8, np.uint16]


def save_tiff(data: np.ndarray, path: Union[str, Path]) -> str:
    """Write a raw image array to a TIFF file.

    Args:
        data: Image data to write.
        path: Destination path (``.tif`` appended if no suffix given).

    Returns:
        The path written to, as a string.
    """
    path = str(path)
    if not path.lower().endswith((".tif", ".tiff")):
        path += ".tif"
    tff.imwrite(path, data)
    return path


def load_tiff(path: Union[str, Path]) -> np.ndarray:
    """Read a raw image array from a TIFF file."""
    return tff.imread(str(path))
