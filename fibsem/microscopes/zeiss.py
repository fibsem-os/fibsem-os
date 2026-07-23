"""ZEISS FIB-SEM microscope backend for fibsem-os.

ZEISS instruments do not provide a direct Python API. Control is done through the
SmartSEM/SmartFIB COM automation interface (``CZ.EMApiCtrl.1``), which is wrapped
into Python by the modules under :mod:`fibsem.microscopes.zeiss_api`.

Those driver modules (``SEM_API.py``, ``read_probe_table.py``) are vendored from
the SerialFIB project; see :mod:`fibsem.microscopes.zeiss_api` for attribution.

This module adapts that layer to the :class:`~fibsem.microscope.FibsemMicroscope`
abstract interface, so ZEISS hardware can be driven with the same structured
dataclasses (``FibsemStagePosition``, ``ImageSettings``, ``FibsemImage``,
``BeamType`` ...) as the Thermo and Tescan backends.

Notes on the logic difference vs. SerialFIB
--------------------------------------------
SerialFIB drives ZEISS through raw SmartSEM parameter strings (``AP_*`` analogue,
``DP_*`` digital, ``CMD_*`` commands) and does patterning by writing ``.ely``
files to disk and firing ``CMD_SMARTFIB_LOAD_ELY``. fibsem-os instead expects the
structured abstract interface. The bridge is :meth:`ZeissMicroscope._get` /
:meth:`ZeissMicroscope._set`, which translate fibsem-os keys into SmartSEM
parameter operations on the vendored :class:`SEM_API`.

Parameter-name provenance
-------------------------
Names marked ``# EVIDENCED`` are used by SerialFIB and are known-good. Names
marked ``# VERIFY`` are the conventional SmartSEM names but were not exercised by
SerialFIB; confirm them against the SmartSEM parameter database on the target
instrument before relying on them on hardware.
"""

import datetime
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import fibsem.constants as constants
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemExperiment,
    FibsemGasInjectionSettings,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    SystemSettings,
)

# The vendored SEM_API wrapper performs COM setup at import time (pywin32 +
# the ZEISS type library), so importing it will fail on any machine that is not
# a ZEISS control PC. Guard it exactly like TESCAN_API_AVAILABLE in tescan.py.
ZEISS_API_AVAILABLE = False
try:
    from fibsem.microscopes.zeiss_api.SEM_API import SEM_API
    from fibsem.microscopes.zeiss_api.read_probe_table import getProbe, readProbeTable

    ZEISS_API_AVAILABLE = True
except Exception as e:  # noqa: BLE001 - COM/pywin32/type-lib may raise many things
    logging.debug(f"SmartSEM API (ZEISS) not available. {e}")


# SmartSEM image stores (DP_IMAGE_STORE, "W * H"). Only these fixed 4:3 sizes are
# available; other requested resolutions are served by resizing after the grab.
ZEISS_IMAGE_STORES = [
    (512, 384), (1024, 768), (2048, 1536), (3072, 2304),
    (4096, 3072), (6144, 4608), (8192, 6144),
]

# SmartSEM API grab target. SEM_API.grab_full_image writes the frame to disk and
# SerialFIB reads it back from here (C:/api/Grab.tif). Keep the same convention.
ZEISS_API_GRAB_PATH = r"C:/api/Grab.tif"

# Field-of-view limits (metres). Placeholder values, refine per instrument.
SEM_LIMITS: Dict[str, Tuple[float, float]] = {"hfw": (1.0e-6, 3.0e-3)}
FIB_LIMITS: Dict[str, Tuple[float, float]] = {"hfw": (1.0e-6, 1.0e-3)}
LIMITS = {BeamType.ELECTRON: SEM_LIMITS, BeamType.ION: FIB_LIMITS}

# Fallback available-value lists, used by get_available_values when a live source
# (e.g. the SmartFIB probe table) is not configured. Refine per instrument.
ZEISS_FIB_CURRENTS = [1.0e-12, 10.0e-12, 30.0e-12, 50.0e-12, 100.0e-12, 300.0e-12,
                      700.0e-12, 1.0e-9, 3.0e-9, 7.0e-9, 15.0e-9, 30.0e-9]  # Ga LMIS probes
ZEISS_SEM_CURRENTS = [1.0e-12, 10.0e-12, 50.0e-12, 100.0e-12, 200.0e-12, 400.0e-12,
                      1.0e-9, 2.0e-9, 4.0e-9, 10.0e-9]
ZEISS_SEM_VOLTAGES = [1000.0, 2000.0, 3000.0, 5000.0, 10000.0, 15000.0, 20000.0, 30000.0]
ZEISS_FIB_VOLTAGE = 30000.0  # Ga FIB is fixed at 30 kV
ZEISS_SCAN_DIRECTIONS = ["TopToBottom", "BottomToTop", "LeftToRight", "RightToLeft"]

# ZEISS FIB imaging probes are selected by a STRING (SmartSEM DP_FIB_IMAGE_PROBE),
# e.g. "30kV:50pA" -- the accelerating voltage and the beam current. fibsem-os
# represents beam current as a float everywhere, so we convert between the two.
ZEISS_FIB_PROBE_STRINGS = [
    "30kV:1pA", "30kV:10pA", "30kV:30pA", "30kV:50pA", "30kV:100pA", "30kV:300pA",
    "30kV:700pA", "30kV:1nA", "30kV:3nA", "30kV:7nA", "30kV:15nA", "30kV:30nA",
]
_PROBE_CURRENT_UNITS = {"pa": 1e-12, "na": 1e-9, "ua": 1e-6, "a": 1.0}


def parse_probe_current(probe: str) -> Optional[float]:
    """Extract the beam current (amps) from a SmartFIB probe string, e.g. "30kV:50pA" -> 50e-12."""
    if not isinstance(probe, str):
        return None
    m = re.search(r"([\d.]+)\s*(pA|nA|uA|µA|A)", probe, re.IGNORECASE)
    if not m:
        return None
    unit = m.group(2).lower().replace("µ", "u")
    return float(m.group(1)) * _PROBE_CURRENT_UNITS.get(unit, 1.0)


def format_probe_current(current: float, kv: int = 30) -> str:
    """Format a current (amps) as a SmartFIB probe string, e.g. 50e-12 -> "30kV:50pA"."""
    if current < 1e-9:
        val, unit = current * 1e12, "pA"
    elif current < 1e-6:
        val, unit = current * 1e9, "nA"
    else:
        val, unit = current * 1e6, "uA"
    return f"{kv}kV:{val:g}{unit}"


# SmartFIB milling geometry/dose. Defaults mirror the SerialFIB .ely templates.
ZEISS_MILL_CYCLES = 6125            # SmartFIB cycle count used in the dose model
ZEISS_MILL_PIXEL_SPACING = 0.5      # fraction (== "50 %")
ZEISS_MILL_TRACK_SPACING = 0.5      # fraction (== "50 %")
ZEISS_DEFAULT_PROBE_DIAMETER = 4.5e-8  # m, fallback Ga probe diameter (from SerialFIB refs)
ZEISS_DEFAULT_MILL_TIME = 60.0      # s, fallback per-pattern mill time when unspecified
# SmartFIB Drop folder: dropping an .ely here and firing CMD_SMARTFIB_LOAD_ELY loads it.
ZEISS_SMARTFIB_DROP_PATH = r"C:/ProgramData/Carl Zeiss/SmartFIB/API/Drop/ApiLayout.ely"


def calculate_dose_area(probe_current: float, probe_size: float, pixel_spacing: float,
                        track_spacing: float, cycle: int, mill_time: float,
                        width: float, height: float) -> float:
    """Compute the SmartFIB area dose (C/m^2) that yields ``mill_time`` seconds.

    Ported from SerialFIB ``calculate_dwell_time`` (caculate_mill_time_fct.py):
    given the desired total mill time, derive the per-pixel dwell and hence the
    area dose that SmartFIB's "by purpose" exposure will apply.
    """
    area = height * width
    spot_area = probe_size ** 2 * pixel_spacing * track_spacing
    if spot_area <= 0 or area <= 0:
        return 0.0
    n_pixel = area / spot_area
    dwell_time = mill_time / (n_pixel * cycle)
    dose = dwell_time * cycle * probe_current / spot_area
    return abs(dose)


def build_ely_xml(rectangles: List["FibsemRectangleSettings"], probe_name: str,
                  probe_current: float, probe_diameter: float,
                  dose_area_fn, layout_name: str = "fibsem_layout") -> bytes:
    """Build a SmartFIB ``.ely`` layout (XML bytes) for a list of rectangle patterns.

    Mirrors the SerialFIB template (``TemplatePatterns/Zeiss/layout001.ely``):
    one ``RECT`` per pattern with geometry in micrometres, an ``EXPOSURE`` with a
    computed ``dose_area``, and a ``PROBE`` selecting the FIB current.
    ``dose_area_fn(rect) -> float`` returns the area dose (C/m^2) per rectangle.
    """
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    root = ET.Element("ELAYOUT", version="2.0", locked="false", name=layout_name)
    ET.SubElement(root, "VERSION", created=ts, modified=ts, number="1.0")
    ET.SubElement(root, "AXES", show="true")
    ET.SubElement(root, "GRID", horizontal="1", show="true", snap_to="false", vertical="1")
    layer_list = ET.SubElement(root, "LAYER_LIST")
    ET.SubElement(layer_list, "LAYER", fill_color="#00FF00", fill_opacity="0.5",
                  hidden="false", locked="false", name="Layer")
    structure_list = ET.SubElement(root, "STRUCTURE_LIST")
    structure = ET.SubElement(structure_list, "STRUCTURE", locked="false", name="Structure")
    ET.SubElement(structure, "VERSION", created=ts, modified=ts, number="1.0")
    ET.SubElement(structure, "INSTANCE_LIST")
    layer_ref = ET.SubElement(structure, "LAYER_REFERENCE",
                              frame_cx="0", frame_cy="0", frame_size="100", ref="Layer")

    for rect in rectangles:
        dose = dose_area_fn(rect)
        rect_el = ET.SubElement(
            layer_ref, "RECT",
            edge_sel_auto_scan_angle="true",
            height=f"{rect.height * 1e6:g}", width=f"{rect.width * 1e6:g}",
            angle="0 deg", x=f"{rect.centre_x * 1e6:g}", y=f"{rect.centre_y * 1e6:g}",
            imaging="false",
        )
        exposure = ET.SubElement(
            rect_el, "EXPOSURE",
            version="2.1", column_type="FIB", purpose="FIB milling",
            computed_parameter="by purpose",
            dwell_times_point="1e-006 s", dwell_times_line="1e-006 s",
            dwell_times_area="1e-006 s", dwell_times_image="1e-006 s",
            delay="none", cycle_delay="0 s",
            dose_image="0 C/m²", dose_area=f"{dose:g} C/m²",
            dose_line="0 C/m", dose_point="0 C", pause="false",
            scanning_mode_fast="by purpose", scanning_mode_cycle_mode="by purpose",
            pixel_spacing_image="0 m", pixel_spacing_area="50 %",
            pixel_spacing_line="50 %", track_spacing="50 %", description="",
        )
        ET.SubElement(exposure, "PROBE", name=probe_name, type="specific",
                      current=f"{probe_current:g} A", diameter=f"{probe_diameter:g} m")
        ET.SubElement(exposure, "GIS", name="unchanged", channel="0", category="0",
                      type="unchanged", ack="false", autopark="false",
                      offset="false", usegas="false")

    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)

# SmartSEM active-beam command per beam type (EVIDENCED, SEM_API.set_active_beam).
_ACTIVE_BEAM_ARG = {BeamType.ELECTRON: "ELECTRON", BeamType.ION: "ION"}


# ----------------------------------------------------------------------------
# Conversion helpers (raw SmartSEM tuples <-> fibsem-os dataclasses)
# ----------------------------------------------------------------------------
# IMPORTANT: SEM_API stage tuples are ordered (x, y, z, t, r, m) -- tilt BEFORE
# rotation -- with x/y/z in metres and t/r in DEGREES. fibsem-os
# FibsemStagePosition is (x, y, z, r, t) with lengths in metres and angles in
# radians. Keep the axis swap + unit conversion in one place.

def from_zeiss_stage_position(position: Tuple[float, ...]) -> FibsemStagePosition:
    """Convert a raw SmartSEM stage tuple (x, y, z, t, r, [m]) to FibsemStagePosition."""
    x, y, z, t, r = position[:5]
    return FibsemStagePosition(
        x=float(x),
        y=float(y),
        z=float(z),
        r=float(r) * constants.DEGREES_TO_RADIANS,
        t=float(t) * constants.DEGREES_TO_RADIANS,
        coordinate_system="RAW",
    )


def to_zeiss_stage_position(position: FibsemStagePosition) -> Tuple[float, float, float, float, float]:
    """Convert a FibsemStagePosition to the SmartSEM absolute-move tuple (x, y, z, t, r).

    ``SEM_API.move_stage_absolute`` expects exactly this 5-tuple ordering and
    fills the 6th axis (m) from the current position itself.
    """
    x = position.x if position.x is not None else 0.0
    y = position.y if position.y is not None else 0.0
    z = position.z if position.z is not None else 0.0
    r = position.r * constants.RADIANS_TO_DEGREES if position.r is not None else 0.0
    t = position.t * constants.RADIANS_TO_DEGREES if position.t is not None else 0.0
    return (x, y, z, t, r)


def _read_image_file(path: str) -> np.ndarray:
    """Read a grabbed image file back into a 2D grayscale numpy array.

    ``SEM_API.grab_full_image`` writes the frame to disk. Note SmartSEM's Grab
    writes a BMP even when the filename ends in ``.tif``, and grayscale SEM/FIB
    frames come back as 3-channel RGB, so:
      - use a format-agnostic reader (auto-detects BMP/TIFF/PNG), and
      - collapse any RGB(A) result to a single channel (the channels are equal
        for grayscale images), returning 2D uint8/uint16 as FibsemImage requires.
    """
    img = None
    try:
        import imageio.v2 as imageio
        img = np.asarray(imageio.imread(path))
    except Exception:  # noqa: BLE001
        from skimage.io import imread
        img = np.asarray(imread(path))

    # collapse multi-channel (RGB/RGBA) grayscale frames to 2D
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def _resize_image(img: np.ndarray, resolution: Optional[Tuple[int, int]]) -> np.ndarray:
    """Resize ``img`` to ``resolution`` (width, height), preserving dtype.

    SmartSEM only provides a fixed set of image stores, so a grabbed frame may
    not match the resolution the caller asked for. Callers rely on
    ``acquire_image`` returning exactly ``image_settings.resolution``.
    """
    if resolution is None:
        return img
    width, height = int(resolution[0]), int(resolution[1])
    if img.shape[0] == height and img.shape[1] == width:
        return img

    dtype = img.dtype
    try:
        import cv2
        # INTER_AREA is the better choice when downsampling
        interp = cv2.INTER_AREA if (width < img.shape[1] or height < img.shape[0]) else cv2.INTER_LINEAR
        resized = cv2.resize(img, (width, height), interpolation=interp)
    except Exception:  # noqa: BLE001
        from skimage.transform import resize as sk_resize
        resized = sk_resize(img, (height, width), preserve_range=True, anti_aliasing=True)
    return resized.astype(dtype)


class ZeissMicroscope(FibsemMicroscope):
    """A ZEISS FIB-SEM microscope, driven through the SmartSEM/SmartFIB COM API.

    Backed by the vendored :class:`SEM_API` wrapper (migrated from SerialFIB).
    Implements the :class:`~fibsem.microscope.FibsemMicroscope` contract.

    Coverage (first pass):
      - fully wired: connect, stage get/move, image acquisition, beam shift,
        auto-focus / auto contrast-brightness, beam on/off, ion beam current.
      - partial: milling (``.ely`` file flow), sputter/GIS, manipulator -- these
        are either unsupported on ZEISS or need additional development
        (pattern -> ``.ely`` generation).
    """

    def __init__(self, system_settings: SystemSettings):
        if not ZEISS_API_AVAILABLE:
            raise ImportError(
                "The ZEISS SmartSEM API is not available. This requires pywin32 and "
                "the SmartSEM Remote API registered so that CZ.EMApiCtrl.1 resolves "
                "(on a Crossbeam this is provided by the ZEISS RRemote Client, which "
                "registers an x64 CZEMApi.ocx), on a ZEISS control PC with SmartSEM "
                "running."
            )

        # microscope client (created on connect)
        self.connection: Optional[SEM_API] = None

        # system settings
        self.system: SystemSettings = system_settings
        self.milling_channel: BeamType = BeamType.ION
        self.stage_is_compustage: bool = False
        self._last_imaging_settings: ImageSettings = ImageSettings()

        # currently active SmartSEM column (FIB vs SEM). None until first switch.
        self._active_beam_type: Optional[BeamType] = None

        # ion beam current lookup table (SmartFIB probe table). Set on connect
        # from system settings; used to map a requested current onto a probe.
        self._probe_table_path: Optional[str] = None

        # last ion probe label read from the instrument, e.g. "30kV:50pA", with the
        # current it parsed to. SmartFIB only accepts labels it knows, and the exact
        # spelling varies per instrument, so re-use the instrument's own string when
        # restoring the same current (e.g. set_microscope_state after an overview).
        self._ion_probe_label: Optional[str] = None
        self._ion_probe_current: Optional[float] = None

        # milling pattern buffer + configured .ely layout (see run_milling)
        self._patterns: List = []
        self._milling_settings: Optional[FibsemMillingSettings] = None

        # user / experiment metadata
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

        # last images
        self.last_image_eb: Optional[FibsemImage] = None
        self.last_image_ib: Optional[FibsemImage] = None

        # fluorescence microscope (not available on ZEISS)
        self.fm = None

        # cached beam parameters (not everything is queryable live)
        self._beam_parameters: Dict[BeamType, BeamSettings] = {
            BeamType.ELECTRON: BeamSettings(BeamType.ELECTRON),
            BeamType.ION: BeamSettings(BeamType.ION),
        }

        logging.debug({"msg": "create_microscope_client", "system_settings": system_settings.to_dict()})

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------
    def connect_to_microscope(self, ip_address: str = "localhost", port: int = 0, reset_beam_shift: bool = True) -> None:
        """Connect to the ZEISS EM server.

        The SmartSEM remoting interface is configured on the control PC (via
        RConfigure) rather than by IP/port here; ``ip_address``/``port`` are
        accepted for interface compatibility. ``state='remote'`` vs ``'local'``
        only affects the real-time image memory-map, which we don't use.
        """
        logging.info("Connecting to ZEISS SmartSEM API...")
        self.connection = SEM_API(state="remote")
        logging.info("Connected to ZEISS SmartSEM API.")

        # system info
        self.system.info.manufacturer = "ZEISS"
        try:
            self.system.info.serial_number = self.connection.GetState("SV_SERIAL_NUMBER")  # EVIDENCED
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Could not read serial number: {e}")

        # probe table for ion-beam current selection
        self._probe_table_path = getattr(self.system.ion, "probe_table_path", None)

        info = self.system.info
        logging.info(
            f"Connected to ZEISS model {info.model} (SN {info.serial_number}, SW {info.software_version})."
        )

        if reset_beam_shift:
            try:
                self.reset_beam_shifts()
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Could not reset beam shifts: {e}")

        # sample stage holder (needed by UI widgets)
        try:
            self._create_sample_stage()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Could not create sample stage: {e}")

    def disconnect(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Error while disconnecting: {e}")
        self.connection = None

    @property
    def manufacturer(self) -> str:
        return "Zeiss"

    # ------------------------------------------------------------------
    # beam helpers
    # ------------------------------------------------------------------
    def _prepare_beam(self, beam_type: BeamType) -> None:
        """Switch the SmartSEM active column to ``beam_type`` if needed."""
        if self._active_beam_type != beam_type:
            self.connection.set_active_beam(_ACTIVE_BEAM_ARG[beam_type])  # EVIDENCED
            self._active_beam_type = beam_type
            time.sleep(0.5)  # allow the column switch to settle

    # ------------------------------------------------------------------
    # imaging
    # ------------------------------------------------------------------
    def acquire_image(self, image_settings: Optional[ImageSettings] = None, beam_type: Optional[BeamType] = None) -> FibsemImage:
        """Acquire an image for the given settings/beam type.

        Grabs a full frame via ``SEM_API.grab_full_image`` (which writes a TIFF
        to disk) and reads it back into a :class:`FibsemImage`.
        """
        if image_settings is None and beam_type is None:
            raise ValueError("Must provide either image_settings or beam_type.")
        elif image_settings is not None:
            effective_beam_type = image_settings.beam_type
            effective_image_settings = image_settings
        else:
            effective_beam_type = beam_type
            effective_image_settings = self.get_imaging_settings(beam_type=beam_type)

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        # switch column + apply field of view / resolution
        self._prepare_beam(effective_beam_type)
        if image_settings is not None:
            if effective_image_settings.hfw is not None:
                try:
                    self.set_field_of_view(effective_image_settings.hfw, effective_beam_type)
                except Exception as e:  # noqa: BLE001
                    logging.debug(f"Could not set field of view: {e}")
            if effective_image_settings.resolution is not None:
                self._set_image_store(effective_image_settings.resolution)

        # grab a full frame. SEM_API.grab_full_image writes the frame to disk
        # (C:/api/Grab.tif by convention); read it back from there.
        fname = ZEISS_API_GRAB_PATH
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.connection.grab_full_image(fname)
        image_data = _read_image_file(fname)
        native_shape = image_data.shape

        # SmartSEM only offers a fixed set of (4:3) image stores, so the grabbed
        # frame may not match the requested resolution. Callers rely on getting
        # exactly image_settings.resolution back (e.g. tiled acquisition
        # pre-allocates from it), so resize to the requested shape.
        image_data = _resize_image(image_data, effective_image_settings.resolution)

        # build metadata (best-effort, live from the API)
        try:
            microscope_state = self.get_microscope_state(beam_type=effective_beam_type)
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Could not read microscope state for image metadata: {e}")
            microscope_state = MicroscopeState()

        pixel_size = self._get_pixel_size(
            image_data.shape, effective_beam_type, effective_image_settings, native_shape
        )

        md = FibsemImageMetadata(
            image_settings=effective_image_settings,
            microscope_state=microscope_state,
            pixel_size=pixel_size,
        )
        md.image_settings.beam_type = deepcopy(effective_beam_type)
        md.user = self.user
        md.experiment = self.experiment
        md.system = self.system

        fibsem_image = FibsemImage(data=image_data, metadata=deepcopy(md))

        if effective_beam_type == BeamType.ELECTRON:
            self.last_image_eb = fibsem_image
        else:
            self.last_image_ib = fibsem_image

        if image_settings is not None:
            self._last_imaging_settings = image_settings

        return fibsem_image

    def _set_image_store(self, resolution: Tuple[int, int]) -> None:
        """Set the SmartSEM image store, when the request matches one it offers.

        ``DP_IMAGE_STORE`` takes "W * H" strings and only the fixed 4:3 stores in
        :data:`ZEISS_IMAGE_STORES` exist. Anything else (e.g. a square overview
        resolution) is left to the resize in :meth:`acquire_image`; attempting it
        here would just make SmartSEM complain.
        """
        try:
            width, height = int(resolution[0]), int(resolution[1])
        except (TypeError, ValueError):
            return
        if (width, height) not in ZEISS_IMAGE_STORES:
            logging.debug(
                f"{width}x{height} is not a SmartSEM image store; grabbing at the "
                "current store and resizing to the requested resolution."
            )
            return
        try:
            target = f"{width} * {height}"
            if self.connection.GetState("DP_IMAGE_STORE") != target:  # EVIDENCED
                self.connection.SetState("DP_IMAGE_STORE", target)
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Could not set image store to {resolution}: {e}")

    def _get_pixel_size(self, image_shape: Tuple[int, ...], beam_type: BeamType,
                        image_settings: ImageSettings,
                        native_shape: Optional[Tuple[int, ...]] = None) -> Point:
        """Pixel size (metres) of the returned image.

        ``AP_IMAGE_PIXEL_SIZE`` describes the native store, so when the frame has
        been resized to the requested resolution the pixel size is scaled per
        axis. That keeps the physical field of view correct and reports the
        anisotropy honestly if the requested aspect ratio differs from SmartSEM's.
        """
        out_h, out_w = image_shape[0], image_shape[1]
        nat_h, nat_w = (native_shape[0], native_shape[1]) if native_shape else (out_h, out_w)

        native_ps: Optional[float] = None
        try:
            ps = self.connection.GetValue("AP_IMAGE_PIXEL_SIZE")  # EVIDENCED (metres)
            if ps and ps > 0:
                native_ps = float(ps)
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Could not read AP_IMAGE_PIXEL_SIZE: {e}")

        if native_ps is not None:
            return Point(x=native_ps * nat_w / out_w, y=native_ps * nat_h / out_h)

        # fall back: derive from hfw, with the vertical extent following the native aspect
        hfw = image_settings.hfw or 0.0
        px = hfw / out_w if out_w else 0.0
        py = (hfw * (nat_h / nat_w)) / out_h if (nat_w and out_h) else px
        return Point(x=px, y=py)

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        return self.last_image_eb if beam_type == BeamType.ELECTRON else self.last_image_ib

    def acquire_chamber_image(self) -> FibsemImage:
        logging.warning("Chamber (navigation camera) image is not supported on ZEISS via this API.")
        raise NotImplementedError("acquire_chamber_image is not supported on ZEISS.")

    def autocontrast(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Run SmartSEM auto contrast-brightness (Quick BC)."""
        self._prepare_beam(beam_type)
        self.connection.Execute("CMD_QUICK_BC")  # EVIDENCED
        time.sleep(0.5)
        while self.connection.GetState("DP_AUTO_FUNCTION") != "Idle":  # EVIDENCED
            time.sleep(0.5)

    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Run SmartSEM fine autofocus."""
        self._prepare_beam(beam_type)
        self.connection.do_autofocus()  # EVIDENCED (CMD_AUTO_FOCUS_FINE + DP_AUTO_FN_STATUS)

    # ------------------------------------------------------------------
    # beam shift
    # ------------------------------------------------------------------
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType = BeamType.ION) -> None:
        """Apply a relative beam shift (metres)."""
        current = self._get("shift", beam_type)
        new_shift = Point(x=current.x + dx, y=current.y + dy)
        self._set("shift", new_shift, beam_type)

    # ------------------------------------------------------------------
    # stage movement
    # ------------------------------------------------------------------
    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        logging.info(f"Moving stage (absolute) to {position}.")
        self.connection.move_stage_absolute(to_zeiss_stage_position(position))
        self.connection.wait_for_stage_idle()
        return self.get_stage_position()

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        logging.info(f"Moving stage (relative) by {position}.")
        # SEM_API.move_stage_relative expects (dx, dy, dz, dt, dr)
        dx = position.x or 0.0
        dy = position.y or 0.0
        dz = position.z or 0.0
        dr = (position.r or 0.0) * constants.RADIANS_TO_DEGREES
        dt = (position.t or 0.0) * constants.RADIANS_TO_DEGREES
        self.connection.move_stage_relative((dx, dy, dz, dt, dr))
        self.connection.wait_for_stage_idle()
        return self.get_stage_position()

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        # No safety/collision model wired yet; direct move.
        self.move_stage_absolute(position)

    def stable_move(self, dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        """Move the stage in the sample plane, compensating for stage tilt in y."""
        base = self.get_stage_position()
        yz = self._y_corrected_stage_movement(dy, beam_type)
        new_position = FibsemStagePosition(x=base.x + dx, y=base.y + yz[0], z=base.z + yz[1], r=base.r, t=base.t, coordinate_system="RAW")
        return self.move_stage_absolute(new_position)

    def project_stable_move(self, dx: float, dy: float, beam_type: BeamType, base_position: FibsemStagePosition) -> FibsemStagePosition:
        yz = self._y_corrected_stage_movement(dy, beam_type)
        return FibsemStagePosition(
            x=base_position.x + dx,
            y=base_position.y + yz[0],
            z=base_position.z + yz[1],
            r=base_position.r,
            t=base_position.t,
            coordinate_system="RAW",
        )

    def vertical_move(self, dy: float, dx: float = 0.0, static_wd: bool = True) -> FibsemStagePosition:
        """Move vertically (perpendicular to the electron beam)."""
        base = self.get_stage_position()
        new_position = FibsemStagePosition(x=base.x + dx, y=base.y, z=base.z + dy, r=base.r, t=base.t, coordinate_system="RAW")
        return self.move_stage_absolute(new_position)

    def _y_corrected_stage_movement(self, expected_y: float, beam_type: BeamType) -> Tuple[float, float]:
        """Split an in-plane y movement into (y, z) given the stage/column tilt.

        Placeholder: pending the ZEISS pretilt/column-tilt geometry, treat the
        movement as pure y. Refine with the same trig as tescan/thermo backends.
        """
        return expected_y, 0.0

    # ------------------------------------------------------------------
    # manipulator (not supported on ZEISS / SerialFIB)
    # ------------------------------------------------------------------
    def insert_manipulator(self, name: str = "PARK") -> None:
        logging.warning("Manipulator is not supported on ZEISS.")

    def retract_manipulator(self) -> None:
        logging.warning("Manipulator is not supported on ZEISS.")

    def move_manipulator_relative(self, position: FibsemManipulatorPosition, name: str = None) -> None:
        logging.warning("Manipulator is not supported on ZEISS.")

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition, name: str = None) -> None:
        logging.warning("Manipulator is not supported on ZEISS.")

    def move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
        logging.warning("Manipulator is not supported on ZEISS.")

    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str = None) -> None:
        logging.warning("Manipulator is not supported on ZEISS.")

    def _get_saved_manipulator_position(self, name: str = None) -> Optional[FibsemManipulatorPosition]:
        logging.warning("Manipulator is not supported on ZEISS.")
        return None

    # ------------------------------------------------------------------
    # milling (partial: SmartFIB .ely file flow)
    # ------------------------------------------------------------------
    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        """Prepare for milling: switch to the ion column and reset the buffer."""
        self._milling_settings = mill_settings
        self.milling_channel = getattr(mill_settings, "milling_channel", None) or BeamType.ION
        self._prepare_beam(self.milling_channel)
        self.clear_patterns()

    def _estimate_pattern_mill_time(self, rect: FibsemRectangleSettings, milling_current: float) -> float:
        """Per-pattern mill time (s): the pattern's own time if set, else derived
        from the milling settings sputter rate, else a default."""
        if getattr(rect, "time", 0) and rect.time > 0:
            return float(rect.time)
        rate = getattr(self._milling_settings, "rate", 0) if self._milling_settings else 0
        if rate and milling_current:
            try:
                t = (rect.width * rect.height * rect.depth) / (rate * milling_current)
                if t > 0:
                    return float(t)
            except Exception:  # noqa: BLE001
                pass
        return ZEISS_DEFAULT_MILL_TIME

    def _generate_ely(self, milling_current: float) -> bytes:
        """Build a SmartFIB ``.ely`` layout from the buffered rectangle patterns."""
        rectangles = [p for p in self._patterns if isinstance(p, FibsemRectangleSettings)]
        skipped = len(self._patterns) - len(rectangles)
        if skipped:
            logging.warning(f"{skipped} non-rectangle pattern(s) skipped: only rectangles "
                            "are supported for ZEISS milling so far.")
        if not rectangles:
            raise ValueError("No rectangle patterns to mill.")

        probe_name = self._nearest_probe_string(milling_current)
        probe_diameter = getattr(self._milling_settings, "spot_size", None) or ZEISS_DEFAULT_PROBE_DIAMETER

        def dose_area_fn(rect: FibsemRectangleSettings) -> float:
            mill_time = self._estimate_pattern_mill_time(rect, milling_current)
            return calculate_dose_area(
                probe_current=milling_current, probe_size=probe_diameter,
                pixel_spacing=ZEISS_MILL_PIXEL_SPACING, track_spacing=ZEISS_MILL_TRACK_SPACING,
                cycle=ZEISS_MILL_CYCLES, mill_time=mill_time,
                width=rect.width, height=rect.height,
            )

        return build_ely_xml(rectangles, probe_name, milling_current, probe_diameter, dose_area_fn)

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False) -> None:
        """Generate a SmartFIB ``.ely`` from the buffered patterns and run it.

        Mirrors the SerialFIB flow: write the layout, drop it into the SmartFIB
        Drop folder, then fire LOAD -> PREPARE -> START and poll ``DP_FIB_MODE``
        until milling completes.
        """
        ely_bytes = self._generate_ely(milling_current)

        os.makedirs(os.path.dirname(ZEISS_SMARTFIB_DROP_PATH), exist_ok=True)
        with open(ZEISS_SMARTFIB_DROP_PATH, "wb") as f:
            f.write(ely_bytes)
        logging.info(f"Wrote SmartFIB layout ({len(self._patterns)} pattern(s)) to {ZEISS_SMARTFIB_DROP_PATH}")

        self._prepare_beam(BeamType.ION)
        self.connection.Execute("CMD_SMARTFIB_LOAD_ELY")  # EVIDENCED
        time.sleep(1)
        self.connection.Execute("CMD_SMARTFIB_PREPARE_EXPOSURE")  # EVIDENCED
        time.sleep(1)
        self.connection.Execute("CMD_FIB_START_MILLING")  # EVIDENCED
        self.connection.SetState("DP_PATTERNING_MODE", "FIB")  # EVIDENCED
        if asynch:
            return
        t0 = time.time()
        while self.connection.GetState("DP_FIB_MODE") != "Milling" and time.time() - t0 < 30:
            time.sleep(0.3)
        while self.connection.GetState("DP_FIB_MODE") == "Milling":  # EVIDENCED
            time.sleep(0.3)
        logging.info("SmartFIB milling finished.")

    def finish_milling(self, imaging_current: float = None, imaging_voltage: float = None) -> None:
        self.clear_patterns()
        self._prepare_beam(BeamType.ELECTRON)

    def stop_milling(self) -> None:
        logging.warning("stop_milling: no direct SmartFIB stop wired; use native UI to abort.")

    def clear_patterns(self) -> None:
        self._patterns = []

    def start_milling(self) -> None:
        self.connection.Execute("CMD_FIB_START_MILLING")  # EVIDENCED

    def pause_milling(self) -> None:
        logging.warning("pause_milling is not implemented for ZEISS.")

    def resume_milling(self) -> None:
        logging.warning("resume_milling is not implemented for ZEISS.")

    def get_milling_state(self) -> MillingState:
        try:
            mode = self.connection.GetState("DP_FIB_MODE")  # EVIDENCED
        except Exception:  # noqa: BLE001
            return MillingState.IDLE
        return MillingState.RUNNING if mode == "Milling" else MillingState.IDLE

    def estimate_milling_time(self) -> float:
        """Estimate total mill time (s) for the buffered rectangle patterns."""
        current = self._milling_settings.milling_current if self._milling_settings else 50e-12
        return sum(
            self._estimate_pattern_mill_time(p, current)
            for p in self._patterns if isinstance(p, FibsemRectangleSettings)
        )

    # ------------------------------------------------------------------
    # pattern drawing (buffered, then emitted as an .ely by run_milling)
    # ------------------------------------------------------------------
    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        self._patterns.append(pattern_settings)
        return pattern_settings

    def draw_line(self, pattern_settings: FibsemLineSettings):
        self._patterns.append(pattern_settings)
        return pattern_settings

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        self._patterns.append(pattern_settings)
        return pattern_settings

    def draw_annulus(self, pattern_settings: FibsemCircleSettings):
        self._patterns.append(pattern_settings)
        return pattern_settings

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings):
        self._patterns.append(pattern_settings)
        return pattern_settings

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings):
        self._patterns.append(pattern_settings)
        return pattern_settings

    # ------------------------------------------------------------------
    # gas injection / sputter (not supported)
    # ------------------------------------------------------------------
    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        logging.warning("Gas injection (cryo deposition) is not supported on ZEISS via this API.")

    def setup_sputter(self, *args, **kwargs):
        logging.warning("Sputtering is not supported on ZEISS via this API.")

    def draw_sputter_pattern(self, *args, **kwargs) -> None:
        logging.warning("Sputtering is not supported on ZEISS via this API.")

    def run_sputter(self, *args, **kwargs):
        logging.warning("Sputtering is not supported on ZEISS via this API.")

    def finish_sputter(self, *args, **kwargs):
        logging.warning("Sputtering is not supported on ZEISS via this API.")

    # ------------------------------------------------------------------
    # available values
    # ------------------------------------------------------------------
    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None) -> List[Union[str, float]]:
        """Return the list of available values for a given key.

        The UI caches several of these on connect (current, voltage, preset,
        application_file, scan_direction) and some widgets index/min() over the
        result, so keys that feed those widgets must return a non-empty list.
        """
        if key == "current":
            if beam_type == BeamType.ION:
                return self._get_ion_beam_currents()
            return list(ZEISS_SEM_CURRENTS)
        if key == "voltage":
            if beam_type == BeamType.ION:
                return [ZEISS_FIB_VOLTAGE]
            return list(ZEISS_SEM_VOLTAGES)
        if key == "scan_direction":
            return list(ZEISS_SCAN_DIRECTIONS)
        if key == "application_file":
            # ZEISS SmartFIB has no Thermo-style application files; placeholder
            return ["Si"]
        if key == "plasma_gas":
            return []
        if key == "preset":
            return []
        if key == "detector_type":
            return []
        return []

    def _get_ion_beam_currents(self) -> List[float]:
        """Available ion beam currents (amps).

        ZEISS FIB probes are strings like "30kV:50pA"; return the parsed float
        currents so the numeric UI works. Prefer the SmartFIB probe table if
        configured, else the parsed default probe-string list.
        """
        if self._probe_table_path and os.path.exists(self._probe_table_path):
            try:
                table = readProbeTable(self._probe_table_path)
                currents = sorted(float(c) for c in table.keys())
                if currents:
                    return currents
            except Exception as e:  # noqa: BLE001
                logging.debug(f"Could not read ion probe table {self._probe_table_path}: {e}")
        return [c for c in (parse_probe_current(p) for p in ZEISS_FIB_PROBE_STRINGS) if c is not None]

    def _nearest_probe_string(self, current: float) -> str:
        """Return the SmartFIB probe string whose current is closest to ``current`` (amps)."""
        return min(
            ZEISS_FIB_PROBE_STRINGS,
            key=lambda p: abs((parse_probe_current(p) or 0.0) - current),
        )

    def check_available_values(self, key: str, values=None, beam_type: Optional[BeamType] = None) -> bool:
        return False

    # ------------------------------------------------------------------
    # get / set: the SmartSEM parameter bridge
    # ------------------------------------------------------------------
    def _read_value(self, ap_name: str, default: float) -> float:
        """Read an analogue SmartSEM value, returning ``default`` on failure.

        Beam-parameter reads must never return None (the UI does numeric
        comparisons/min() over them), and several AP names are not yet verified
        on every instrument, so failures degrade to a sensible default.
        """
        try:
            value = self.connection.GetValue(ap_name)
            return float(value) if value is not None else default
        except Exception as e:  # noqa: BLE001
            logging.debug(f"GetValue({ap_name}) failed: {e}; using default {default}")
            return default

    def _safe_set_value(self, ap_name: str, value) -> bool:
        """Write an analogue SmartSEM value, warning instead of raising on failure.

        SmartSEM rejects out-of-range or unavailable parameters with an API error.
        Those must not propagate: a single failed write while restoring state
        would otherwise abort the caller's whole operation.
        """
        try:
            self.connection.SetValue(ap_name, value)
            return True
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Could not set {ap_name} to {value}: {e}")
            return False

    def _safe_execute(self, cmd_name: str) -> bool:
        """Execute a SmartSEM command, warning instead of raising on failure."""
        try:
            self.connection.Execute(cmd_name)
            return True
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Could not execute {cmd_name}: {e}")
            return False

    def _get(self, key: str, beam_type: Optional[BeamType] = None):
        """Translate a fibsem-os key into a SmartSEM parameter read."""
        conn = self.connection

        # stage
        if key == "stage_position":
            raw = conn.GetStagePosition()  # (x, y, z, t, r, m)
            return from_zeiss_stage_position(raw)
        if key == "stage_calibrated":
            return True

        # beam / optics (all reads fall back to numeric defaults, never None)
        if key == "shift":
            try:
                if beam_type == BeamType.ION:
                    x = conn.GetValue("AP_FIB_BEAM_SHIFT_X")  # EVIDENCED
                    y = conn.GetValue("AP_FIB_BEAM_SHIFT_Y")  # EVIDENCED
                else:
                    x = conn.GetValue("AP_BEAMSHIFT_X")  # EVIDENCED
                    y = conn.GetValue("AP_BEAMSHIFT_Y")  # EVIDENCED
                return Point(x=float(x), y=float(y))
            except Exception as e:  # noqa: BLE001
                logging.debug(f"Could not read beam shift: {e}")
                return Point(0.0, 0.0)
        if key == "working_distance":
            return self._read_value("AP_WD", 5.0e-3)  # VERIFY (metres)
        if key == "current":
            if beam_type == BeamType.ELECTRON:
                return self._read_value("AP_IPROBE", ZEISS_SEM_CURRENTS[0])  # VERIFY (amps)
            # ion current == the selected SmartFIB probe string, e.g. "30kV:50pA"
            try:
                probe = conn.GetState("DP_FIB_IMAGE_PROBE")  # EVIDENCED
                current = parse_probe_current(probe)
                if current is not None:
                    # remember the instrument's exact label so we can set it back verbatim
                    self._ion_probe_label = probe
                    self._ion_probe_current = current
                    return current
            except Exception as e:  # noqa: BLE001
                logging.debug(f"Could not read DP_FIB_IMAGE_PROBE: {e}")
            cached = self._beam_parameters[BeamType.ION].beam_current
            return cached if cached is not None else parse_probe_current(ZEISS_FIB_PROBE_STRINGS[0])
        if key == "voltage":
            default_v = ZEISS_FIB_VOLTAGE if beam_type == BeamType.ION else 2000.0
            return self._read_value("AP_ACTUALKV", default_v)  # VERIFY (volts)
        if key == "hfw":
            return self._read_value("AP_WIDTH", 150.0e-6)  # VERIFY (metres, field width)
        if key == "scan_rotation":
            return self._read_value("AP_SCANROTATION", 0.0)  # VERIFY (degrees)

        # cached-only parameters (not directly queryable)
        if key == "resolution":
            res = self._beam_parameters[beam_type].resolution if beam_type else None
            return res if res is not None else (1536, 1024)
        if key == "dwell_time":
            dt = self._beam_parameters[beam_type].dwell_time if beam_type else None
            return dt if dt is not None else 1.0e-6
        if key == "stigmation":
            return Point(0, 0)
        if key == "preset":
            return None
        if key == "on":
            return True

        # system properties
        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.eucentric_height
            if beam_type is BeamType.ION:
                return self.system.ion.eucentric_height
        if key == "column_tilt":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.column_tilt
            if beam_type is BeamType.ION:
                return self.system.ion.column_tilt
        if key == "beam_enabled":
            return True
        if key == "plasma":
            return self.system.ion.plasma if beam_type is BeamType.ION else False
        if key == "plasma_gas":
            return self.system.ion.plasma_gas if beam_type is BeamType.ION else None

        # detector
        if key == "detector_type":
            try:
                return conn.GetState("DP_DETECTOR_CHANNEL")  # EVIDENCED
            except Exception:  # noqa: BLE001
                return None
        if key in ("detector_brightness", "detector_contrast", "detector_mode"):
            return None

        # manufacturer
        if key == "manufacturer":
            return self.system.info.manufacturer
        if key == "model":
            return self.system.info.model
        if key == "serial_number":
            return self.system.info.serial_number
        if key == "software_version":
            return self.system.info.software_version
        if key == "hardware_version":
            return self.system.info.hardware_version

        logging.warning(f"Unknown/unsupported get key: {key} ({beam_type})")
        return None

    def _set(self, key: str, value, beam_type: Optional[BeamType] = None) -> None:
        """Translate a fibsem-os key/value into a SmartSEM parameter write."""
        conn = self.connection

        if key == "shift":
            if beam_type == BeamType.ION:
                self._safe_set_value("AP_FIB_BEAM_SHIFT_X", value.x)  # EVIDENCED
                self._safe_set_value("AP_FIB_BEAM_SHIFT_Y", value.y)  # EVIDENCED
            else:
                self._safe_set_value("AP_BEAMSHIFT_X", value.x)  # EVIDENCED
                self._safe_set_value("AP_BEAMSHIFT_Y", value.y)  # EVIDENCED
            return
        if key == "working_distance":
            self._safe_set_value("AP_WD", value)  # VERIFY (metres)
            return
        if key == "current":
            if beam_type == BeamType.ION:
                # ion current is selected by a SmartFIB probe string ("30kV:50pA").
                if isinstance(value, str):
                    probe_name = value
                    current_val = parse_probe_current(value)
                elif (self._ion_probe_label is not None
                      and self._ion_probe_current is not None
                      and np.isclose(value, self._ion_probe_current, rtol=1e-3)):
                    # restoring the current we last read: re-use the instrument's own
                    # label verbatim, since SmartFIB only accepts labels it knows
                    probe_name = self._ion_probe_label
                    current_val = float(value)
                elif self._probe_table_path:
                    probe = getProbe(value, self._probe_table_path)  # nearest from XML table
                    probe_name = probe["name"]
                    current_val = float(value)
                else:
                    probe_name = self._nearest_probe_string(value)
                    current_val = float(value)
                # a probe label SmartFIB does not know fails with API_E_SET_STATE_FAIL;
                # warn rather than raise, so a failed restore cannot abort the caller
                # (this previously discarded a completed tiled overview).
                try:
                    conn.SetState("DP_FIB_IMAGE_PROBE", probe_name)  # EVIDENCED
                    self._beam_parameters[BeamType.ION].beam_current = current_val
                    logging.info(f"Set ion beam probe to {probe_name}.")
                except Exception as e:  # noqa: BLE001
                    logging.warning(
                        f"Could not set ion beam probe to '{probe_name}' "
                        f"({value} A): {e}. Leaving the current probe selected."
                    )
            else:
                self._safe_set_value("AP_IPROBE", value)  # VERIFY (amps)
            return
        if key == "voltage":
            self._safe_set_value("AP_MANUALKV", value)  # VERIFY (volts)
            return
        if key == "hfw":
            limits = LIMITS.get(beam_type, SEM_LIMITS)["hfw"]
            value = float(np.clip(value, limits[0], limits[1]))
            self._safe_set_value("AP_WIDTH", value)  # VERIFY (metres)
            return
        if key == "scan_rotation":
            self._safe_set_value("AP_SCANROTATION", value)  # VERIFY (degrees)
            return
        if key == "on":
            self._safe_execute("CMD_BEAM_ON" if value else "CMD_BEAM_OFF")  # EVIDENCED (SEM)
            return
        if key == "detector_type":
            try:
                conn.SetState("DP_DETECTOR_CHANNEL", value)  # EVIDENCED
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Could not set detector channel: {e}")
            return

        # cached-only / unsupported
        if key == "resolution":
            # SmartSEM exposes fixed image stores; set the closest it accepts and
            # cache the request (acquire_image resizes to it if needed).
            self._set_image_store(value)
            if beam_type is not None:
                self._beam_parameters[beam_type].resolution = tuple(value)
            return

        if key in ("dwell_time", "stigmation", "preset",
                   "detector_mode", "detector_brightness", "detector_contrast",
                   "beam_enabled", "eucentric_height", "column_tilt",
                   "plasma", "plasma_gas"):
            logging.debug(f"Setting {key} is not supported directly on ZEISS; ignored.")
            return

        logging.warning(f"Unknown/unsupported set key: {key}, value: {value} ({beam_type})")

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def home(self) -> None:
        logging.warning("No homing available via API; please use the native SmartSEM UI.")
