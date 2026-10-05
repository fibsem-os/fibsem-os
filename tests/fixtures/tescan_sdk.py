"""A recording fake of the ``tescanautomation`` connection, and a TescanMicroscope over it.

The fake answers what the driver asks of SharkSEM and logs every call under its vendor
path (``Stage.MoveTo``, ``FIB.Optics.GetImageRotation``, ...). The stage moves: a
``MoveTo`` writes the axes it was given, so the read-back after a move sees the move.
The stage holds Tescan's own units (mm and degrees), as the instrument reports them.

:func:`connect` builds the microscope the way the app does, through
``connect_to_microscope``, with ``Automation`` patched to return the fake. Whatever
connect builds (devices, routes) the fake is underneath it, so a test written against
this keeps working while the driver moves onto devices.

:func:`image_at` makes the image the app would have acquired at the fake's current
stage, through the driver's own header parse and metadata stamp. Its header carries the
stage in metres and degrees, as a Tescan image header does.

No ``tescanautomation`` install is needed.
"""

import copy
from typing import List, Optional, Sequence

import numpy as np

from fibsem.microscopes import tescan as tescan_module
from fibsem.microscopes.tescan import TescanMicroscope, fromTescanImage
from fibsem.structures import BeamType, FibsemImage, ImageSettings, SystemSettings


def _plain(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (int, str, bool, type(None))):
        return value
    if isinstance(value, FakeDetector):
        return value.name
    return repr(value)


class FakeDetector:
    """Stands in for ``tescanautomation.Common.Detector``."""

    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class Node:
    """Any vendor path: records calls, returns a child for any attribute."""

    def __init__(self, sdk: "FakeTescan", path: str):
        self._sdk = sdk
        self._path = path
        self._kids = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._kids:
            self._kids[name] = Node(self._sdk, f"{self._path}.{name}")
        return self._kids[name]

    def __call__(self, *args, **kwargs):
        self._sdk.record(self._path, args, kwargs)
        return None


class FakeStage(Node):
    def __init__(self, sdk: "FakeTescan"):
        super().__init__(sdk, "Stage")
        # x, y, z (mm), rotation, tilt (degrees)
        self.position: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

    def GetPosition(self):
        self._sdk.record("Stage.GetPosition", (), {})
        return tuple(self.position)

    def MoveTo(self, x=None, y=None, z=None, rot=None, tiltx=None, tilty=None):
        self._sdk.record(
            "Stage.MoveTo", (), {"x": x, "y": y, "z": z, "rot": rot, "tiltx": tiltx}
        )
        for i, value in enumerate((x, y, z, rot, tiltx)):
            if value is not None:
                self.position[i] = float(value)

    def IsCalibrated(self):
        self._sdk.record("Stage.IsCalibrated", (), {})
        return True


class FakeOptics(Node):
    def __init__(self, sdk: "FakeTescan", path: str):
        super().__init__(sdk, path)
        self.image_rotation = 0.0  # degrees
        self.image_shift = (0.0, 0.0)  # mm

    def GetImageRotation(self):
        self._sdk.record(f"{self._path}.GetImageRotation", (), {})
        return self.image_rotation

    def GetImageShift(self):
        self._sdk.record(f"{self._path}.GetImageShift", (), {})
        return self.image_shift

    def SetImageShift(self, x, y):
        self._sdk.record(f"{self._path}.SetImageShift", (x, y), {})
        self.image_shift = (float(x), float(y))


class FakeDetectors(Node):
    def __init__(self, sdk: "FakeTescan", path: str, names: Sequence[str]):
        super().__init__(sdk, path)
        self.detectors = [FakeDetector(name, i) for i, name in enumerate(names)]

    def Enum(self):
        self._sdk.record(f"{self._path}.Enum", (), {})
        return list(self.detectors)


class FakeColumn(Node):
    def __init__(self, sdk: "FakeTescan", path: str, detectors: Sequence[str]):
        super().__init__(sdk, path)
        self.Optics = FakeOptics(sdk, f"{path}.Optics")
        self.Detector = FakeDetectors(sdk, f"{path}.Detector", detectors)


class FakeTescan(Node):
    """The ``Automation`` connection."""

    def __init__(self):
        self.log: list = []
        super().__init__(self, "connection")
        self.Stage = FakeStage(self)
        self.SEM = FakeColumn(self, "SEM", ["SE", "E-T", "BSE"])
        self.FIB = FakeColumn(self, "FIB", ["SE", "SI"])

    def record(self, path, args, kwargs):
        self.log.append([path, _plain(list(args)), _plain(kwargs)])

    def calls(self, path: str) -> list:
        """The keyword arguments of each call made to ``path``, in order."""
        return [kwargs for p, _, kwargs in self.log if p == path]

    def column(self, beam_type: BeamType) -> FakeColumn:
        return self.SEM if beam_type is BeamType.ELECTRON else self.FIB


def connect(monkeypatch, system: SystemSettings, fake: Optional[FakeTescan] = None):
    """A TescanMicroscope as connect leaves it, over ``fake``. Returns (microscope, fake)."""
    fake = fake or FakeTescan()
    monkeypatch.setattr(tescan_module, "TESCAN_API_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        tescan_module, "Automation", lambda *args, **kwargs: fake, raising=False
    )
    monkeypatch.setattr(tescan_module, "Detector", FakeDetector, raising=False)
    microscope = TescanMicroscope(copy.deepcopy(system))
    microscope.connect_to_microscope(ip_address="localhost", port=8300)
    fake.log.clear()
    return microscope, fake


class _Document:
    def __init__(self, data: np.ndarray, header: dict):
        self.Image = data
        self.Header = header


def image_at(
    microscope: TescanMicroscope,
    fake: FakeTescan,
    beam_type: BeamType,
    pixel_size: float = 1e-7,
    shape=(1024, 1536),
) -> FibsemImage:
    """The image the app would have taken here: header parse plus metadata stamp."""
    x, y, z, r, t = fake.Stage.position
    column = "SEM" if beam_type is BeamType.ELECTRON else "FIB"
    header = {
        "MAIN": {
            "PixelSizeX": pixel_size,
            "PixelSizeY": pixel_size,
            "Date": "2026-10-05",
            "Time": "12:00:00",
        },
        column: {
            "StageX": x * 1e-3,
            "StageY": y * 1e-3,
            "StageZ": z * 1e-3,
            "StageRotation": r,
            "StageTilt": t,
            "WD": 0.007,
            "HV": 30000.0,
            "PredictedBeamCurrent": 1e-10,
            "DwellTime": 1e-6,
            "ScanRotation": fake.column(beam_type).Optics.image_rotation,
            "StigmatorX": 0.0,
            "StigmatorY": 0.0,
            "ImageShiftX": 0.0,
            "ImageShiftY": 0.0,
            "Detector0": "SE",
            "Detector0Gain": 50.0,
            "Detector0Offset": 50.0,
        },
    }
    document = _Document(np.zeros(shape, dtype=np.uint8), header)
    image = fromTescanImage(
        document,
        ImageSettings(resolution=(shape[1], shape[0]), beam_type=beam_type),
    )
    image.metadata.image_settings.beam_type = beam_type
    microscope._set_additional_metadata(image)
    return image
