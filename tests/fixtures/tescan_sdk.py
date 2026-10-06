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
import types
from typing import List, Optional, Sequence

import numpy as np

from fibsem.microscopes import tescan as tescan_module
from fibsem.microscopes.tescan import TescanMicroscope
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


class _Recorded(Node):
    """A vendor object whose methods record their call before they answer."""

    def _call(self, name, *args, **kwargs):
        self._sdk.record(f"{self._path}.{name}", args, kwargs)


class FakeOptics(_Recorded):
    def __init__(self, sdk: "FakeTescan", path: str):
        super().__init__(sdk, path)
        self.image_rotation = 0.0  # degrees
        self.image_shift = (0.0, 0.0)  # mm
        self.wd = 7.0  # mm
        self.viewfield = 0.15  # mm

    def GetImageRotation(self):
        self._call("GetImageRotation")
        return self.image_rotation

    def SetImageRotation(self, value):
        self._call("SetImageRotation", value)
        self.image_rotation = float(value)

    def GetImageShift(self):
        self._call("GetImageShift")
        return self.image_shift

    def SetImageShift(self, x, y):
        self._call("SetImageShift", x, y)
        self.image_shift = (float(x), float(y))

    def GetWD(self):
        self._call("GetWD")
        return self.wd

    def SetWD(self, value):
        self._call("SetWD", value)
        self.wd = float(value)

    def GetViewfield(self):
        self._call("GetViewfield")
        return self.viewfield

    def SetViewfield(self, value):
        self._call("SetViewfield", value)
        self.viewfield = float(value)


class _Status:
    BeamOn = "BeamOn"
    BeamOff = "BeamOff"


class FakeBeamUnit(_Recorded):
    """``SEM.Beam`` or ``FIB.Beam``: status, current (pA) and voltage (V)."""

    Status = _Status

    def __init__(self, sdk: "FakeTescan", path: str):
        super().__init__(sdk, path)
        self.status = _Status.BeamOn
        self.current = 50.0  # pA
        self.voltage = 30000.0

    def GetStatus(self):
        self._call("GetStatus")
        return self.status

    def On(self):
        self._call("On")
        self.status = _Status.BeamOn

    def Off(self):
        self._call("Off")
        self.status = _Status.BeamOff

    def GetCurrent(self):
        self._call("GetCurrent")
        return self.current

    def SetCurrent(self, value):
        self._call("SetCurrent", value)
        self.current = float(value)

    def ReadProbeCurrent(self):
        self._call("ReadProbeCurrent")
        return self.current

    def GetVoltage(self):
        self._call("GetVoltage")
        return self.voltage

    def SetVoltage(self, value):
        self._call("SetVoltage", value)
        self.voltage = float(value)


class FakeDetectors(_Recorded):
    def __init__(self, sdk: "FakeTescan", path: str, names: Sequence[str]):
        super().__init__(sdk, path)
        self.detectors = [FakeDetector(name, i) for i, name in enumerate(names)]
        self.selected = self.detectors[0]
        self.gain_black = {d.name: (50.0, 40.0) for d in self.detectors}

    def Enum(self):
        self._call("Enum")
        return list(self.detectors)

    def Get(self, Channel=0):
        self._call("Get", Channel=Channel)
        return self.selected

    def Set(self, Channel=0, Detector=None):
        self._call("Set", Channel=Channel, Detector=Detector)
        self.selected = Detector

    def GetGainBlack(self, Detector=None):
        self._call("GetGainBlack", Detector=Detector)
        return self.gain_black[Detector.name]

    def SetGainBlack(self, Detector=None, Gain=None, Black=None):
        self._call("SetGainBlack", Detector=Detector, Gain=Gain, Black=Black)
        self.gain_black[Detector.name] = (float(Gain), float(Black))


class FakePresets(_Recorded):
    def __init__(self, sdk: "FakeTescan", path: str, names: Sequence[str]):
        super().__init__(sdk, path)
        self.names = list(names)

    def Enum(self):
        self._call("Enum")
        return list(self.names)

    def IsAvailable(self, name):
        self._call("IsAvailable", name)
        return name in self.names

    def Activate(self, name):
        self._call("Activate", name)


class FakeScan(_Recorded):
    """``SEM.Scan`` or ``FIB.Scan``: the acquires return a frame with a header for the
    fake's current stage, as the instrument's would."""

    def __init__(self, sdk: "FakeTescan", path: str, beam_type: BeamType):
        super().__init__(sdk, path)
        self.beam_type = beam_type

    def _document(self, width, height):
        return _Document(
            np.full((height, width), 7, dtype=np.uint8),
            header_at(self._sdk, self.beam_type),
        )

    def AcquireImage(self, **kwargs):
        self._call("AcquireImage", **kwargs)
        return self._document(kwargs["Width"], kwargs["Height"])

    def AcquireROI(self, **kwargs):
        self._call("AcquireROI", **kwargs)
        return self._document(
            kwargs["Right"] - kwargs["Left"] + 1, kwargs["Bottom"] - kwargs["Top"] + 1
        )


class FakeColumn(_Recorded):
    def __init__(
        self,
        sdk: "FakeTescan",
        path: str,
        detectors: Sequence[str],
        presets: Sequence[str] = (),
    ):
        super().__init__(sdk, path)
        beam_type = BeamType.ELECTRON if path == "SEM" else BeamType.ION
        self.Scan = FakeScan(sdk, f"{path}.Scan", beam_type)
        self.Optics = FakeOptics(sdk, f"{path}.Optics")
        self.Beam = FakeBeamUnit(sdk, f"{path}.Beam")
        self.Detector = FakeDetectors(sdk, f"{path}.Detector", detectors)
        self.Preset = FakePresets(sdk, f"{path}.Preset", presets)

    def IsBusy(self):
        self._call("IsBusy")
        return False


class FakeTescan(Node):
    """The ``Automation`` connection."""

    def __init__(self):
        self.log: list = []
        super().__init__(self, "connection")
        self.Stage = FakeStage(self)
        self.SEM = FakeColumn(self, "SEM", ["SE", "E-T", "BSE"])
        self.FIB = FakeColumn(
            self, "FIB", ["SE", "SI"], ["30 keV; 1 nA", "30 keV; 100 pA"]
        )
        # The microscope's connection lock, once connected: every call made without
        # it held is listed in `unlocked`.
        self.lock = None
        self.unlocked: list = []

    def record(self, path, args, kwargs):
        self.log.append([path, _plain(list(args)), _plain(kwargs)])
        if self.lock is not None and not self.lock._is_owned():
            self.unlocked.append(path)

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
    monkeypatch.setattr(
        tescan_module,
        "Bpp",
        types.SimpleNamespace(Grayscale_8_bit="Grayscale_8_bit"),
        raising=False,
    )
    microscope = TescanMicroscope(copy.deepcopy(system))
    microscope.connect_to_microscope(ip_address="localhost", port=8300)
    fake.lock = microscope._connection_lock
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
    document = _Document(
        np.zeros(shape, dtype=np.uint8), header_at(fake, beam_type, pixel_size)
    )
    image = microscope._image_from_tescan(
        document,
        ImageSettings(resolution=(shape[1], shape[0]), beam_type=beam_type),
    )
    image.metadata.image_settings.beam_type = beam_type
    microscope._set_additional_metadata(image)
    return image


def header_at(fake: FakeTescan, beam_type: BeamType, pixel_size: float = 1e-7) -> dict:
    """A Tescan image header for the fake's current stage."""
    x, y, z, r, t = fake.Stage.position
    column = "SEM" if beam_type is BeamType.ELECTRON else "FIB"
    return {
        "MAIN": {
            "PixelSizeX": pixel_size,
            "PixelSizeY": pixel_size,
            "Date": "2026-10-05",
            "Time": "12:00:00",
            "DeviceModel": "FAKE",
            "SerialNumber": "0000",
            "SoftwareVersion": "1.0",
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
