"""A recording fake of JEOL's external control library, and a JeolMicroscope over it.

JEOL's library is not public, and no signatures for it are available: this fake knows
only the calls the old-API driver (``fibsem/microscopes/cryolameller.py``) made, with
the arguments and returns it used. Every return shape is a guess from that code until
the instrument confirms it. A call the fake does not know raises, so the driver can't
call something nothing has seen.

The fake keeps state: magnifications, beam modes, the FIB scan type, the stage (in
JEOL's units: read in metres and radians, written in millimetres and degrees, as the
old driver had it), and the process queue, which runs down one second per look.

:func:`connect` builds the microscope the way the app does, through
``connect_to_microscope``, with the library loader patched to return the fake and the
driver's waits skipped.
"""

from __future__ import annotations

import copy
import enum
import io
import math
import os
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fibsem import config as cfg
from fibsem import utils
from fibsem.drivers.jeol import microscope as jeol_microscope
from fibsem.drivers.jeol.connection import JeolConnection, JeolSdk
from fibsem.drivers.jeol.microscope import JeolMicroscope
from fibsem.structures.system import SystemSettings

# -- the library's enums and models -------------------------------------------------


class EViewMode(enum.IntEnum):
    SINGLE_VIEW = 0
    DUAL_VIEW = 1


class EViewType(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


# 381 is the FIB and 388-398 the SEM's signals; only FIB and SEM_LED are named in the
# old driver, the rest are made up.
EViewSignal = enum.IntEnum(
    "EViewSignal",
    {
        "FIB": 381,
        **{
            name: 388 + i
            for i, name in enumerate(
                [
                    "SEM_LED",
                    "SEM_UED",
                    "SEM_SED",
                    "SEM_BED",
                    "SEM_LABE",
                    "SEM_UBE",
                    "SEM_STEM",
                    "SEM_CL",
                    "SEM_EDS",
                    "SEM_MIX",
                    "SEM_AUX",
                ]
            )
        },
    },
)


class EStageType(enum.IntEnum):
    BSS = 0


class EStageAxisFlags(enum.IntFlag):
    X = 1
    Y = 2
    Z = 4
    T = 8
    R = 16
    FineZ = 32


class EProcessMode(enum.IntEnum):
    MILL = 0x01
    RAPID = 0x02
    DEPO = 0x04


class EProcessLineType(enum.IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class EProcessEndside(enum.IntEnum):
    Top = 0
    Bottom = 1
    Left = 2
    Right = 3


class ErrorStatus(enum.IntEnum):
    SUCCESS = 0
    FAILURE = 1


class ProcessItemData:
    """A process item: what it was built with, and the time JEOL gives it."""

    def __init__(self, kind: str, **fields: Any):
        self.kind = kind
        self.fields = fields
        # one second per item, so estimates are easy to check
        self.process_time = 1.0

    @classmethod
    def area(cls, **fields: Any) -> "ProcessItemData":
        return cls("area", **fields)

    @classmethod
    def line(cls, **fields: Any) -> "ProcessItemData":
        return cls("line", **fields)

    @classmethod
    def spot(cls, **fields: Any) -> "ProcessItemData":
        return cls("spot", **fields)

    def __repr__(self) -> str:
        return f"ProcessItemData.{self.kind}({self.fields})"


def _png_payload(width: int, height: int) -> bytes:
    """A display image payload: a header, then an RGBA PNG."""
    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 4), 128, dtype=np.uint8), "RGBA").save(
        buffer, "PNG"
    )
    return b"JEOLHDR\x00" + buffer.getvalue()


# -- the clients ---------------------------------------------------------------------


class _Client:
    """Records every call; a call the fake has no method for raises."""

    def __init__(self, fake: "FakeJeol", name: str):
        self._fake = fake
        self._name = name

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self._fake.log.append((self._name, method, args, kwargs))


def _recorded(method):
    def wrapper(self, *args, **kwargs):
        self._record(method.__name__, *args, **kwargs)
        return method(self, *args, **kwargs)

    wrapper.__name__ = method.__name__
    return wrapper


class FakeFib(_Client):
    def __init__(self, fake: "FakeJeol", ip: str, port: int, loglevel: str):
        super().__init__(fake, "fib")
        self.address = (ip, port)
        self.magnification = 1000
        self.beam_mode = 10  # 30 pA
        self.column_voltage = 30000
        self.scan_rotation = 0
        self.fine_shift = [0.0, 0.0]  # mm
        self.ol_voltage = 16864
        self.gun_ht = True
        self.contrast = 0
        self.brightness = 0
        self.scan_types = {0: 1, 1: 7, 2: 10, 3: 10}  # memory -> scan type
        self.view_signals = {
            EViewType.LEFT: EViewSignal.SEM_LED,
            EViewType.RIGHT: EViewSignal.FIB,
        }
        self.selected_view = EViewType.LEFT
        # metres and radians
        self.stage = {"x": 0.0, "y": 0.0, "fz": 0.0, "rz": 0.0, "rx": 0.0}
        self.items: List[ProcessItemData] = []
        self.process_state = "Exit"
        self.remaining = 0.0
        self.executable = True

    @_recorded
    def connect(self):
        return True

    @_recorded
    def close(self):
        return None

    # views
    @_recorded
    def setViewMode(self, mode):
        return 0

    @_recorded
    def setViewSignal(self, view, signal):
        self.view_signals[EViewType(view)] = EViewSignal(signal)
        return 0

    @_recorded
    def getViewSignal(self, view):
        return 0, int(self.view_signals[EViewType(view)])

    @_recorded
    def setSelectedView(self, view):
        self.selected_view = view
        return 0

    # scan
    @_recorded
    def setFibScanType(self, mode, scan_type):
        self.scan_types[mode] = scan_type
        return 0

    @_recorded
    def getFibScanType(self, mode):
        from fibsem.drivers.jeol.devices import FIB_SCAN_TYPES

        scan_type = self.scan_types[mode]
        (width, height), dwell, _ = FIB_SCAN_TYPES[scan_type]
        return 0, scan_type, dwell * width * height, width, height

    @_recorded
    def startFibOneShotScan(self, mode):
        return 0

    @_recorded
    def getDispImage(self, flag):
        from fibsem.drivers.jeol.devices import FIB_SCAN_TYPES

        (width, height), _, _ = FIB_SCAN_TYPES[self.scan_types[self._fake.fib_mode()]]
        return 0, _png_payload(width, height)

    @_recorded
    def setFibFreezeMode(self, mode):
        return 0

    # beam
    @_recorded
    def getFibMagnification(self):
        return 0, self.magnification

    @_recorded
    def setFibMagnification(self, magnification):
        self.magnification = magnification
        return 0

    @_recorded
    def getFibBeamMode(self):
        return 0, self.beam_mode

    @_recorded
    def setFibBeamMode(self, mode):
        self.beam_mode = mode
        return 0

    @_recorded
    def getFibColumnVoltage(self):
        return 0, self.column_voltage

    @_recorded
    def setFibColumnVoltage(self, voltage):
        self.column_voltage = voltage
        return 0

    @_recorded
    def getFibScanRotation(self):
        return 0, self.scan_rotation

    @_recorded
    def setFibScanRotation(self, degrees):
        self.scan_rotation = degrees
        return 0

    @_recorded
    def getFibFineShiftX(self):
        return 0, self.fine_shift[0]

    @_recorded
    def getFibFineShiftY(self):
        return 0, self.fine_shift[1]

    @_recorded
    def setFibFineShiftXAbs(self, value):
        self.fine_shift[0] = value
        return 0

    @_recorded
    def setFibFineShiftYAbs(self, value):
        self.fine_shift[1] = value
        return 0

    @_recorded
    def getFibOlVoltage(self):
        return 0, self.ol_voltage

    @_recorded
    def setFibOlVoltage(self, voltage):
        self.ol_voltage = voltage
        return 0

    @_recorded
    def getFibGunHtSw(self):
        return 0, self.gun_ht

    @_recorded
    def setFibGunHtSw(self, on):
        self.gun_ht = on
        return 0

    @_recorded
    def getFibContrast(self):
        return 0, self.contrast

    @_recorded
    def setFibContrast(self, value):
        self.contrast = value
        return 0

    @_recorded
    def getFibBrightness(self):
        return 0, self.brightness

    @_recorded
    def setFibBrightness(self, value):
        self.brightness = value
        return 0

    # stage
    @_recorded
    def getStagePosition(self):
        return 0, dict(self.stage)

    @_recorded
    def setMultiAxisStagePositionAbs(self, stage_type, flags, x, y, t, r, fz):
        self.stage = {
            "x": x * 1e-3,
            "y": y * 1e-3,
            "fz": fz * 1e-3,
            "rz": math.radians(r),
            "rx": math.radians(t),
        }
        return 0

    @_recorded
    def getStageMoveStatus(self):
        return 0, False

    # process
    @_recorded
    def deleteProcessGroup(self):
        self.items.clear()
        return 0

    @_recorded
    def addProcessItem(self, item):
        self.items.append(item)
        return 0

    @_recorded
    def getProcessExecutable(self):
        return 0, self.executable

    @_recorded
    def getProcessStatus(self):
        return 0, 0, self.process_state

    @_recorded
    def startProcess(self):
        if self.process_state != "Pause":
            self.remaining = sum(item.process_time for item in self.items)
        self.process_state = "Start"
        return 0

    @_recorded
    def stopProcess(self):
        self.process_state = "Stop"
        self.remaining = 0.0
        return 0

    @_recorded
    def pauseProcess(self):
        self.process_state = "Pause"
        return 0

    @_recorded
    def getRemainProcessTime(self):
        # the queue runs down a second per look
        if self.process_state == "Start":
            self.remaining = max(0.0, self.remaining - 1.0)
            if self.remaining == 0:
                self.process_state = "Exit"
        return 0, self.remaining

    @_recorded
    def getProcessItemCount(self):
        return 0, len(self.items)

    @_recorded
    def getProcessItem(self, index):
        return 0, self.items[index]


class FakeSem(_Client):
    def __init__(self, fake: "FakeJeol", ip: str, port: int, loglevel: str):
        super().__init__(fake, "sem")
        self.address = (ip, port)
        self.gun_valve = 1
        self.ht = 1
        self.blanking = 0
        self.objective_lens = 552960
        self.probe = 5  # 0.1 nA
        self.kv = 5.0
        self.magnification = 1000
        self.scan_rotation = 0
        self.scan_rotation_switch = 0
        self.image_shift = [0, 0]  # DAC
        self.stigmator = [0, 0]
        self.photo_size = 2
        self.rate_no = 5
        self.live_freeze = 1
        self.contrast = 2048
        self.brightness = 2048

    @_recorded
    def connect_sem(self):
        return True

    @_recorded
    def close(self):
        return None

    @_recorded
    def getGunValve(self):
        return 0, self.gun_valve

    @_recorded
    def setHtOnOff(self, on):
        self.ht = on
        return 0

    @_recorded
    def getBeamBlanking(self):
        return 0, self.blanking

    @_recorded
    def setBeamBlanking(self, value):
        self.blanking = value
        return 0

    @_recorded
    def getObjectLens(self):
        return 0, self.objective_lens

    @_recorded
    def setObjectLensAbs(self, dac):
        self.objective_lens = dac
        return 0

    @_recorded
    def getProbeNo(self):
        return 0, self.probe

    @_recorded
    def setProbeNo(self, probe):
        self.probe = probe
        return 0

    @_recorded
    def getAcceleratingVoltage(self):
        return 0, self.kv

    @_recorded
    def setAcceleratingVoltage(self, kv):
        self.kv = kv
        return 0

    @_recorded
    def getMagnification(self):
        return 0, self.magnification

    @_recorded
    def setMagnification(self, magnification):
        self.magnification = magnification
        return 0

    @_recorded
    def getScanRotationAngle(self):
        return 0, self.scan_rotation

    @_recorded
    def setScanRotationAngle(self, degrees):
        self.scan_rotation = degrees
        return 0

    @_recorded
    def setScanRotationSwitch(self, on):
        self.scan_rotation_switch = on
        return 0

    @_recorded
    def getImageShiftX(self):
        return 0, self.image_shift[0]

    @_recorded
    def getImageShiftY(self):
        return 0, self.image_shift[1]

    @_recorded
    def setAbsoluteImageShiftX(self, dac):
        self.image_shift[0] = dac
        return 0

    @_recorded
    def setAbsoluteImageShiftY(self, dac):
        self.image_shift[1] = dac
        return 0

    @_recorded
    def getOlStigmatorX(self):
        return 0, self.stigmator[0]

    @_recorded
    def getOlStigmatorY(self):
        return 0, self.stigmator[1]

    @_recorded
    def setPhotoScanMode2(self, photo_type, scan_mode):
        return 0

    @_recorded
    def setPhotoImageSize(self, index):
        self.photo_size = index
        return 0

    @_recorded
    def setScanMode(self, mode):
        return 0

    @_recorded
    def getImageStatus(self):
        return 0, {"RATE_NO": str(self.rate_no), "LIVE_FREEZE": str(self.live_freeze)}

    @_recorded
    def setScanSpeed(self, mode, speed, flag):
        self.rate_no = speed
        return 0

    @_recorded
    def getSignalContrastBrightness(self, channel):
        return 0, {"CONTRAST": str(self.contrast), "BRIGHTNESS": str(self.brightness)}

    @_recorded
    def setContrast(self, value):
        self.contrast = value
        return 0

    @_recorded
    def setBrightness(self, value):
        self.brightness = value
        return 0

    @_recorded
    def setPhotoIntegration(self, on):
        return 0

    @_recorded
    def setLive(self):
        self.live_freeze = 0
        return 0

    @_recorded
    def setFreeze(self):
        self.live_freeze = 1
        return 0

    @_recorded
    def execExternalPhoto2(self, base_path, file_type, flag):
        from fibsem.drivers.jeol.devices import SEM_PHOTO_SIZES

        size = {index: frame for frame, index in SEM_PHOTO_SIZES.items()}[
            self.photo_size
        ]
        width, height = size
        Image.fromarray(np.full((height, width), 100, dtype=np.uint8), "L").save(
            base_path + ".tif"
        )
        return 0

    @_recorded
    def execACB(self):
        return 0

    @_recorded
    def execAutoFocus(self, mode):
        return 0


class FakeSemLive:
    """``sem_live``: the SEM's live buffers, read as a fixed frame."""

    def __init__(self, fake: "FakeJeol"):
        self._fake = fake

    def find_pid(self):
        return 1234

    def get_handle(self, pid):
        return object()

    def locate_buffers(self, handle):
        return [(0x1000, 1280 * 960), (0x2000, 640 * 480)]

    def read_image(self, handle, address, size):
        shape = (960, 1280) if address == 0x1000 else (480, 640)
        return np.full(shape, 50, dtype=np.uint8)


class FakeJeol:
    """The fake library, its clients once connected, and the call log."""

    def __init__(self):
        self.log: List[Tuple[str, str, tuple, dict]] = []
        self.fib: Optional[FakeFib] = None
        self.sem: Optional[FakeSem] = None
        self.connection: Optional[JeolConnection] = None

    def fib_mode(self) -> int:
        return self.connection.fib_scan_mode if self.connection else 0

    def _fib_client(self, ip, port, loglevel):
        self.fib = FakeFib(self, ip, port, loglevel)
        return self.fib

    def _sem_client(self, ip, port, loglevel):
        self.sem = FakeSem(self, ip, port, loglevel)
        return self.sem

    def sdk(self) -> JeolSdk:
        command = types.SimpleNamespace(
            JeolFibExternalControlClient=self._fib_client,
            EViewMode=EViewMode,
            EViewType=EViewType,
            EViewSignal=EViewSignal,
            EStageType=EStageType,
            EStageAxisFlags=EStageAxisFlags,
        )
        ecommand = types.SimpleNamespace(PcSemExternalControlClient=self._sem_client)
        models = types.SimpleNamespace(
            ProcessItemData=ProcessItemData,
            EProcessMode=EProcessMode,
            EProcessLineType=EProcessLineType,
            EProcessEndside=EProcessEndside,
        )
        common = types.SimpleNamespace(ErrorStatus=ErrorStatus)
        return JeolSdk(
            common=common,
            command=command,
            ecommand=ecommand,
            models=models,
            sem_live=FakeSemLive(self),
        )

    def calls(self, client: Optional[str] = None) -> List[str]:
        return [method for name, method, _, _ in self.log if client in (None, name)]


# The steps JEOL sets each beam parameter to, as a fraction of a value the contract
# suite writes: a 12-bit contrast and brightness DAC, the objective lens DAC and OL
# voltage for the working distance, 18.6 nm SEM shift steps, whole-degree rotation.
STEP_TOLERANCES: Dict[str, float] = {
    "detector_contrast": 1e-3,
    "detector_brightness": 1e-3,
    "working_distance": 1e-3,
    "shift": 1e-2,
    "scan_rotation": 1e-2,
}


# The SEM's on is read from the gun valve and written as the high tension, as the
# old driver had it: whether switching the HT moves the valve is a hardware check
# (port-plan 3.1), so turning it off does not read back off here.
_SEM_ON = "the SEM's on reads the gun valve but writes the HT: check on hardware"
# Every detector read and write first sets the view layout, as the old driver did,
# and the layout puts SEM_LED in the SEM's view: another SEM detector set does not
# read back (Patrick 2026-10-14: keep the layout as it was).
_SEM_LAYOUT = "the view layout puts the SEM back on SEM_LED before each read"
KNOWN_GAPS: Dict[str, str] = {
    "test_on_off_and_blanking[BeamType.ELECTRON]": _SEM_ON,
    "test_beam_key_round_trips[BeamType.ELECTRON-on]": _SEM_ON,
    "test_beam_key_round_trips[BeamType.ELECTRON-detector_type]": _SEM_LAYOUT,
    "test_every_choice_can_be_set[BeamType.ELECTRON-detector_type]": _SEM_LAYOUT,
    "test_beam_wrappers_match_their_key[BeamType.ELECTRON-get_detector_type-"
    "set_detector_type-detector_type]": _SEM_LAYOUT,
}


def jeol_system(electron: bool = True, ion: bool = True) -> SystemSettings:
    """The shipped JEOL configuration, with either column disabled."""
    system = utils.load_microscope_configuration(
        os.path.join(cfg.SHIPPED_CONFIG_PATH, "jeol-configuration.yaml")
    ).system
    system.electron.enabled = electron
    system.ion.enabled = ion
    return system


def install(monkeypatch, fake: Optional[FakeJeol] = None) -> FakeJeol:
    """Patch the driver so a connect, however it is made, connects to ``fake``."""
    fake = fake or FakeJeol()
    monkeypatch.setattr(jeol_microscope, "load_sdk", lambda api_path=None: fake.sdk())
    monkeypatch.setattr(JeolConnection, "wait", lambda self, seconds: None)
    return fake


def connect(
    monkeypatch,
    system: Optional[SystemSettings] = None,
    fake: Optional[FakeJeol] = None,
):
    """A JeolMicroscope as connect leaves it, over ``fake``. Returns (microscope, fake)."""
    fake = install(monkeypatch, fake)
    microscope = JeolMicroscope(copy.deepcopy(system or jeol_system()))
    microscope.connect_to_microscope(ip_address="localhost")
    fake.connection = microscope._connection
    fake.log.clear()
    return microscope, fake
