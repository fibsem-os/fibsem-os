"""Stub odemis modules so fibsem.fm.odemis can be imported and unit-tested
without an odemis installation or a running backend.

Install with install_odemis_stubs() BEFORE importing fibsem.fm.odemis (or
fibsem.microscopes.odemis_microscope). The stubs mimic the odemis conventions
that matter to the driver:

- excitation/emission/power stream VAs in SI units (bands in metres, power in
  watts), choices as unordered sets
- MD_PIXEL_SIZE includes binning (the backend recomputes it on binning change)
- the camera resolution VA rescales when binning changes (AOI preserved)
- focuser favourite positions via MD_FAV_POS_ACTIVE/DEACTIVE

See docs/design/odemis-fm-driver.md for the verified behaviour these model.
"""

import sys
import types
from typing import Dict, Optional

import numpy as np

# metadata keys: only identity/equality matters, values mirror odemis for clarity
MD_PIXEL_SIZE = "Pixel size"
MD_BASELINE = "Baseline value"
MD_FAV_POS_ACTIVE = "Favourite position active"
MD_FAV_POS_DEACTIVE = "Favourite position deactive"
MD_ACQ_DATE = "Acquisition date"
MD_EXP_TIME = "Exposure time"
BAND_PASS_THROUGH = "pass-through"


class FakeDataArray(np.ndarray):
    """odemis model.DataArray stand-in: an ndarray carrying a metadata dict."""

    def __new__(cls, array, metadata=None):
        obj = np.asarray(array).view(cls)
        obj.metadata = dict(metadata) if metadata else {}
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.metadata = getattr(obj, "metadata", {})

ODEMIS_MODULE_NAMES = (
    "odemis",
    "odemis.model",
    "odemis.acq",
    "odemis.acq.acqmng",
    "odemis.acq.stream",
    "odemis.util",
    "odemis.util.dataio",
    "odemis.util.fluo",
)

# fibsem modules bound to the odemis import; must be re-imported against stubs
FIBSEM_ODEMIS_MODULE_NAMES = (
    "fibsem.fm.odemis",
    "fibsem.microscopes.odemis_microscope",
)


def band_ex(centre_nm: float) -> tuple:
    """Excitation band 5-tuple in metres (99%low, 25%low, centre, 25%high, 99%high)."""
    c = centre_nm * 1e-9
    return (c - 10e-9, c - 5e-9, c, c + 5e-9, c + 10e-9)


def band_em(bottom_nm: float, width_nm: float = 50.0) -> tuple:
    """Emission band 2-tuple in metres (bottom, top)."""
    b = bottom_nm * 1e-9
    return (b, b + width_nm * 1e-9)


class FakeVA:
    """Minimal odemis VigilantAttribute stand-in with an optional setter hook."""

    def __init__(self, value, choices=None, range=None, unit=None, setter=None):
        self.choices = choices
        self.range = range
        self.unit = unit
        self._setter = setter
        self._value = value
        self.set_count = 0  # number of assignments, for skip-if-unchanged tests

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self.set_count += 1
        if self._setter is not None:
            result = self._setter(new_value)
            if result is not None:
                new_value = result
        self._value = new_value

    def subscribe(self, listener, init=False):
        pass

    def unsubscribe(self, listener):
        pass


class FakeFuture:
    def __init__(self, result=None):
        self._result = result

    def result(self, timeout=None):
        return self._result


class FakeDataflow:
    def __init__(self, camera):
        self._camera = camera
        self.listeners = []

    def subscribe(self, listener):
        self.listeners.append(listener)

    def unsubscribe(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def get(self, asap=True):
        res = self._camera.resolution.value
        return np.zeros(res[::-1], dtype=np.uint16)

    def push(self, data):
        """Deliver a frame to all subscribers, like the camera driver would."""
        for listener in list(self.listeners):
            listener(self, data)


class FakeFocuser:
    """Focus actuator with favourite positions and a z axis."""

    def __init__(
        self,
        active_z: float = 8.0e-3,
        deactive_z: float = -1.0e-3,
        position_z: Optional[float] = None,
        axis_range: tuple = (-2.0e-3, 10.0e-3),
    ):
        self._metadata = {
            MD_FAV_POS_ACTIVE: {"z": active_z},
            MD_FAV_POS_DEACTIVE: {"z": deactive_z},
        }
        z = deactive_z if position_z is None else position_z
        self.position = FakeVA({"z": z})
        self.axes = {"z": types.SimpleNamespace(range=axis_range)}

    def getMetadata(self):
        return self._metadata

    def moveAbs(self, pos: Dict[str, float]):
        self.position.value = dict(self.position.value, **pos)
        return FakeFuture()

    def moveRel(self, shift: Dict[str, float]):
        new = {k: self.position.value[k] + v for k, v in shift.items()}
        self.position.value = dict(self.position.value, **new)
        return FakeFuture()


class FakeCamera:
    """Digital camera whose binning couples resolution + MD_PIXEL_SIZE.

    Mirrors the real behaviour: the resolution VA rescales to preserve the AOI
    (driver-side) and the backend MetadataUpdater recomputes MD_PIXEL_SIZE
    whenever binning changes.
    """

    def __init__(
        self,
        resolution: tuple = (2048, 2048),
        pixel_size: tuple = (1e-7, 1e-7),
        baseline: int = 100,
        sensor_pixel_size: tuple = (6.5e-6, 6.5e-6),
    ):
        self.resolution = FakeVA(tuple(resolution))
        self.exposureTime = FakeVA(0.1, range=(1e-3, 10.0), unit="s")
        self.binning = FakeVA(
            (1, 1), range=((1, 1), (16, 16)), setter=self._on_binning
        )
        self.pixelSize = FakeVA(tuple(sensor_pixel_size), unit="m")  # sensor
        self._metadata = {
            MD_PIXEL_SIZE: tuple(pixel_size),
            MD_BASELINE: baseline,
        }
        self.data = FakeDataflow(self)

    def getMetadata(self):
        return self._metadata

    def _on_binning(self, new_binning):
        old_binning = self.binning.value
        factor = (new_binning[0] / old_binning[0], new_binning[1] / old_binning[1])
        self.resolution.value = tuple(
            int(round(r / f)) for r, f in zip(self.resolution.value, factor)
        )
        px = self._metadata[MD_PIXEL_SIZE]
        self._metadata[MD_PIXEL_SIZE] = tuple(p * f for p, f in zip(px, factor))
        return new_binning


class FakeLens:
    def __init__(self, magnification: float = 84.0, numerical_aperture: float = 0.85):
        self.magnification = FakeVA(magnification)
        self.numericalAperture = FakeVA(numerical_aperture)


class FakeLight:
    def __init__(self, max_power: float = 0.4):
        # one power entry per source; the stream slices out the selected channel
        self.power = FakeVA([0.0], range=((0.0,), (max_power,)), unit="W")
        self.spectra = FakeVA(
            [band_ex(365), band_ex(450), band_ex(550), band_ex(635)], unit="m"
        )


class FakeFilter:
    def __init__(self):
        self.axes = {"band": types.SimpleNamespace(choices={})}


class FakeFluoStream:
    """FluoStream stand-in: excitation/emission/power VAs in SI units."""

    def __init__(self, name, detector, dataflow, emitter, em_filter, focuser=None, **kwargs):
        self.name = name
        self._detector = detector
        self._emitter = emitter
        self._em_filter = em_filter
        self._focuser = focuser

        excitations = [band_ex(365), band_ex(450), band_ex(550), band_ex(635)]
        self.excitation = FakeVA(
            excitations[2], choices=frozenset(excitations), unit="m"
        )
        emissions = [
            BAND_PASS_THROUGH,
            band_em(420),
            band_em(500),
            band_em(590),
            band_em(680),
        ]
        self.emission = FakeVA(emissions[2], choices=frozenset(emissions), unit="m")

        max_power = emitter.power.range[1][0] if emitter is not None else 0.4
        self.power = FakeVA(0.0, range=(0.0, max_power), unit="W")
        self.is_active = FakeVA(False)


def fake_acquire(streams, settings_obs=None):
    """acqmng.acquire stand-in: future resolving to ([DataArray], None).

    The returned frame carries per-exposure metadata like a real odemis
    DataArray (pixel size, acquisition date, exposure time).
    """
    detector = streams[0]._detector
    res = detector.resolution.value
    data = FakeDataArray(
        np.zeros(res[::-1], dtype=np.uint16),
        metadata={
            MD_PIXEL_SIZE: detector.getMetadata()[MD_PIXEL_SIZE],
            MD_ACQ_DATE: 1_780_000_000.0,  # fixed timestamp for determinism
            MD_EXP_TIME: detector.exposureTime.value,
        },
    )
    return FakeFuture(([data], None))


def _band_center(band) -> float:
    """Centre of a 2-tuple (mean) or 5-tuple (index 2) band, like odemis fluo.get_center."""
    if len(band) == 5:
        return band[2]
    return sum(band) / len(band)


def fake_get_one_band_em(bands, ex_band):
    """odemis.util.fluo.get_one_band_em stand-in.

    Faithful to the real logic: pick the band whose centre is closest above
    the excitation centre; fall back to the highest band if none is above.
    """
    if isinstance(bands, str):
        return bands
    ex_center = 1e9 if isinstance(ex_band, str) else _band_center(ex_band)
    centers = {tuple(b): _band_center(b) for b in bands}
    above = [b for b, c in centers.items() if c > ex_center]
    if above:
        return min(above, key=centers.get)
    return max(centers, key=centers.get)


def default_components() -> Dict[str, object]:
    """Fresh set of stub hardware components keyed by odemis role."""
    return {
        "focus": FakeFocuser(),
        "ccd": FakeCamera(),
        "light": FakeLight(),
        "filter": FakeFilter(),
        "lens": FakeLens(),
    }


# registry read by the stub model.getComponent; swap per-test via use_components()
_COMPONENTS: Dict[str, object] = {}


def use_components(components: Dict[str, object]) -> Dict[str, object]:
    """Set the components returned by the stub model.getComponent."""
    _COMPONENTS.clear()
    _COMPONENTS.update(components)
    return components


def _get_component(role: str):
    try:
        return _COMPONENTS[role]
    except KeyError:
        raise LookupError(f"No stub component with role '{role}'")


def install_odemis_stubs() -> None:
    """Insert stub odemis modules into sys.modules (idempotent)."""
    odemis_pkg = types.ModuleType("odemis")
    odemis_pkg.__path__ = []  # mark as package

    model_mod = types.ModuleType("odemis.model")
    model_mod.MD_PIXEL_SIZE = MD_PIXEL_SIZE
    model_mod.MD_BASELINE = MD_BASELINE
    model_mod.MD_FAV_POS_ACTIVE = MD_FAV_POS_ACTIVE
    model_mod.MD_FAV_POS_DEACTIVE = MD_FAV_POS_DEACTIVE
    model_mod.MD_ACQ_DATE = MD_ACQ_DATE
    model_mod.MD_EXP_TIME = MD_EXP_TIME
    model_mod.BAND_PASS_THROUGH = BAND_PASS_THROUGH
    model_mod.Actuator = FakeFocuser
    model_mod.DigitalCamera = FakeCamera
    model_mod.Emitter = FakeLight
    model_mod.DataArray = FakeDataArray
    model_mod.getComponent = lambda role: _get_component(role)
    model_mod.hasVA = lambda comp, name: isinstance(getattr(comp, name, None), FakeVA)

    acq_pkg = types.ModuleType("odemis.acq")
    acq_pkg.__path__ = []

    acqmng_mod = types.ModuleType("odemis.acq.acqmng")
    acqmng_mod.acquire = fake_acquire

    stream_mod = types.ModuleType("odemis.acq.stream")
    stream_mod.FluoStream = FakeFluoStream

    util_pkg = types.ModuleType("odemis.util")
    util_pkg.__path__ = []

    dataio_mod = types.ModuleType("odemis.util.dataio")
    dataio_mod.open_acquisition = lambda path: []

    fluo_mod = types.ModuleType("odemis.util.fluo")
    fluo_mod.get_one_band_em = fake_get_one_band_em
    fluo_mod.get_center = _band_center

    odemis_pkg.model = model_mod
    odemis_pkg.acq = acq_pkg
    odemis_pkg.util = util_pkg
    acq_pkg.acqmng = acqmng_mod
    acq_pkg.stream = stream_mod
    util_pkg.dataio = dataio_mod
    util_pkg.fluo = fluo_mod

    modules = {
        "odemis": odemis_pkg,
        "odemis.model": model_mod,
        "odemis.acq": acq_pkg,
        "odemis.acq.acqmng": acqmng_mod,
        "odemis.acq.stream": stream_mod,
        "odemis.util": util_pkg,
        "odemis.util.dataio": dataio_mod,
        "odemis.util.fluo": fluo_mod,
    }
    sys.modules.update(modules)


def remove_odemis_stubs() -> None:
    """Remove stub odemis modules and the fibsem modules bound to them."""
    for name in ODEMIS_MODULE_NAMES + FIBSEM_ODEMIS_MODULE_NAMES:
        sys.modules.pop(name, None)
