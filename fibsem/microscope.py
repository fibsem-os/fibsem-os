from __future__ import annotations

import dataclasses
import datetime
import logging
import threading
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import replace
from enum import Enum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from psygnal import Signal

from fibsem.control import Control
from fibsem.devices.core import IMAGING_CHANNEL, Resources, Session
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.geometry.movement import beam_view_tilt
from fibsem.geometry.orientations import (
    compucentric_rotation,
)
from fibsem.imaging.spot import SpotBurnProgress
from fibsem.imaging.tiling.progress import TiledProgress
from fibsem.milling.progress import MillingProgress
from fibsem.structures import (
    DEFAULT_STAGE_DEVICES,
    DEVICE_AXES,
    STAGE_FRAME_FIBSEM,
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    CameraImageTransform,
    ChamberState,
    DeviceImagingState,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    InsertableDeviceState,
    MicroscopeState,
    MillingState,
    Point,
    StageDeviceSettings,
    SystemSettings,
)

if TYPE_CHECKING:
    from fibsem.cancellation import AnyStopEvent
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.services.imaging import Imaging
    from fibsem.services.milling import Milling
    from fibsem.services.sample_handling import SampleHandling
    from fibsem.services.spot_burn import SpotBurn
    from fibsem.services.stage_movement import StageMovement


# The beam parameters a wrapper may ask for without a beam type; any other needs one.
_BEAM_PARAMETERS_WITHOUT_BEAM = frozenset(("plasma_gas", "preset"))


def _chamber_state_name(state: ChamberState) -> str:
    """The name today's pump() and vent() return for a chamber device's state, as the
    Demo and AutoScript chambers name it ("Pumped", "Vented")."""
    return state.value.capitalize()


def _plain_value(value: Any) -> Any:
    """A beam parameter's value as the wrappers return it: an enum as its plain
    value ("spot")."""
    if isinstance(value, Enum):
        return value.value
    return value


def _moves(translation: FibsemStagePosition) -> bool:
    """Whether a relative move moves anything: an axis set and non-zero."""
    return any(getattr(translation, axis) for axis in DEVICE_AXES)


class RequiredDeviceUnavailable(RuntimeError):
    """A device the configuration marks `required` could not be reached at connect."""


class _PerInstance:
    """A class-level default made once per instance, on first use.

    For state every backend needs whether or not its ``__init__`` calls the base
    class's: each microscope gets its own, made by ``factory(instance)``. Assigning the
    attribute on an instance replaces it there, as with a plain attribute.
    """

    def __init__(self, factory: Callable[[Any], Any]):
        self._factory = factory
        self._name = ""
        self._guard = threading.Lock()

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            return self
        try:
            return instance.__dict__[self._name]
        except KeyError:
            pass
        with self._guard:  # two threads asking first must get the same one
            if self._name not in instance.__dict__:
                instance.__dict__[self._name] = self._factory(instance)
            return instance.__dict__[self._name]


# The names the beams have in `FibsemMicroscope.devices`, which are their
# configuration entries' names.
_BEAM_DEVICE_NAMES: Dict[BeamType, str] = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}


class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""

    # THREADING CONTRACT: these are psygnal Signals — subscribers run synchronously
    # on whatever thread emits (workflow/movement/acquisition workers, not the GUI
    # thread). Any handler that touches Qt or a canvas MUST marshal, e.g. with
    # @superqt.ensure_main_thread; a bare .connect() of a GUI handler is a
    # crash-on-hardware bug that won't reproduce on a dev machine.
    #
    # The two progress signals send on what the milling and spot burn services'
    # ``progress`` parameters report (`bind_milling`, `bind_spot_burn`): report there
    # (`fibsem.services.milling.report_progress`), never emit on them directly.
    milling_progress_signal = Signal(MillingProgress)
    tiled_acquisition_signal = Signal(TiledProgress)
    spot_burn_progress_signal = Signal(SpotBurnProgress)
    system: SystemSettings
    #: The registry name of this class's driver, as its package's
    #: ``DRIVER.manufacturer`` spells it (``"Tescan"``). Each driver's class sets it;
    #: `manufacturer` answers it, and connect code names its own builds with it.
    driver_name: ClassVar[str]
    milling_channel: BeamType = BeamType.ION
    #: The milling service over the beams (`fibsem.services.milling.Milling`); the
    #: milling methods go to it. None when no ion beam was built: they raise then,
    #: a stop has nothing to stop, and the state reads idle.
    milling: Optional[Milling] = None
    #: The stage movement service over the stage and beams
    #: (`fibsem.services.stage_movement.StageMovement`); the view-corrected moves go
    #: to it. None on a backend that doesn't build one yet, whose own move code below
    #: is then in charge.
    stage_movement: Optional[StageMovement] = None
    #: The spot burn service over the beams (`fibsem.services.spot_burn.SpotBurn`);
    #: `run_spot_burn` goes to it. None when the ion beam can't burn (not built,
    #: or it can't blank and park): the method raises then.
    spot_burn: Optional[SpotBurn] = None
    #: The sample handling service over the stage and the sample loader
    #: (`fibsem.services.sample_handling.SampleHandling`): the holder, the grids and
    #: the commands that make a grid reachable. None without a stage device.
    sample_handling: Optional[SampleHandling] = None
    #: The imaging service over the beams (`fibsem.services.imaging.Imaging`):
    #: `auto_focus`, and the focus sweep. None when no beam was built: `auto_focus`
    #: raises then.
    imaging: Optional[Imaging] = None
    #: Whether this driver's sample loader reads its magazine when the sample
    #: handling service is built: only where that read is free (the simulator).
    _read_magazine_at_connect: bool = False
    #: The file `system` was loaded from, when it was loaded from one. Set by
    #: `utils.setup_session`; what a calibration action writes back to.
    configuration_path: Optional[str] = None

    # The views a coincidence correction can be measured in -- the beam_type
    # values vertical_move accepts. The FIB view is universal; the SEM view
    # needs a backend that knows how to slide along the FIB line of sight.
    vertical_move_views: Tuple[BeamType, ...] = (BeamType.ION,)

    # Where a half turn of this backend's stage is centred, raw (x, y) in metres: a
    # position p on one side of the stage is at 2c - p on the other. A property of the
    # hardware, so the driver sets it (adding the configured
    # `rotation_centre_correction` to the centre its instrument reports). None means no rotating stage is
    # known: stage moves and image reprojection assume the stage origin (FIB-1081).
    rotation_centre: Optional[Tuple[float, float]] = None

    # live acquisition
    sem_acquisition_signal = Signal(FibsemImage)
    fib_acquisition_signal = Signal(FibsemImage)
    # The beams whose live_frame already forwards to the signals above.
    _live_forwarded = _PerInstance(lambda _: set())
    # One acquisition at a time on this microscope's imaging view. Devices claim the
    # same lock as the `imaging_channel` resource (`resources` below).
    _threading_lock = _PerInstance(lambda _: threading.RLock())
    # The shared resources this microscope's devices claim. `imaging_channel` is
    # `_threading_lock`, so the old path and the devices exclude each other; every other
    # name shares one lock of its own, the default in `Resources`.
    resources = _PerInstance(
        lambda m: Resources(
            groups={IMAGING_CHANNEL: IMAGING_CHANNEL},
            locks={IMAGING_CHANNEL: m._threading_lock},
        )
    )
    # What this microscope's devices and services report to: events for the
    # experiment's record (`record_event`) and the stamp on each image.
    # Looked up on each call, so a backend or test that replaces either is used.
    session = _PerInstance(
        lambda m: Session(
            record=lambda kind, payload: m.record_event(kind, payload),
            stamp=lambda image: m._set_additional_metadata(image),
        )
    )

    # Who is in control of this microscope (FIB-1270): one action at a time,
    # taken at entry points (a run, a script, a server route); see fibsem.control.
    control = _PerInstance(lambda _: Control())

    # fluorescence: none until a backend builds one on connect
    fm: Optional[FluorescenceMicroscope] = None

    stage_position_changed = Signal(FibsemStagePosition)
    _stage_position: FibsemStagePosition = None
    # The sample holder changed: slots calibrated, or the stage rebuilt around another
    # holder. Carries the holder. On the microscope rather than the `Stage`, because
    # applying a configuration that names a holder rebuilds the `Stage`, and a signal
    # on it would take every subscriber with it.
    holder_changed = Signal(object)

    # (kind, payload): a fact for the experiment's record -- an image acquired, a
    # task step. Emit through record_event, which never raises; the app records it
    # to events.jsonl (autolamella/event_recording.py). Progress for a UI belongs
    # on the typed signals above, not here.
    record_signal = Signal(str, object)

    @abstractmethod
    def connect_to_microscope(
        self, ip_address: str, port: int, reset_beam_shift: bool = True
    ) -> None:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    def try_disconnect(self) -> bool:
        """Disconnect, logging rather than raising if the client will not close.

        For teardown paths -- a window closing -- where there is nothing left to
        tell the user and an exception would only stop the rest of the cleanup.
        Returns whether the disconnect went through.
        """
        try:
            self.disconnect()
        except Exception as e:
            logging.warning(f"Could not cleanly disconnect the microscope: {e}")
            return False
        return True

    # Imaging and the autofunctions are the beam device's commands, so every backend
    # runs the same four methods: the driver's work is in its beam's hooks.

    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        """Acquire an image with `image_settings`, or with the current settings of
        `beam_type`. Given both, the settings win.

        Raises:
            ValueError: If neither is given, or the beam is not enabled.
        """
        if image_settings is None and beam_type is None:
            raise ValueError(
                "Must provide either image_settings (to acquire with specific settings) or beam_type (to acquire with current microscope settings for that beam type)."
            )
        target = image_settings.beam_type if image_settings is not None else beam_type
        return self._imaging_service().acquire(target, image_settings)

    @property
    def is_acquiring(self) -> bool:
        """Check if the microscope is currently acquiring an image."""
        return any(beam.is_live for beam in self._live_beams())

    def _live_beams(self) -> List[Any]:
        """The beam devices whose driver has live view (`start_live`)."""
        found = []
        for beam in self.beams.values():
            info = getattr(beam, "commands", {}).get("start_live")
            if info is not None and info.available:
                found.append(beam)
        return found

    def start_acquisition(self, beam_type: BeamType) -> None:
        """Start the image acquisition process.
        Args:
            beam_type: The beam type to start acquisition for.
        """
        if self.is_acquiring:
            logging.warning("Acquisition thread is already running.")
            return

        # The beam's live view: its frames reach the old signals, so a viewer
        # listening there sees no difference. A beam with no live view (Odemis), or
        # no such beam, has nothing to start.
        beam = self.beams.get(beam_type)
        if beam is None or beam not in self._live_beams():
            logging.warning(f"The {beam_type.name} beam has no live view here.")
            return
        if beam_type not in self._live_forwarded:
            signal = (
                self.sem_acquisition_signal
                if beam_type is BeamType.ELECTRON
                else self.fib_acquisition_signal
            )
            # A function, not signal.emit: psygnal calls a connected emit once
            # as it connects it.
            beam.live_frame.connect(lambda image, signal=signal: signal.emit(image))
            self._live_forwarded.add(beam_type)
        self._imaging_service().start_live(beam_type)

    def stop_acquisition(self) -> None:
        """Stop the image acquisition process."""
        for beam in self._live_beams():
            beam.stop_live()

    def acquire_chamber_image(self) -> FibsemImage:
        """A frame from the ``chamber_camera`` device, as an image without metadata."""
        camera = self.devices.get("chamber_camera")
        if camera is None:
            raise self._unsupported("acquire_chamber_image")
        return FibsemImage(data=camera.acquire(), metadata=None)

    def autocontrast(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        """Run the instrument's brightness and contrast routine for `beam_type`,
        optionally on a rectangle of the frame (0 to 1)."""
        self._imaging_service().autocontrast(beam_type, reduced_area)

    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        """Focus a beam: with the instrument's own autofocus where the beam has one,
        else with the image-based sweep at its field of view
        (`microscope.imaging.auto_focus`)."""
        self._imaging_service().auto_focus(beam_type, reduced_area)

    def _imaging_service(self) -> "Imaging":
        """The imaging service, or the error for a microscope with none."""
        if self.imaging is None:
            raise ValueError(
                "There is no imaging: no beam is enabled on this microscope."
            )
        return self.imaging

    def _build_imaging(self) -> None:
        """Build the imaging service over the beams; `auto_focus` then goes to it.
        Without a beam there is none."""
        from fibsem.services.imaging import Imaging, bind_imaging

        self.imaging = bind_imaging(Imaging, self)

    def reset_beam_shifts(self) -> None:
        """Set the beam shift to zero for the electron and ion beams."""
        self.set_beam_shift(Point(0, 0), BeamType.ELECTRON)
        self.set_beam_shift(Point(0, 0), BeamType.ION)

    def beam_shift(
        self, dx: float, dy: float, beam_type: BeamType = BeamType.ION
    ) -> Optional[Point]:
        """Shift the beam by (dx, dy) from where it is: the beam's ``shift_by``, which
        records it. Returns the shift afterwards, or None where there is no such beam
        or shift (absent = unsupported)."""
        if self._beam_parameter("shift", beam_type) is None:
            logging.debug(f"shift is not supported here ({beam_type}).")
            return None
        return self.beams[beam_type].shift_by(dx, dy)

    def get_stage_position(self) -> FibsemStagePosition:
        """
        Get the current stage position.

        This method retrieves the current stage position from the microscope and returns it as
        a FibsemStagePosition object. FibsemStage Position is in the RAW coordinate frame

        Returns:
            FibsemStagePosition: The current stage position.
        """

        if self.stage is None:
            raise self._unsupported("get_stage_position")
        stage_position = self.stage.position.get_value()

        if not isinstance(stage_position, FibsemStagePosition):
            raise TypeError(f"Expected FibsemStagePosition, got {type(stage_position)}")

        logging.debug({"msg": "get_stage_position", "pos": stage_position.to_dict()})

        if self._stage_position is None:
            self._stage_position = deepcopy(stage_position)

        if not self._stage_position.is_close2(stage_position, tol=1e-6):
            self._stage_position = deepcopy(stage_position)
            self.stage_position_changed.emit(self._stage_position)

        return deepcopy(stage_position)

    def _read_stage_capabilities(self) -> None:
        """Ask the instrument what its stage can do, and record it.

        Only `rotation` today, and it is not a preference: it decides where the FIB
        orientation is, because `rotation_180` is derived from it (FIB-834). A
        configuration file used to state it, which meant a compustage could be
        described as a rotating stage by a typo and nothing would disagree --
        `sim-arctis-configuration.yaml` was, for as long as nothing read the flag.

        The axes are the honest form of the question: a compustage has no `r` limit
        because it has no rotation axis. It agrees with the stage type on every
        backend today, and deliberately does not ask that instead -- "has a rotation
        axis" is the property the derivation needs, and a future stage that lacks one
        without being a compustage would answer correctly here for free.

        The stage device's axes say it: they are built from the limits its driver
        reports. Without a stage device there is nothing to ask, and the field keeps
        its default.
        """
        if self.stage is not None:
            self.system.stage.rotation = "r" in self.stage.axes

    def _build_sample_handling(self) -> None:
        """Build the sample handling service (the holder, and the grid model over the
        sample loader if there is one) from the system settings."""
        from fibsem.services.sample_handling import bind_sample_handling

        # Before the service, not after. Every backend wraps this call in a
        # try/except that logs and carries on, so a failure below leaves the
        # capability at its field default -- `True`, which on a compustage is the
        # wrong answer and a silent one. Reading it first cannot fail: it asks the
        # stage device's axes, which were read when the device was built.
        self._read_stage_capabilities()
        self._read_hardware_capabilities()

        self.sample_handling = bind_sample_handling(self)
        if self.sample_handling is not None:
            self.holder_changed.emit(self.sample_handling.holder)

    # ---- fitted subsystems ---------------------------------------------------
    #
    # Whether a manipulator is fitted used to be a configuration key, which meant a
    # site could describe hardware it does not have, or omit hardware it does, and
    # nothing would disagree. It is not in the file any more. The device answers:
    # a backend builds its own manipulator, asks it (`Manipulator.is_installed`), and
    # takes it out of the map when the instrument says it is not fitted
    # (`_keep_if_fitted`). A device that cannot ask answers None, and the backend's
    # `DEFAULT_FITTED` stands in, which is what its shipped configuration used to say.
    # None falls back to that default rather than to "not fitted" -- a subsystem that
    # wrongly appears is a menu entry that errors, one that wrongly disappears is a
    # working instrument that lost a feature on upgrade.

    #: What this backend assumes is fitted when it cannot ask.
    DEFAULT_FITTED: Dict[str, bool] = {
        "manipulator": True,
    }

    #: Who answered each fitted-subsystem question at connect: ``"instrument"`` when
    #: a probe did, ``"backend"`` when the backend's `DEFAULT_FITTED` stood in,
    #: ``"device"`` for a configured device from another driver, and ``"not_built"``
    #: when its device failed to build.
    capability_sources: Dict[str, str]

    #: The fitted subsystems that are a device of the same name.
    _FITTED_DEVICES: Tuple[str, ...] = ("manipulator",)

    def _probe_fitted(self, key: str) -> Tuple[bool, str]:
        """Whether *key*'s device says it is fitted, and who answered
        (``"instrument"`` or ``"backend"``). With no device, or one that cannot say,
        the backend's `DEFAULT_FITTED` answers."""
        device = self._devices.get(key)
        try:
            present = None if device is None else device.is_installed()
        except Exception as e:
            # A raising answer is ambiguous -- a missing subsystem and a sick
            # connection look the same -- so it is read as "cannot say".
            logging.debug(f"{key}: is_installed failed: {e}")
            present = None
        if present is None:
            return bool(self.DEFAULT_FITTED[key]), "backend"
        return bool(present), "instrument"

    def _keep_if_fitted(self, key: str) -> None:
        """Ask the device this backend just built for *key* whether the instrument
        has it fitted, record the answer, and take the device out of the map when the
        answer is no. A device from another driver is fitted whatever it says: a
        manipulator on its own controller is one the vendor never reports."""
        if self._devices.get(key) is None or self._built_by_another_driver(key):
            return
        present, source = self._probe_fitted(key)
        if not hasattr(self, "_fitted_sources"):
            self._fitted_sources: Dict[str, str] = {}
        self._fitted_sources[key] = source
        sources = getattr(self, "capability_sources", None)
        if sources is not None:
            sources[key] = source
        if not present:
            logging.info(f"The instrument reports no {key}; its device is not kept.")
            self._set_device(key, None)

    def _read_hardware_capabilities(self) -> None:
        """Record who answered whether each subsystem is fitted.

        The answer itself is the device map (`is_available`): a device that is there
        is asked again, and one `_keep_if_fitted` took out keeps the answer it gave
        then. A backend that builds its parts after this (AutoScript) records them
        when it builds them.
        """
        self.capability_sources = {}
        not_built = getattr(self, "_devices_not_built", set())
        asked = getattr(self, "_fitted_sources", {})
        for key in self._FITTED_DEVICES:
            if key in self._devices or key not in asked:
                source = self._probe_fitted(key)[1]
            else:
                source = asked[key]
            self.capability_sources[key] = "not_built" if key in not_built else source
            # A device for it from another driver is fitted whatever the backend
            # says. `_set_device` does the same for a device built afterwards.
            self._mark_device_fitted(key)
        self._read_plasma_source()

    def _mark_device_fitted(self, name: str) -> None:
        """Record that a configured device from a driver other than this backend's
        answered whether *name* is fitted (`capability_sources`)."""
        if name not in self._FITTED_DEVICES or self._devices.get(name) is None:
            return
        if not self._built_by_another_driver(name):
            return
        # Before the probes run there is nothing to record a source in; they call
        # this again.
        sources = getattr(self, "capability_sources", None)
        if sources is not None:
            sources[name] = "device"

    def _mark_device_not_built(self, name: str) -> None:
        """Record that *name*'s device failed to build
        (`fibsem.devices.entries.build_device_entries`). With no device it is not
        available; this only says why, in `capability_sources`."""
        if name not in self._FITTED_DEVICES:
            return
        if not hasattr(self, "_devices_not_built"):
            self._devices_not_built = set()
        self._devices_not_built.add(name)
        sources = getattr(self, "capability_sources", None)
        if sources is not None:
            sources[name] = "not_built"

    def _bind_device_roles(self) -> None:
        """Fill the roles the configuration's ``hardware.devices`` entries bind
        (``roles: {scanner: scan_generator}``), once every device is built
        (`fibsem.devices.entries.bind_device_roles`).

        Each backend calls this after its last device is built. The FM's roles are
        filled by the FM API, so an ``fm`` entry's bindings are not read here.

        First it fills the roles every backend fills the same way: the manipulator's
        ``stage`` is the microscope's stage. A binding in the configuration replaces it.
        """
        manipulator, stage = self.manipulator_device, self.stage
        if manipulator is not None and stage is not None:
            manipulator.fill_roles(stage=stage)

        from fibsem.devices.entries import (
            ResolvedEntry,
            bind_device_roles,
            configured_device_entries,
        )

        configured = configured_device_entries(self.system)
        resolved = [
            ResolvedEntry(entry, str(entry.driver or self.system.info.manufacturer))
            for entry in configured.values()
            if entry.roles
            and entry.enabled is not False
            and entry.name != "fm"
            and entry.type != "fm"
        ]
        bind_device_roles(resolved, self.devices, configured.keys())

    def _built_by_another_driver(self, name: str) -> bool:
        """Whether the configuration's entry for *name* names a driver other than
        this backend's own."""
        from fibsem.devices.entries import configured_device_entries
        from fibsem.drivers.registry import normalize_name

        entry = configured_device_entries(self.system).get(name)
        if entry is None or not entry.driver:
            return False

        def canonical(driver: Optional[str]) -> str:
            return str(normalize_name(driver)).strip().casefold()

        return canonical(entry.driver) != canonical(self.system.info.manufacturer)

    # ---- the ion column's plasma source ----------------------------------------
    #
    # A plasma column is one with a gas (`BeamSystemSettings.plasma_gas`). Files
    # written before that said so with `plasma: true` beside the gas, and a site that
    # set the flag and left the gas at `None` loads with no gas -- so it would lose
    # its plasma controls on upgrade. For that case only, the gas the instrument is
    # running is read and recorded.
    #
    # Read, never changed: nothing here sets a gas on the instrument, and a gas the
    # configuration states is left alone -- the instrument is not even asked. The
    # probe can only find a plasma source; "cannot say" leaves the column as the
    # file described it. The ion beam device answers (`Beam.installed_plasma_gas`).

    def _read_plasma_source(self) -> None:
        """Record the ion column's plasma gas when the configuration names none."""
        if self.system.ion.plasma_gas is not None:
            return
        ion = self._devices.get("ion")
        try:
            gas = None if ion is None else ion.installed_plasma_gas()
        except Exception as e:
            # A Ga column refuses the question the same way a sick connection does.
            logging.debug(f"Reading the installed plasma gas failed: {e}")
            gas = None
        if not gas:
            return
        logging.info(
            f"The configuration names no plasma gas; the ion column reports '{gas}'."
        )
        self.system.ion.plasma_gas = str(gas)
        # The device was built before the gas was recorded, so it has no plasma_gas
        # yet, and what depends on the gas (the currents) was read without it.
        if "plasma_gas" not in ion.parameters:
            ion.bind_implemented("plasma_gas")
        for param in ion.parameters.values():
            if "plasma_gas" in param.spec.depends_on:
                param.refresh_metadata()

    def _apply_fluorescence_calibration(self) -> None:
        """Push the configured objective calibration onto the objective.

        `focus_position` and `limit_position` are calibration: somebody focused this
        objective on this microscope, and somebody decided how far it may safely be
        inserted. They lived only in `fm-configuration.yaml`, which a one-second
        debounced autosave rewrites after any channel edit -- a safety limit carried
        by a file that turns over while someone adjusts colours.

        The configuration wins when it states one. It states nothing by default, and
        then the working-state file answers exactly as before -- the FM widget applies
        that file's positions only while the configuration is silent, and stops
        writing them once it is not. Nothing is written back here.
        """
        if self.fm is None or self.fm.objective is None:
            return
        for name in ("focus_position", "limit_position"):
            value = getattr(self.system.fm, name)
            if value is None:
                continue
            try:
                setattr(self.fm.objective, name, float(value))
            except Exception as e:
                logging.warning(f"Could not apply configured objective {name}: {e}")

    def beam_uses_presets(self, beam_type: BeamType) -> bool:
        """Whether this column is set by choosing a preset rather than a voltage and
        a current. Backends where that is so override it."""
        return False

    def capture_defaults(self, beam_type: Optional[BeamType] = None) -> None:
        """Record what the instrument is doing now as the defaults a session starts from.

        The gesture the `defaults:` block exists for. Nobody wants to type 2.00 kV,
        100 pA, 150 um, 1536x1024, 1 us, ETD and SecondaryElectrons into a grid -- and
        somebody who does will type what they believe the instrument is doing. "It is
        set up how I like it, remember this" is the natural way to say it, and the
        instrument already knows the answer.

        **The stage position is deliberately not captured.** `get_microscope_state`
        returns it alongside the beams, and it has no business here: applying it
        would move the stage, and where the stage happened to be sitting when
        somebody pressed Save is session state, not a preference. The objective
        position is out for the same reason from the other direction -- it is
        calibration, not a default.

        Writes into the settings only. Saving the configuration to disk is a separate
        act, so pressing this is reversible until someone means it. *beam_type*
        limits it to one column; by default both are captured.
        """
        state = self.get_microscope_state()
        for beam, detector, record in (
            (state.electron_beam, state.electron_detector, self.system.electron),
            (state.ion_beam, state.ion_detector, self.system.ion),
        ):
            if beam_type is not None and record.beam_type is not beam_type:
                continue
            # Only the defaults. `BeamSettings` also carries the beam shift, the
            # stigmation and the working distance, and those are alignment state:
            # capturing them would put the shift the column happened to have into
            # the file, and Apply would push it back. Scan rotation is a standing
            # choice (a FIB run at 180 degrees), so it is captured with the rest.
            if beam is not None:
                for name in (
                    "voltage",
                    "beam_current",
                    "hfw",
                    "resolution",
                    "dwell_time",
                    "scan_rotation",
                ):
                    setattr(record.beam, name, deepcopy(getattr(beam, name)))
                # A preset-driven column is set by its preset, so that is its
                # default. Kept when the backend cannot say which one is active.
                if self.beam_uses_presets(record.beam_type) and beam.preset:
                    record.beam.preset = beam.preset
            if detector is not None:
                record.detector.type = detector.type
                record.detector.mode = detector.mode

    # The raw moves are the stage device's checked moves: a target outside the stage
    # limits is refused with StageLimitError, never sent to the vendor.

    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        if self.stage is None:
            raise self._unsupported("move_stage_absolute")
        self.stage.move_absolute(position)
        return self.get_stage_position()

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        if self.stage is None:
            raise self._unsupported("move_stage_relative")
        self.stage.move_relative(position)
        return self.get_stage_position()

    # The view-corrected moves below are the stage movement service's
    # (`fibsem.services.stage_movement`); these keep their names and signatures and
    # route to it. A driver's differences are its service subclass (Tescan's is
    # `TescanStageMovement`). The geometry is in `fibsem.geometry.movement`.

    # TODO: migrate from stable_move vocab to sample_stage
    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType, static_wd: bool = False
    ) -> FibsemStagePosition:
        """
        Calculate the corrected stage movements based on the beam_type stage tilt, shuttle pre-tilt,
        and then move the stage relatively.

        Args:
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
            static_wd (bool, optional): whether to fix the working distance to the eucentric heights. Defaults to False.
        """
        return self._movement_service().stable_move(
            dx=dx, dy=dy, beam_type=beam_type, static_wd=static_wd
        )

    def vertical_move(
        self,
        dy: float,
        dx: float = 0.0,
        beam_type: BeamType = BeamType.ION,
        relaxation: float = 1.0,
    ) -> FibsemStagePosition:
        """Restore coincidence from an offset measured in one of the beam views.

        Args:
            dy: offset along the image y-axis, in the view named by beam_type.
            dx: offset along the image x-axis, in the same view.
            beam_type: the view the offset was measured in. ION (the default, and
                the historical behaviour) corrects a feature already centred in the
                SEM; ELECTRON corrects one already centred in the FIB.
            relaxation: under-relaxation of the correction. 1.0 applies the
                geometrically exact move; below 1.0 deliberately undershoots it,
                which keeps a manual look-correct-look loop convergent where a
                slight model error would otherwise make it oscillate. This replaces
                the old hard-coded 0.9, which was found to be absorbing a
                decomposition error rather than correcting perspective (FIB-773).
                Every backend must accept it, because ensure_coincident passes it;
                a backend may ignore it (Tescan's SEM-view move does).

        Raises:
            NotImplementedError: if this backend cannot correct from that view.
                Ask supports_vertical_move first rather than catching this.
        """
        return self._movement_service().vertical_move(
            dy=dy, dx=dx, beam_type=beam_type, relaxation=relaxation
        )

    def supports_vertical_move(self, beam_type: BeamType = BeamType.ION) -> bool:
        """Whether coincidence can be restored from the given view on this system."""
        if self.stage_movement is None:
            return False
        return self.stage_movement.supports_vertical_move(beam_type)

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        """Where the stage would end up after a stable move from ``base_position``.

        Nothing moves. The projection is taken at the current stage pose, not at
        ``base_position``, as it always has been.
        """
        return self._movement_service().project_stable_move(
            dx=dx, dy=dy, beam_type=beam_type, base_position=base_position
        )

    def _axis_restrictions_apply(
        self, position: Optional[FibsemStagePosition] = None
    ) -> bool:
        """Whether an absolute move to *position* leaves any stage axis alone; see
        `_blocked_axes`."""
        return bool(self._blocked_axes(position))

    def _blocked_axes(
        self, position: Optional[FibsemStagePosition] = None
    ) -> Tuple[str, ...]:
        """The stage axes an absolute move to *position* leaves alone, because the
        microscope would refuse them.

        Two halves, and they are not the same rule.

        The **orientation** half asks where the move is *going*, not where the stage is
        standing, and blocks z (and r, which a stage with an FM pose does not have).
        Asking the current pose is what dropped z from the very move that was leaving
        the fluorescence pose: the stage landed at the requested x/y/t at the old
        z-height, and the operator pressed Move a second time to finish it. Measured on
        the Arctis (Aug 2026): with the objective retracted, z and t are both available
        at t = -180, so a move out of that pose has nothing to lose. A move *into* it is
        still restricted here -- more conservative than the measurement requires, and
        free, because the one path that enters the pose sends r and t only.

        `position` is the destination. Without one, or with a partial pose that has no
        orientation to read (the safe move's rotation step sends a bare tilt), the stage's
        own pose stands in -- what such a move got before.

        `get_stage_orientation` can never return "FM" on an offset mount -- the FM is a
        device there and `orientations["FM"]` is a copy of the FIB entry -- so that half
        is naturally confined to the stages that declare an FM pose.

        The **objective** half applies on every mount: while the objective is
        inserted, the axes it names (`ObjectiveLens.blocked_axes`, z and t as FIB-640
        measured) are left alone. It read the stage type until those axes were
        settled, because it dropped z and r, and r is a real rotation axis on an
        offset mount. The objective state is read live, every move: a stale
        "Retracted" here moves the stage with the objective in the chamber.
        """
        blocked = []
        destination = (
            position
            if position is not None
            and position.r is not None
            and position.t is not None
            else None
        )
        if self.get_stage_orientation(destination) == "FM":
            blocked += ["z", "r"]

        if (
            self.fm is not None
            and self.fm.objective.state is InsertableDeviceState.INSERTED
        ):
            blocked += [a for a in self.fm.objective.blocked_axes if a not in blocked]
        return tuple(blocked)

    def _without_blocked_axes(
        self, position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """*position* with the axes `_blocked_axes` names left unset, so an absolute
        move does not send them."""
        blocked = self._blocked_axes(position)
        if not blocked:
            return position
        position = deepcopy(position)
        for axis in blocked:
            setattr(position, axis, None)
        return position

    def _fluorescence_is_configured(self) -> bool:
        """Whether this site has said its instrument has a fluorescence microscope.

        The **flag decides**; the driver's own probe only confirms afterwards, and
        that order matters. A driver may have no way to ask whether an FM is fitted
        short of selecting it and seeing whether the microscope refuses. Running that
        on every system would be autodetection, and an instrument with an FM on an
        offset mount would find half-built support appearing in its UI on upgrade: a
        fluorescence tab that builds, a button that traverses the offset, pose
        derivation that is only partly right. Probing also touches the shared imaging
        channel on machines that have never had an FM.

        **A stage that reaches the FM by re-posing keeps its answer.** The stage type
        is read from the hardware, not from configuration, and the shipped
        configurations for such stages carry no `fm:` block at all. Replacing the old
        check rather than widening it would take the FM away from every such site on
        upgrade. So a configuration that does not state the flag keeps a re-posing
        stage exactly as it was, and an offset mount must opt in. An explicit `false`
        turns the FM off on either.
        """
        if self.system.fm.enabled is None:
            return self._fluorescence_default()
        return self.system.fm.enabled

    def _fluorescence_default(self) -> bool:
        """Whether this backend has an FM when the configuration does not say.

        The answer each backend gave before the flag was read: a stage that reaches
        the FM by re-posing (a compustage) has one. An explicit `fm.enabled` overrides
        it either way.
        """
        return self._fm_is_a_pose()

    def _fluorescence_uses_own_driver(self) -> bool:
        """Whether the FM, if there is one, comes from this microscope's own driver:
        the configuration names no FM driver, which is every site with an FM on the
        beams' connection. An FM on another driver (`fm.driver: remote`, an FM on its
        own PC) must not be looked for there."""
        return not self.system.fm.driver

    def _build_fluorescence(self, driver: str) -> Optional[FluorescenceMicroscope]:
        """The FM, built from the configuration's `fm` entry like any other device
        (`fibsem.devices.entries`), and its devices put in `fm_devices`; or None.

        None when the site has no FM (`_fluorescence_is_configured`). Otherwise the
        entry's `driver` builds it, else *driver*, this backend's own: an `fm` builder
        returns the FM API over its devices. A driver with no FM builder, or a build
        that fails, leaves the microscope without an FM, with a warning: an FM is not
        required unless its entry says `required: true`, and then connecting fails
        with `RequiredDeviceUnavailable`.
        """
        from fibsem.devices.entries import (
            DeviceBuildError,
            build_device_entries,
            resolve_system_devices,
        )
        from fibsem.structures import DeviceEntry

        if not self._fluorescence_is_configured():
            logging.info(
                "No fluorescence microscope configured for this system. Set "
                "`fm.enabled` in the microscope configuration to enable it."
            )
            return None
        resolved = [
            # Every configuration names the FM, so naming it does not make it
            # required, as it does another device: only `required: true` does.
            replace(item, entry=replace(item.entry, required=bool(item.entry.required)))
            for item in resolve_system_devices(
                self.system, [DeviceEntry(name="fm", type="fm")], ("fm",), driver=driver
            )
            if item.name == "fm"
        ]
        try:
            built = build_device_entries(resolved, self)
        except DeviceBuildError as e:
            raise RequiredDeviceUnavailable(str(e)) from e.__cause__
        fm = built.get("fm")
        if fm is None:
            return None
        self.fm_devices = MappingProxyType(dict(fm.devices or {}))
        logging.info(f"Fluorescence microscope connected: {type(fm).__name__}.")
        return fm

    def _refuse_rotation_at_the_fluorescence_microscope(
        self, stage_position: FibsemStagePosition
    ) -> None:
        """Refuse a stage rotation while the stage is parked at the FM.

        The objective is inserted over the sample there, and the rotation is
        compucentric about a centre back at the beams -- some 48.8 mm away -- so a half
        turn swings the sample most of the width of the chamber, under the objective.

        The route to another pose at the FM is the one `get_target_position` computes:
        traverse back to the beams, re-pose there, traverse out again. This refuses the
        shortcut. It is a refusal rather than a silent correction because a caller
        asking for the shortcut has a wrong idea of where the stage is going, and
        quietly sending it somewhere else would leave that idea intact.

        **Rotation only.** A tilt pivots about an axis through the sample instead of
        swinging it, and where the objective does restrict tilt the microscope refuses
        it itself -- FIB-640 measured z and t. This does not duplicate that.

        Not a restriction on FM-MILLING. That pose is a half turn from the FM's own
        orientation (measured: FM sits at r=180, MILLING at r=0), so it was never
        reachable by rotating in place -- it is reached the way everything else at the
        FM is, via the beams.

        Dormant until the connection gate opens: `microscope.fm` is `None` on every
        non-compustage system today, so nothing can park at the FM to begin with.
        """
        # A stage that reaches the FM by re-posing (a compustage, flipping) has its
        # devices at one place, so "parked at the FM" is not a state it can be in --
        # and it has no rotation axis to be compucentric about either.
        if self._fm_is_a_pose() or self.fm is None:
            return

        if stage_position.r is None:
            return

        current_position = self.get_stage_position()
        if self.get_current_device(current_position) != "FM":
            return

        from fibsem import movement

        # The same 5 degrees `get_stage_orientation` classifies within, so a caller
        # asking for the pose the stage is already in does not trip this on the slop a
        # real stage always carries.
        if movement.rotation_angle_is_smaller(
            stage_position.r, current_position.r, atol=5
        ):
            return

        raise ValueError(
            "Cannot rotate the stage while it is at the fluorescence microscope: the "
            "rotation is compucentric about a centre at the beams, so it would swing "
            "the sample across the chamber under the objective. Move to the beams "
            "first (move_to_device('FIBSEM')), re-pose there, and travel back -- "
            "or ask move_to_device for the pose and let it order the legs."
        )

    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        """Move the stage to the given named orientation (e.g. 'SEM', 'FIB', 'MILLING').
        Args:
            orientation: The name of the orientation to move to.
        Returns:
            FibsemStagePosition: The new stage position after moving to the orientation.
        """
        return self._movement_service().move_to_orientation(orientation)

    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system
        """
        self._movement_service().safe_absolute_stage_movement(stage_position)

    def get_manipulator_state(self) -> Optional[bool]:
        """Get the manipulator state (Inserted = True, Retracted = False), or None
        when there is no manipulator."""
        # TODO: convert to enum
        if self.manipulator_device is None:
            return None
        state = self.manipulator_device.state.get_value()
        return state is InsertableDeviceState.INSERTED

    def get_manipulator_position(self) -> Optional[FibsemManipulatorPosition]:
        """Get the manipulator position, or None when there is no manipulator."""
        if self.manipulator_device is None:
            return None
        return self.manipulator_device.position.get_value()

    # Every manipulator move returns where the needle is afterwards, as the
    # Manipulator device's commands do. The raw moves go through the device when the
    # backend builds one. The corrected and offset moves depend on the stage and
    # beam geometry, which differs by backend, so they stay the backend's.

    def insert_manipulator(
        self, name: Optional[str] = None
    ) -> Optional[FibsemManipulatorPosition]:
        if self.manipulator_device is None:
            raise self._unsupported("insert_manipulator")
        return self.manipulator_device.insert(name)

    def retract_manipulator(self) -> Optional[FibsemManipulatorPosition]:
        if self.manipulator_device is None:
            raise self._unsupported("retract_manipulator")
        return self.manipulator_device.retract()

    def move_manipulator_relative(
        self, position: FibsemManipulatorPosition
    ) -> Optional[FibsemManipulatorPosition]:
        if self.manipulator_device is None:
            raise self._unsupported("move_manipulator_relative")
        return self.manipulator_device.move_relative(position)

    def move_manipulator_absolute(
        self, position: FibsemManipulatorPosition
    ) -> Optional[FibsemManipulatorPosition]:
        if self.manipulator_device is None:
            raise self._unsupported("move_manipulator_absolute")
        return self.manipulator_device.move_absolute(position)

    def move_manipulator_corrected(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> Optional[FibsemManipulatorPosition]:
        """Move the needle by (dx, dy) in the image coordinates of *beam_type*'s view,
        through the manipulator's ``move_corrected`` command."""
        if "corrected" not in self.manipulator_move_types:
            raise self._unsupported("move_manipulator_corrected")
        return self.manipulator_device.move_corrected(dx, dy, beam_type)

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> Optional[FibsemManipulatorPosition]:
        raise self._unsupported("move_manipulator_to_position_offset")

    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        if self.manipulator_device is None:
            raise self._unsupported("_get_saved_manipulator_position")
        return self.manipulator_device.saved_position(name)

    # What a manipulator offers beyond the raw moves is the backend's to say, so the
    # manipulator widget asks these rather than checking the class. Whether the arm
    # rotates is `is_available("manipulator_rotation")`, from the device's axes.

    @property
    def manipulator_move_types(self) -> Tuple[str, ...]:
        """The moves this backend offers: "relative" always, and "corrected" when its
        manipulator can move the needle in a beam's image coordinates
        (`move_manipulator_corrected`)."""
        device = self.manipulator_device
        if device is not None and device.commands["move_corrected"].available:
            return ("relative", "corrected")
        return ("relative",)

    def manipulator_named_positions(self) -> List[str]:
        """The instrument's own named manipulator positions, if it has any."""
        if self.manipulator_device is None:
            return []
        return self.manipulator_device.named_positions()

    def move_manipulator_to_named_position(
        self, name: str
    ) -> Optional[FibsemManipulatorPosition]:
        """Move the needle to one of `manipulator_named_positions`.

        Returns where it is afterwards.
        """
        position = self._get_saved_manipulator_position(name)
        return self.move_manipulator_absolute(position)

    def set_channel(self, channel: BeamType) -> None:
        """Make `channel` the active view and device, for the calls that act on it."""
        raise self._unsupported("set_channel")

    def _build_stage_movement(self, service: Optional[type] = None) -> None:
        """Build the stage movement service over the stage and beams, of the driver's
        class (`StageMovement` by default); the view-corrected moves then go to it.
        Without a stage device there is none."""
        from fibsem.services.stage_movement import StageMovement, bind_stage_movement

        self.stage_movement = bind_stage_movement(service or StageMovement, self)

    def _movement_service(self) -> "StageMovement":
        """The stage movement service, or the error for a microscope with none."""
        if self.stage_movement is None:
            raise ValueError(
                "There is no stage movement: this microscope has no stage device."
            )
        return self.stage_movement

    def _milling_service(self) -> Milling:
        """The milling service, or the error for a microscope with none."""
        if self.milling is None:
            raise ValueError(
                "There is no milling: the ION beam is not enabled on this microscope."
            )
        return self.milling

    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        """Apply a milling recipe: the channel, the patterning defaults and the beam
        conditions. The first setup saves the beam conditions `finish_milling` puts
        back."""
        self._milling_service().setup(mill_settings)

    def run_milling(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None
    ) -> None:
        """Mill what is drawn, with the beam conditions `setup_milling` applied, and
        return when the mill ends. Progress is reported on ``milling_progress_signal``.
        A set ``stop_event`` stops the beam and raises `OperationCancelledError`. To
        start a mill and return at once, use `start_milling`."""
        logging.debug("running milling now...")
        self._milling_service().run(stop_event=stop_event)

    def finish_milling(
        self,
        imaging_current: Optional[float] = None,
        imaging_voltage: Optional[float] = None,
    ) -> None:
        """Clear the patterns and put the milling beam back as ``setup_milling`` found
        it. An imaging current or voltage given wins over what was saved."""
        milling = self._milling_service()
        milling.clear()
        milling.restore()
        # only what the beam can set: a Tescan ion column takes both from its preset
        beam = milling.beam(self.milling_channel)

        def settable(name: str) -> bool:
            return (
                beam is not None
                and name in beam.parameters
                and getattr(beam, name).settable
            )

        if imaging_voltage is not None and settable("voltage"):
            self.set_beam_voltage(
                voltage=imaging_voltage, beam_type=self.milling_channel
            )
        if imaging_current is not None and settable("current"):
            self.set_beam_current(
                current=imaging_current, beam_type=self.milling_channel
            )
        logging.debug(
            {
                "msg": "finish_milling",
                "imaging_current": imaging_current,
                "imaging_voltage": imaging_voltage,
            }
        )

    def clear_patterns(self) -> None:
        self._milling_service().clear()

    def stop_milling(self) -> None:
        if self.milling is None:
            return  # nothing is milling
        self.milling.stop()

    def start_milling(self) -> None:
        self._milling_service().start()

    def pause_milling(self) -> None:
        self._milling_service().pause()

    def resume_milling(self) -> None:
        self._milling_service().resume()

    def get_milling_state(self) -> MillingState:
        if self.milling is None:
            return MillingState.IDLE
        return self.milling.state.get_value()

    def estimate_milling_time(self) -> float:
        return self._milling_service().estimate()

    def draw_patterns(self, patterns: List[FibsemPatternSettings]) -> None:
        """Draw milling patterns on the microscope from the list of settings. All
        are checked first: one this instrument can't draw raises `UnsupportedPattern`
        with none drawn.

        Args:
            patterns (List[FibsemPatternSettings]): List of milling patterns
        """
        self._milling_service().draw(patterns)

    def draw_pattern(self, pattern: FibsemPatternSettings) -> None:
        """Draw a milling pattern from settings. A pattern this instrument can't
        draw raises `UnsupportedPattern`.

        Args:
            pattern_settings (FibsemPatternSettings): pattern settings
        """
        self._milling_service().draw([pattern])

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings) -> None:
        self._milling_service().draw([pattern_settings])

    def draw_line(self, pattern_settings: FibsemLineSettings) -> None:
        self._milling_service().draw([pattern_settings])

    def draw_circle(self, pattern_settings: FibsemCircleSettings) -> None:
        self._milling_service().draw([pattern_settings])

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings) -> None:
        self._milling_service().draw([pattern_settings])

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        self._milling_service().draw([pattern_settings])

    # ---- devices -----------------------------------------------------------------
    #
    # Every device this microscope built, by name, in the order they were built:
    # `electron`, `ion`, `stage`, `chamber`, `manipulator`, and the FM group and parts
    # by theirs (`fm`, `camera`, ...). The typed attributes below (`beams`, `stage`,
    # `chamber_device`, ...) are views of it, and assigning one is how a backend adds
    # its devices, so the two can't disagree.
    _devices = _PerInstance(lambda _: {})

    @property
    def devices(self) -> Mapping[str, Any]:
        """Every device this microscope built, by name. Read-only.

        `devices["stage"] is microscope.stage`, `devices["electron"] is
        microscope.beams[BeamType.ELECTRON]`, and the FM's parts are under the names
        the device server serves them by. Empty on a backend that builds no devices.
        """
        return MappingProxyType(self._devices)

    # The service attributes, in the order `services` lists them.
    _SERVICE_ATTRIBUTES = (
        "milling",
        "stage_movement",
        "spot_burn",
        "sample_handling",
        "imaging",
    )

    @property
    def services(self) -> Mapping[str, Any]:
        """Every service this microscope built, by name (``services["milling"] is
        microscope.milling``). Read-only. A service the backend didn't build (None)
        is left out. The App Server serves them under ``/microscope/services``."""
        services = (getattr(self, name, None) for name in self._SERVICE_ATTRIBUTES)
        return MappingProxyType(
            {service.name: service for service in services if service is not None}
        )

    def _set_device(self, name: str, device: Optional[Any]) -> None:
        """Put *device* in the map as *name*, or take *name* out when it is None."""
        if device is None:
            self._devices.pop(name, None)
        else:
            self._devices[name] = device
            self._mark_device_fitted(name)

    @property
    def beams(self) -> Mapping[BeamType, Any]:
        """The beam devices, by beam type: the `electron` and `ion` devices."""
        return MappingProxyType(
            {
                beam_type: self._devices[_BEAM_DEVICE_NAMES[beam_type]]
                for beam_type in _BEAM_DEVICE_NAMES
                if _BEAM_DEVICE_NAMES[beam_type] in self._devices
            }
        )

    @beams.setter
    def beams(self, beams: Mapping[BeamType, Any]) -> None:
        for beam_type, name in _BEAM_DEVICE_NAMES.items():
            self._set_device(name, (beams or {}).get(beam_type))

    # The stage as a device (fibsem.devices.Stage), once a backend builds one. A
    # backend keeps its vendor stage object private, on its driver's connection.
    @property
    def stage(self) -> Optional[Any]:
        return self._devices.get("stage")

    @stage.setter
    def stage(self, device: Optional[Any]) -> None:
        self._set_device("stage", device)

    @property
    def stage_device(self) -> Optional[Any]:
        """The stage device's earlier name, kept as an alias of `stage`."""
        return self.stage

    @stage_device.setter
    def stage_device(self, device: Optional[Any]) -> None:
        self.stage = device

    # The chamber and the manipulator as devices (fibsem.devices.Chamber and
    # .Manipulator), named `*_device` because `chamber` is taken on Demo. The methods
    # that read them (pump/vent, get_manipulator_state/position) use the device.
    @property
    def chamber_device(self) -> Optional[Any]:
        return self._devices.get("chamber")

    @chamber_device.setter
    def chamber_device(self, device: Optional[Any]) -> None:
        self._set_device("chamber", device)

    @property
    def manipulator_device(self) -> Optional[Any]:
        return self._devices.get("manipulator")

    @manipulator_device.setter
    def manipulator_device(self, device: Optional[Any]) -> None:
        self._set_device("manipulator", device)

    # The FM's parts and its group as devices (fibsem.devices.fm), by device name,
    # beside `fm`. They drive the same FM objects `fm` holds, so the two share one
    # state. Empty when there is no FM or the backend builds no devices. In
    # `devices` they keep the names the device server serves them by (`fm`,
    # `camera`, `light_source`, `filter_set`, `objective`).
    _fm_names = _PerInstance(lambda _: [])

    @property
    def fm_devices(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: self._devices[name]
                for name in self._fm_names
                if name in self._devices
            }
        )

    @fm_devices.setter
    def fm_devices(self, devices: Mapping[str, Any]) -> None:
        for name in self._fm_names:
            self._devices.pop(name, None)
        self._fm_names[:] = list(devices or {})
        for name, device in (devices or {}).items():
            self._devices[name] = device

    def _unsupported(self, method: str) -> NotImplementedError:
        """The error an optional method raises on a backend that does not have it.

        The instrument parts not every system has (the manipulator, the chamber
        camera) and the patterns not every vendor can draw are optional: the base
        class raises this, and a backend that has them overrides the method.
        """
        return NotImplementedError(f"{type(self).__name__} does not support {method}.")

    def _beam_config(self, key: str, beam_type: Optional[BeamType]) -> Any:
        """The system settings of a beam, for a config key."""
        if beam_type is BeamType.ELECTRON:
            return self.system.electron
        if beam_type is BeamType.ION:
            return self.system.ion
        raise ValueError(f"Unknown beam type: {beam_type} for {key}")

    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        """The beam's current imaging settings (`Beam.image_settings`), or the
        defaults with nothing read where there is no such beam."""
        beam = self.beams.get(beam_type)
        if beam is None:
            return ImageSettings(
                beam_type=beam_type, resolution=None, dwell_time=None, hfw=None
            )
        return beam.image_settings()

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        """Set the imaging settings for the specified beam type."""
        logging.debug(f"Setting {image_settings.beam_type.name} imaging settings...")
        self.set_resolution(image_settings.resolution, image_settings.beam_type)
        self.set_dwell_time(image_settings.dwell_time, image_settings.beam_type)
        self.set_field_of_view(image_settings.hfw, image_settings.beam_type)

        # TODO: implement the rest of these settings... @patrick
        logging.debug(
            {
                "msg": "set_imaging_settings",
                "image_settings": image_settings.to_dict(),
                "beam_type": image_settings.beam_type.name,
            }
        )

        return

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        """Get the current beam settings for the specified beam type."""

        logging.debug(f"Getting {beam_type.name} beam settings...")
        beam_settings = BeamSettings(
            beam_type=beam_type,
            working_distance=self.get_working_distance(beam_type),
            beam_current=self.get_beam_current(beam_type),
            voltage=self.get_beam_voltage(beam_type),
            hfw=self.get_field_of_view(beam_type),
            resolution=self.get_resolution(beam_type),
            dwell_time=self.get_dwell_time(beam_type),
            stigmation=self.get_stigmation(beam_type),
            shift=self.get_beam_shift(beam_type),
            scan_rotation=self.get_scan_rotation(beam_type),
            preset=self.get_preset(beam_type),
        )
        logging.debug(
            {
                "msg": "get_beam_settings",
                "beam_settings": beam_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        return beam_settings

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        """Set the beam settings for the specified beam type"""
        logging.debug(f"Setting {beam_settings.beam_type.name} beam settings...")
        # A None is "not stated", not a value to push. A configuration may leave
        # any of these out -- a `defaults:` block with only the voltage in it is a
        # configuration -- and the readers default them to None, so each one is
        # pushed only when there is something to push.
        beam_type = beam_settings.beam_type
        setters = (
            (self.set_working_distance, beam_settings.working_distance),
            # voltage before current: on some instruments the available currents
            # are calibrated per voltage.
            (self.set_beam_voltage, beam_settings.voltage),
            (self.set_beam_current, beam_settings.beam_current),
            (self.set_field_of_view, beam_settings.hfw),
            (self.set_resolution, beam_settings.resolution),
            (self.set_dwell_time, beam_settings.dwell_time),
            (self.set_stigmation, beam_settings.stigmation),
            (self.set_beam_shift, beam_settings.shift),
            (self.set_scan_rotation, beam_settings.scan_rotation),
        )
        for setter, value in setters:
            if value is not None:
                setter(value, beam_type)
        if beam_settings.preset is not None:
            self.set_preset(beam_settings.preset, beam_type)

        logging.debug(
            {
                "msg": "set_beam_settings",
                "beam_settings": beam_settings.to_dict(),
                "beam_type": beam_settings.beam_type.name,
            }
        )
        return

    def get_beam_system_settings(self, beam_type: BeamType) -> BeamSystemSettings:
        """Get the current beam system settings for the specified beam type."""
        logging.debug(f"Getting {beam_type.name} beam system settings...")
        config = self._beam_config("beam_enabled", beam_type)
        beam_system_settings = BeamSystemSettings(
            beam_type=beam_type,
            enabled=config.enabled,
            beam=self.get_beam_settings(beam_type),
            detector=self.get_detector_settings(beam_type),
            eucentric_height=config.eucentric_height,
            column_tilt=config.column_tilt,
            plasma_gas=self._read_beam("plasma_gas", beam_type),
        )

        logging.debug(
            {
                "msg": "get_beam_system_settings",
                "settings": beam_system_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )
        return beam_system_settings

    def set_beam_system_settings(self, settings: BeamSystemSettings) -> None:
        """Set the beam system settings for the specified beam type."""
        beam_type = settings.beam_type
        logging.debug(f"Setting {settings.beam_type.name} beam system settings...")
        config = self._beam_config("beam_enabled", beam_type)
        config.enabled = settings.enabled
        self._apply_beam_defaults(settings)
        config.eucentric_height = settings.eucentric_height
        config.column_tilt = settings.column_tilt

        # Only a plasma column has a gas to set; a None here means "no plasma source",
        # not "clear the gas".
        if beam_type is BeamType.ION and settings.plasma_gas is not None:
            self._write_beam("plasma_gas", settings.plasma_gas, beam_type)

        logging.debug(
            {
                "msg": "set_beam_system_settings",
                "settings": settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        return

    def _apply_beam_defaults(self, settings: BeamSystemSettings) -> None:
        """Set one column to its defaults: the beam settings, and the detector type
        and mode.

        What the configuration decides, not what the column is aligned to. The
        working distance, stigmation and beam shift are the column's current
        alignment, and the detector's brightness and contrast are what the last
        autocontrast left; a configuration that does not state them used to load
        them as 0 (and the working distance as the eucentric height), and Apply
        pushed those -- refocusing both columns and blacking out the detectors.
        """
        beam_type = settings.beam_type
        self.set_beam_settings(
            replace(settings.beam, working_distance=None, stigmation=None, shift=None)
        )
        if settings.detector.type is not None:
            self.set_detector_type(settings.detector.type, beam_type)
        if settings.detector.mode is not None:
            self.set_detector_mode(settings.detector.mode, beam_type)

    def turn_beams_on(self) -> None:
        """Turn on each available column that is off. Never turns one off.

        What `defaults.beams_on_at_connect` does at connect.
        """
        for beam_type, key in (
            (BeamType.ELECTRON, "electron_beam"),
            (BeamType.ION, "ion_beam"),
        ):
            if self.is_available(key) and not self.is_on(beam_type):
                self.turn_on(beam_type)

    def apply_defaults(self) -> None:
        """Set each available column to the configured defaults, and nothing else.

        What `defaults.apply_on_connect` does at connect. Narrower than
        `apply_configuration`: the beams are not switched on or off, the column
        geometry and the plasma gas are not set, and the stage is left alone --
        connecting must not change any of those.
        """
        for record, key in (
            (self.system.electron, "electron_beam"),
            (self.system.ion, "ion_beam"),
        ):
            if self.is_available(key):
                self._apply_beam_defaults(record)
        logging.info("Configured defaults applied to the microscope.")

    def get_detector_settings(
        self, beam_type: BeamType = BeamType.ELECTRON
    ) -> FibsemDetectorSettings:
        """Get the current detector settings for the specified beam type."""
        logging.debug(f"Getting {beam_type.name} detector settings...")
        detector_settings = FibsemDetectorSettings(
            type=self.get_detector_type(beam_type),
            mode=self.get_detector_mode(beam_type),
            brightness=self.get_detector_brightness(beam_type),
            contrast=self.get_detector_contrast(beam_type),
        )
        logging.debug(
            {
                "msg": "get_detector_settings",
                "detector_settings": detector_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )
        return detector_settings

    def set_detector_settings(
        self,
        detector_settings: FibsemDetectorSettings,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> None:
        """Set the detector settings for the specified beam type"""
        logging.debug(f"Setting {beam_type.name} detector settings...")
        self.set_detector_type(detector_settings.type, beam_type)
        self.set_detector_mode(detector_settings.mode, beam_type)
        self.set_detector_brightness(detector_settings.brightness, beam_type)
        self.set_detector_contrast(detector_settings.contrast, beam_type)
        logging.debug(
            {
                "msg": "set_detector_settings",
                "detector_settings": detector_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        return

    def get_microscope_state(
        self, beam_type: Optional[BeamType] = None
    ) -> MicroscopeState:
        """Get the current microscope state."""

        # default values
        electron_beam, electron_detector = None, None
        ion_beam, ion_detector = None, None
        stage_position = None
        get_electron_state = beam_type in [BeamType.ELECTRON, None]
        get_ion_state = beam_type in [BeamType.ION, None]

        # get the state of the electron beam
        if self.is_available("electron_beam") and get_electron_state:
            electron_beam = self.get_beam_settings(beam_type=BeamType.ELECTRON)
            electron_detector = self.get_detector_settings(beam_type=BeamType.ELECTRON)

        # get the state of the ion beam
        if self.is_available("ion_beam") and get_ion_state:
            ion_beam = self.get_beam_settings(beam_type=BeamType.ION)
            ion_detector = self.get_detector_settings(beam_type=BeamType.ION)

        # get the state of the stage
        if self.is_available("stage"):
            stage_position = self.get_stage_position()

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            stage_position=stage_position,  # get absolute stage coordinates (RAW)
            electron_beam=electron_beam,  # electron beam state
            ion_beam=ion_beam,  # ion beam state
            electron_detector=electron_detector,  # electron beam detector state
            ion_detector=ion_detector,  # ion beam detector state
        )

        logging.debug(
            {"msg": "get_microscope_state", "state": current_microscope_state.to_dict()}
        )

        return deepcopy(current_microscope_state)

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        """Reset the microscope state to the provided state."""

        if self.is_available("electron_beam"):
            if microscope_state.electron_beam is not None:
                self.set_beam_settings(microscope_state.electron_beam)
            if microscope_state.electron_detector is not None:
                self.set_detector_settings(
                    microscope_state.electron_detector, BeamType.ELECTRON
                )
        if self.is_available("ion_beam"):
            if microscope_state.ion_beam is not None:
                self.set_beam_settings(microscope_state.ion_beam)
            if microscope_state.ion_detector is not None:
                self.set_detector_settings(microscope_state.ion_detector, BeamType.ION)
        if self.is_available("stage") and microscope_state.stage_position is not None:
            self.safe_absolute_stage_movement(microscope_state.stage_position)
        if self.fm is not None and microscope_state.objective_position is not None:
            self.fm.objective.move_absolute(microscope_state.objective_position)

        logging.debug(
            {"msg": "set_microscope_state", "state": microscope_state.to_dict()}
        )

        return

    def is_available(self, system: str) -> bool:
        """Whether *system* is on this microscope, read from its devices.

        A subsystem is there when its device was built: `enabled: false` builds no
        device, and neither does a part the instrument says is not fitted or a device
        that failed to build. ``stage_rotation`` and the manipulator's ``rotation`` and
        ``tilt`` are the axes their device reports; ``ion_plasma`` is an ion column
        with a plasma gas. Anything else is not available.
        """
        devices = self._devices
        if system == "electron_beam":
            return "electron" in devices
        if system == "ion_beam":
            return "ion" in devices
        if system == "ion_plasma":
            return "ion" in devices and self.system.ion.plasma
        if system == "stage":
            return "stage" in devices
        if system == "stage_rotation":
            stage = devices.get("stage")
            return stage is not None and "r" in stage.axes
        if system == "manipulator":
            return "manipulator" in devices
        if system in ("manipulator_rotation", "manipulator_tilt"):
            axis = "r" if system == "manipulator_rotation" else "t"
            return axis in self._manipulator_axes()
        return False

    def _manipulator_axes(self) -> Tuple[str, ...]:
        """The axes the manipulator device moves on; none without one."""
        axes = getattr(self._devices.get("manipulator"), "axes", None)
        return tuple(axes) if axes is not None else ()

    def apply_configuration(
        self, system_settings: Optional[SystemSettings] = None
    ) -> None:
        """Apply the system settings to the microscope."""

        logging.info("Applying Microscope Configuration...")

        if system_settings is None:
            system_settings = self.system
            logging.info("Using current system settings.")

        # apply the system settings
        if self.is_available("electron_beam"):
            self.set_beam_system_settings(system_settings.electron)
        if self.is_available("ion_beam"):
            self.set_beam_system_settings(system_settings.ion)

        if self.is_available("stage"):
            previous_stage = self.system.stage
            self.system.stage = system_settings.stage
            # The line above replaces the whole record, including the capability the
            # instrument told us about at connect. `system_settings` came from a file,
            # and files no longer state `rotation` -- so on a compustage this would
            # silently restore the field default of `True`, move the FIB orientation
            # half a turn away, and hand the compucentric correction a rotation the
            # stage cannot make. Re-read rather than preserve: the instrument is the
            # authority, and it has not changed because someone pressed Apply.
            self._read_stage_capabilities()
            # The same replacement empties the holder map, for the same reason: a
            # configuration written before the holder moved into it has no `holders:`.
            # But which holder is in the shuttle is a physical fact, and pressing Apply
            # did not change it -- so it is carried across, and the running service
            # keeps agreeing with the configuration it came from. A file that *does*
            # name holders wins, because then the user is choosing one.
            if not self.system.stage.holders:
                self.system.stage.holders = previous_stage.holders
                self.system.stage.active_holder = previous_stage.active_holder
            else:
                # The file chose a holder, so the sample handling service -- built
                # once at connect -- has to be rebuilt around it, or `system.stage`
                # says one holder while `sample_handling.holder` still moves to the
                # slots of another.
                self._build_sample_handling()

        # `system_settings.manipulator` is not taken from the incoming settings:
        # what is fitted is not in the file, so the incoming record only carries
        # defaults, and the one already here carries what the instrument (or the
        # backend) said at connect.

        # dont update info -> read only
        logging.info("Microscope configuration applied.")
        logging.debug(
            {"msg": "apply_configuration", "system_settings": system_settings.to_dict()}
        )

    def _stage_value(self, name: str) -> Any:
        """The stage device's parameter `name`, or None where there is no stage or
        the stage has no such parameter (a compustage is never linked)."""
        param = None if self.stage is None else self.stage.parameters.get(name)
        return None if param is None else param.get_value()

    def home(self) -> Optional[bool]:
        """Home the stage. A stage that cannot home is left alone; returns whether
        it is homed (None when it cannot say).

        Deprecated: call ``microscope.stage.home()``.
        """
        warnings.warn(
            "microscope.home() is deprecated and will be removed in the next minor "
            "release; use microscope.stage.home() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.stage is not None and self.stage.commands["home"].available:
            return self.stage.home()
        return self._stage_value("homed")

    def link_stage(self) -> Optional[bool]:
        """Link the stage to the working distance. A stage that cannot link is left
        alone; returns whether it is linked (None when it cannot say)."""
        if self.stage is not None and self.stage.commands["link"].available:
            return self.stage.link()
        return self._stage_value("linked")

    def pump(self) -> Optional[str]:
        """Pump the chamber and return its state. Raises where there is no chamber
        device, rather than doing nothing."""
        if self.chamber_device is None:
            raise self._unsupported("pump")
        return _chamber_state_name(self.chamber_device.pump())

    def vent(self) -> Optional[str]:
        """Vent the chamber and return its state. Raises where there is no chamber
        device, rather than doing nothing."""
        if self.chamber_device is None:
            raise self._unsupported("vent")
        return _chamber_state_name(self.chamber_device.vent())

    def _beam_parameter(
        self, name: str, beam_type: Optional[BeamType]
    ) -> Optional[Any]:
        """The beam device's parameter `name`, or None when the backend builds no such
        beam, or the beam has no such parameter. Only the plasma gas and the preset may
        be asked without a beam type."""
        if beam_type is None and name not in _BEAM_PARAMETERS_WITHOUT_BEAM:
            raise ValueError(f"{name} needs a beam type")
        beam = self.beams.get(beam_type)
        return None if beam is None else beam.parameters.get(name)

    def _beam_device(self, beam_type: BeamType) -> Any:
        """The beam device for ``beam_type``; a column disabled in the config has none."""
        device = self.beams.get(beam_type)
        if device is None:
            raise ValueError(f"The {beam_type.name} beam is not enabled.")
        return device

    def _read_beam(self, name: str, beam_type: BeamType) -> Any:
        """A beam wrapper's read: the beam device's parameter, an enum as its plain
        value, or None where there is no such beam or parameter (absent =
        unsupported)."""
        param = self._beam_parameter(name, beam_type)
        if param is None:
            logging.debug(f"{name} is not supported here ({beam_type}).")
            return None
        return _plain_value(param.get_value())

    def _write_beam(self, name: str, value: Any, beam_type: BeamType) -> None:
        """A beam wrapper's write: to the beam device's parameter, unvalidated, or
        nothing where there is no such beam or parameter (absent = unsupported)."""
        param = self._beam_parameter(name, beam_type)
        if param is None:
            logging.debug(f"{name} is not supported here ({beam_type}).")
            return
        param.write_through(value)

    def turn_on(self, beam_type: BeamType) -> bool:
        """Turn on the specified beam type."""
        self._write_beam("on", True, beam_type)
        return self._read_beam("on", beam_type)

    def turn_off(self, beam_type: BeamType) -> bool:
        "Turn off the specified beam type."
        self._write_beam("on", False, beam_type)
        return self._read_beam("on", beam_type)

    def is_on(self, beam_type: BeamType) -> bool:
        """Check if the specified beam type is on."""
        return self._read_beam("on", beam_type)

    def blank(self, beam_type: BeamType) -> bool:
        """Blank the specified beam type."""
        self._write_beam("blanked", True, beam_type)
        return self._read_beam("blanked", beam_type)

    def unblank(self, beam_type: BeamType) -> bool:
        """Unblank the specified beam type."""
        self._write_beam("blanked", False, beam_type)
        return self._read_beam("blanked", beam_type)

    def is_blanked(self, beam_type: BeamType) -> bool:
        """Check if the specified beam type is blanked."""
        return self._read_beam("blanked", beam_type)

    def get_available_beams(self) -> List[BeamType]:
        """Get the available beams for the microscope."""
        available_beams = []
        if self.is_available("electron_beam"):
            available_beams.append(BeamType.ELECTRON)
        if self.is_available("ion_beam"):
            available_beams.append(BeamType.ION)
        return available_beams

    def _scan_beam(self, beam_type: BeamType) -> Optional[Any]:
        """The beam device whose scan commands the scan-mode methods use, if any.

        Without one (no such beam, or a beam with no scan modes, as on TESCAN) the
        scan-mode methods do nothing.
        """
        beam = self.beams.get(beam_type)
        if beam is None or not beam.commands["spot"].available:
            return None
        return beam

    def set_spot_scanning_mode(self, point: Point, beam_type: BeamType) -> None:
        """Set the spot scanning mode for the specified beam type."""
        beam = self._scan_beam(beam_type)
        if beam is None:
            logging.debug(f"No spot scan mode here ({beam_type}).")
            return
        beam.spot(point)

    def set_reduced_area_scanning_mode(
        self, reduced_area: FibsemRectangle, beam_type: BeamType
    ) -> None:
        """Set the reduced area scanning mode for the specified beam type."""
        beam = self._scan_beam(beam_type)
        if beam is None:
            logging.debug(f"No reduced area scan mode here ({beam_type}).")
            return
        beam.reduced_area(reduced_area)

    def set_full_frame_scanning_mode(self, beam_type: BeamType) -> None:
        """Set the full frame scanning mode for the specified beam type."""
        beam = self._scan_beam(beam_type)
        if beam is None:
            logging.debug(f"No full frame scan mode here ({beam_type}).")
            return
        beam.full_frame()

    def run_spot_burn(
        self,
        settings: SpotBurnSettings,
        beam_type: BeamType = BeamType.ION,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Burn each coordinate in *settings* with the beam for the configured exposure time.

        Goes to the microscope's spot burn service, ``spot_burn``. Progress is reported
        via ``spot_burn_progress_signal``, which the status bar and the spot burn
        widget subscribe to.

        Args:
            settings: What to burn — coordinates (0-1 image coordinates), exposure time
                per point in seconds, and the milling current to burn at.
            beam_type: The type of beam to use. Only the ion beam burns.
                (Default: BeamType.ION)
            stop_event: Threading event to signal cancellation. (Default: None)

        Raises:
            ValueError: *beam_type* is not the ion beam, or there is no spot burn
                service on this microscope.
        """
        if beam_type is not BeamType.ION:
            raise ValueError(
                f"Spot burn is only supported on the ion beam, got {beam_type.name}."
            )
        if self.spot_burn is None:
            raise ValueError(
                "There is no spot burn on this microscope: the ION beam is not "
                "enabled, or it can't blank and park."
            )
        self.spot_burn.run(settings, stop_event=stop_event)

    def get_beam_current(self, beam_type: BeamType) -> float:
        """Get the beam current for the specified beam type."""
        return self._read_beam("current", beam_type)

    def set_beam_current(self, current: float, beam_type: BeamType) -> float:
        """Set the beam current for the specified beam type."""
        self._write_beam("current", current, beam_type)
        return self._read_beam("current", beam_type)

    def get_beam_voltage(self, beam_type: BeamType) -> float:
        """Get the beam voltage for the specified beam type."""
        return self._read_beam("voltage", beam_type)

    def set_beam_voltage(self, voltage: float, beam_type: BeamType) -> float:
        """Set the beam voltage for the specified beam type."""
        self._write_beam("voltage", voltage, beam_type)
        return self._read_beam("voltage", beam_type)

    def set_resolution(
        self, resolution: Tuple[int, int], beam_type: BeamType
    ) -> List[int]:
        """Set the resolution for the specified beam type."""
        self._write_beam("resolution", resolution, beam_type)
        return self._read_beam("resolution", beam_type)

    def get_resolution(self, beam_type: BeamType) -> Tuple[int, int]:
        """Get the resolution for the specified beam type."""
        return self._read_beam("resolution", beam_type)

    def get_field_of_view(self, beam_type: BeamType) -> float:
        """Get the field of view for the specified beam type."""
        return self._read_beam("hfw", beam_type)

    def set_field_of_view(self, hfw: float, beam_type: BeamType) -> float:
        """Set the field of view for the specified beam type."""
        self._write_beam("hfw", hfw, beam_type)
        return self._read_beam("hfw", beam_type)

    def get_working_distance(self, beam_type: BeamType) -> float:
        """Get the working distance for the specified beam type."""
        return self._read_beam("working_distance", beam_type)

    def set_working_distance(self, wd: float, beam_type: BeamType) -> float:
        """Set the working distance for the specified beam type."""
        self._write_beam("working_distance", wd, beam_type)
        return self._read_beam("working_distance", beam_type)

    def is_working_distance_settable(self, beam_type: BeamType) -> bool:
        """Whether set_working_distance actually reaches the hardware for this beam.

        The image-based autofocus sweep gates on this: on backends where the write is
        a best-effort no-op (TESCAN ION -- focus there is preset-driven, the SDK has
        no FIB working-distance control), the sweep would score images against a focus
        that never moved and report a working distance that was never applied (FIB-508).
        """
        return True

    def get_dwell_time(self, beam_type: BeamType) -> float:
        """Get the dwell time for the specified beam type."""
        return self._read_beam("dwell_time", beam_type)

    def set_dwell_time(self, dwell_time: float, beam_type: BeamType) -> float:
        """Set the dwell time for the specified beam type."""
        self._write_beam("dwell_time", dwell_time, beam_type)
        return self._read_beam("dwell_time", beam_type)

    def get_stigmation(self, beam_type: BeamType) -> Point:
        """Get the stigmation for the specified beam type."""
        return self._read_beam("stigmation", beam_type)

    def set_stigmation(self, stigmation: Point, beam_type: BeamType) -> Point:
        """Set the stigmation for the specified beam type."""
        self._write_beam("stigmation", stigmation, beam_type)
        return self._read_beam("stigmation", beam_type)

    def get_beam_shift(self, beam_type: BeamType) -> Point:
        """Get the beam shift for the specified beam type."""
        return self._read_beam("shift", beam_type)

    def set_beam_shift(self, shift: Point, beam_type: BeamType) -> Point:
        """Set the beam shift for the specified beam type."""
        self._write_beam("shift", shift, beam_type)
        return self._read_beam("shift", beam_type)

    def get_scan_rotation(self, beam_type: BeamType) -> float:
        """Get the scan rotation for the specified beam type."""
        return self._read_beam("scan_rotation", beam_type)

    def set_scan_rotation(self, rotation: float, beam_type: BeamType) -> float:
        """Set the scan rotation for the specified beam type."""
        self._write_beam("scan_rotation", rotation, beam_type)
        return self._read_beam("scan_rotation", beam_type)

    def get_detector_type(self, beam_type: BeamType) -> str:
        """Get the detector type for the specified beam type."""
        return self._read_beam("detector_type", beam_type)

    def set_detector_type(self, detector_type: str, beam_type: BeamType) -> str:
        """Set the detector type for the specified beam type."""
        self._write_beam("detector_type", detector_type, beam_type)
        return self._read_beam("detector_type", beam_type)

    def get_detector_mode(self, beam_type: BeamType) -> str:
        """Get the detector mode for the specified beam type."""
        return self._read_beam("detector_mode", beam_type)

    def set_detector_mode(self, mode: str, beam_type: BeamType) -> str:
        """Set the detector mode for the specified beam type."""
        self._write_beam("detector_mode", mode, beam_type)
        return self._read_beam("detector_mode", beam_type)

    def get_detector_contrast(self, beam_type: BeamType) -> float:
        """Get the detector contrast for the specified beam type."""
        return self._read_beam("detector_contrast", beam_type)

    def set_detector_contrast(self, contrast: float, beam_type: BeamType) -> float:
        """Set the detector contrast for the specified beam type."""
        self._write_beam("detector_contrast", contrast, beam_type)
        return self._read_beam("detector_contrast", beam_type)

    def get_detector_brightness(self, beam_type: BeamType) -> float:
        """Get the detector brightness for the specified beam type."""
        return self._read_beam("detector_brightness", beam_type)

    def set_detector_brightness(self, brightness: float, beam_type: BeamType) -> float:
        """Set the detector brightness for the specified beam type."""
        self._write_beam("detector_brightness", brightness, beam_type)
        return self._read_beam("detector_brightness", beam_type)

    def get_preset(self, beam_type: BeamType) -> Optional[str]:
        """Get the active preset for the specified beam type, or None if it has none."""
        return self._read_beam("preset", beam_type)

    def set_preset(self, preset: str, beam_type: BeamType) -> str:
        """Set the preset for the specified beam type."""
        self._write_beam("preset", preset, beam_type)
        return self._read_beam("preset", beam_type)

    def _get_compucentric_rotation_offset(self) -> FibsemStagePosition:
        """Specimen minus raw coordinates: the offset a half turn is taken about.

        `_get_compucentric_rotation_position` reflects a position through minus this,
        so a driver's `rotation_centre` c is an offset of -c -- the same centre its
        images are reprojected with (FIB-1081). Without one, the rotation centre is
        assumed to be the stage origin, as it always was here.
        """
        centre = self.rotation_centre
        if centre is None:
            return FibsemStagePosition(x=0, y=0)
        return FibsemStagePosition(x=-centre[0], y=-centre[1])

    def _compucentric_centre(self) -> Tuple[float, float]:
        """The centre a half turn is taken about: minus the compucentric offset."""
        offset = self._get_compucentric_rotation_offset()
        return (-offset.x, -offset.y)

    def _get_compucentric_rotation_position(
        self, position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """`position` turned half way round about the compucentric centre.

        :func:`fibsem.geometry.orientations.compucentric_rotation`; a stage with no
        rotation axis (a compustage) has no compucentric rotation, so it is returned as is.
        """
        if not self.system.stage.rotation:
            return position
        return compucentric_rotation(position, self._compucentric_centre())

    # The orientation reads are the stage movement service's
    # (`StageMovement.orientation`, `stage_orientation`, `current_milling_angle`, ...);
    # these keep their names and signatures. Without a stage there is no service, no
    # orientation and nothing turned over.

    @property
    def orientations(self) -> Dict[str, FibsemStagePosition]:
        """Each orientation's pose (r and t), by name: the stage movement service's."""
        if self.stage_movement is None:
            return {}
        return self.stage_movement.orientations

    @orientations.setter
    def orientations(self, table: Dict[str, FibsemStagePosition]) -> None:
        self._movement_service().orientations = table

    def get_stage_orientation(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> str:
        """The orientation ("SEM", "FIB", "MILLING", "FM" or "NONE") a stage position is
        in, the current one when None. See `StageMovement.stage_orientation`."""
        return self._movement_service().stage_orientation(stage_position)

    def get_orientation(self, orientation: str) -> FibsemStagePosition:
        """The pose (r, t) of an orientation. See `StageMovement.orientation`."""
        if self.stage_movement is None:
            raise ValueError(f"Orientation {orientation} not supported.")
        return self.stage_movement.orientation(orientation)

    def _stage_turned_over(self, tilt: float) -> bool:
        """Whether the stage has the sample turned over at this tilt, in radians."""
        if self.stage_movement is None:
            return False
        return self.stage_movement.turned_over(tilt)

    def _stage_poses(self) -> Dict[str, FibsemStagePosition]:
        """The stage's pose for each orientation name (`StageMovement.stage_poses`)."""
        if self.stage_movement is None:
            return {}
        return self.stage_movement.stage_poses()

    def _fm_is_a_pose(self) -> bool:
        """Does the stage reach the FM by re-posing rather than by travelling?
        See `StageMovement.fm_is_a_pose`."""
        return self.stage_movement is not None and self.stage_movement.fm_is_a_pose()

    def _update_orientations(self) -> None:
        """Rebuild the orientation table (`StageMovement.update_orientations`)."""
        if self.stage_movement is not None:
            self.stage_movement.update_orientations()

    def set_milling_angle(self, milling_angle: float) -> None:
        """Set the stored milling angle, in degrees: the stage movement service's
        ``milling_angle``, which rebuilds the MILLING orientation."""
        if self.stage_movement is None:
            self.system.stage.milling_angle = milling_angle
            return
        self.stage_movement.milling_angle.set_value(milling_angle)

    def get_current_milling_angle(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> float:
        """The milling angle, in degrees, a stage position gives (the current one when
        None). See `StageMovement.current_milling_angle`."""
        return self._movement_service().current_milling_angle(stage_position)

    def is_close_to_milling_angle(
        self, milling_angle: float, atol: float = 2.0
    ) -> bool:
        """Whether the current milling angle is within ``atol`` of ``milling_angle``,
        both in degrees. See `StageMovement.is_close_to_milling_angle`."""
        return self._movement_service().is_close_to_milling_angle(
            milling_angle, atol=atol
        )

    def move_to_milling_angle(
        self, milling_angle: float, rotation: Optional[float] = None
    ) -> bool:
        """Move the stage to the milling angle, based on the current pretilt and column tilt.
        Args:
            milling_angle (float): The target milling angle in radians.
            rotation (Optional[float]): The target rotation angle in radians. If None, uses the current rotation reference.
        Returns:
            bool: True if the stage is close to the target milling angle after the move, False otherwise.
        """
        return self._movement_service().move_to_milling_angle(
            milling_angle, rotation=rotation
        )

    def _beam_view_tilt(self, beam_type: BeamType) -> float:
        """Tilt of a beam column's viewing axis from the electron column, in radians.
        See `fibsem.geometry.movement.beam_view_tilt`."""
        return beam_view_tilt(beam_type, self.hardware_geometry())

    def _view_corrected_stage_movement(
        self,
        expected_y: float,
        view_tilt: float = 0.0,
    ) -> FibsemStagePosition:
        """The relative y/z stage movement for a y-displacement in a view. See
        `StageMovement.view_corrected_movement`."""
        return self._movement_service().view_corrected_movement(
            expected_y, view_tilt=view_tilt
        )

    def _view_stage_delta(
        self, dx: float, dy: float, view_tilt: float
    ) -> FibsemStagePosition:
        """:func:`fibsem.geometry.movement.image_to_stage_delta` at the current pose.
        See `StageMovement._view_stage_delta`."""
        return self._movement_service()._view_stage_delta(dx, dy, view_tilt=view_tilt)

    def _y_corrected_stage_movement(
        self,
        expected_y: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> FibsemStagePosition:
        """The relative y/z stage movement for a y-displacement in a beam's view,
        corrected for the shuttle pre-tilt. See `StageMovement.y_corrected_movement`."""
        return self._movement_service().y_corrected_movement(
            expected_y, beam_type=beam_type
        )

    def _inverse_y_corrected_stage_movement(
        self,
        dy: float,
        dz: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> float:
        """The y-displacement in a beam's view that a y/z stage movement corresponds
        to. See `StageMovement.inverse_y_corrected_movement`."""
        return self._movement_service().inverse_y_corrected_movement(
            dy, dz, beam_type=beam_type
        )

    def _inverse_view_corrected_stage_movement(
        self,
        dy: float,
        dz: float,
        view_tilt: float = 0.0,
    ) -> float:
        """The y-displacement in a view that a y/z stage movement corresponds to. See
        `StageMovement.inverse_view_corrected_movement`."""
        return self._movement_service().inverse_view_corrected_movement(
            dy, dz, view_tilt=view_tilt
        )

    def project_fm_stable_move(
        self, dx: float, dy: float, base_position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """Where the stage would end up after an FM displacement, without moving.

        The fluorescence counterpart of :meth:`project_stable_move`. That one is
        abstract and reimplemented by every driver, because it carries beam-specific
        work -- scan rotation, `beam_type` dispatch. A camera has neither, and both
        `camera_tilt` and the projection itself are already concrete here, so this
        needs no per-driver override.

        Like its beam counterpart, it maps a displacement in the image the user is
        looking at, so the display transform is undone before projecting -- the same
        role scan rotation plays for a beam. Both take the displayed frame as their
        input convention, including for synthesised displacements: `tiled.py` hands
        `project_stable_move` raw grid offsets and relies on the scan-rotation undo
        for exactly this reason.

        Args:
            dx: distance along the x-axis, in displayed image coordinates.
            dy: distance along the y-axis, in displayed image coordinates.
            base_position: the position the displacement is measured from.

        Returns:
            FibsemStagePosition: the absolute position the displacement lands on.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        return self._movement_service().project_fm_stable_move(
            dx=dx, dy=dy, base_position=base_position
        )

    def hardware_geometry(self) -> FibsemHardwareGeometry:
        """The fixed geometry this instrument is arranged in.

        The beam counterpart of :meth:`fm_image_geometry`, and its shared core. Both
        stamp the same terms onto an acquired image so it can be reprojected later
        without a live microscope to consult.

        ``fm_is_pose`` is stamped from the stage device's poses, and
        ``mirrors_back_view`` from the stage device. Recording them is what lets the
        reprojection stop inferring them from the model name (FIB-481).
        """
        stage = self.stage_device
        return FibsemHardwareGeometry.from_system_settings(
            self.system,
            fm_is_pose=self._fm_is_a_pose(),
            mirrors_back_view=stage is not None and stage.mirrors_back_view(),
            rotation_centre=self.rotation_centre,
            poses=self._stage_poses(),
            stage_frame=self.stage_frame,
        )

    @property
    def stage_frame(self) -> str:
        """The frame the stage reports positions in: the stage device's, else fibsem's."""
        return getattr(self.stage_device, "frame", STAGE_FRAME_FIBSEM)

    def record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Report a fact for the experiment's record on ``record_signal``.

        Never raises. psygnal hands a subscriber's exception back to whoever
        emitted, and the callers are acquiring and running tasks: a failure to
        record must cost the record, never the acquisition.

        ``payload`` is small plain data built by the caller -- no pixels. Kinds
        recorded today:

        * ``image_acquired`` -- from ``acquire.new_image``, with the saved path
        * ``task_step`` -- from the AutoLamella task bases
        * ``milling_stage_started`` -- from ``FibsemMillingTask``, with the stage
        * ``spot_burn_started`` -- from the spot burn service, with the field of view
        * ``stage_moved`` -- from the stage device's moves and the stage movement
          service's, the outermost only (`Stage.recording_move`)
        * ``beam_shifted`` -- from the beam device's ``shift_by``
        * ``milling`` -- from ``FibsemMillingTask``, how a run ended, stage by
          stage; from a task that milled nothing, why; and from the milling
          service's ``run`` outside a task, its name and how it ended
        * ``spot_burn_started`` -- from ``run_spot_burn``, with the field of view
        * ``stage_moved`` -- from each backend's stage moves, the outermost only
        * ``beam_shifted`` -- from each backend's ``beam_shift``
        * ``fm_image_acquired`` -- from ``fm.acquisition``, a z-stack, image or
          stitched overview, with the saved path
        * ``fm_autofocus`` -- from ``run_coarse_fine_autofocus``
        * ``alignment`` -- from ``multi_step_alignment_v2``, every step's shift
        * ``coincidence_measured`` -- from ``check_coincidence``
        * ``autofocus`` -- from ``run_auto_focus``, the working distance it left;
          and from a beam's own ``auto_focus`` (``method: instrument``)
        * ``grid_loaded`` / ``grid_unloaded`` -- from the sample handling service's
          magazine exchanges, with the grid, its magazine slot and the working slot
        """
        try:
            self.record_signal.emit(kind, payload)
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not record a {kind} event", exc_info=True)

    def _set_additional_metadata(self, image: FibsemImage) -> None:
        """Stamp who, which run, which instrument and how it is arranged onto an image.

        Lifted here from ``ThermoMicroscope`` in FIB-481: the same three lines were
        written out at eight acquisition sites across three microscope classes, so a
        change to what an image records meant finding all eight.
        """
        image.metadata.user = self.user
        # Copied, not referenced. This now carries the item and task as well as the
        # experiment (FIB-466), and those change as the run progresses -- sharing one
        # object would rewrite the item on every image already acquired. It was shared
        # before, which was harmless only because nothing mutated it.
        image.metadata.experiment = deepcopy(self.experiment)
        # Copied, not referenced. `hardware_geometry()` returns a fresh record, so
        # aliasing the live SystemInfo here would leave one field on the image a
        # snapshot and the other a window onto the microscope. TescanMicroscope
        # rewrites system.info.model/serial/software_version on every acquisition,
        # so the alias reached back into images already taken. Pre-dates FIB-481 --
        # `metadata.system = self.system` aliased the whole thing -- but the two
        # fields set together should not disagree about what they are.
        image.metadata.system_info = deepcopy(self.system.info)
        image.metadata.hardware_geometry = self.hardware_geometry()

    def fm_image_geometry(self) -> FibsemHardwareGeometry:
        """The geometry the FM is currently imaging under.

        The same record :meth:`hardware_geometry` returns, with the camera's own two
        terms filled in. Stamped onto acquired images so they can be reprojected later
        without a live microscope, and used for the live view here so that both paths
        run the same projection rather than two that merely agree today.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        if self.fm is None:
            raise ValueError("Fluorescence microscope is not available.")

        # replace() on the shared record rather than rebuilding it: the instrument
        # terms are gathered in one place, and only the camera's are added here.
        return dataclasses.replace(
            self.hardware_geometry(),
            transform=self.fm._transform or CameraImageTransform.NONE,
            camera_tilt=self.fm.camera_tilt,
        )

    def fm_stable_move(self, dx: float, dy: float) -> FibsemStagePosition:
        """Move the stage by a displacement seen in the fluorescence image.

        The fluorescence counterpart of :meth:`stable_move`: click a point in the FM
        view and the stage goes there, holding the focal plane. The projection is the
        same one the beams use, with the camera's own axis tilt
        (`FluorescenceMicroscope.camera_tilt`) in place of a column tilt, so the
        foreshortening of a tilted sample is accounted for and the movement stays in
        the sample plane -- which is what keeps the objective in focus.

        Args:
            dx: distance along the x-axis (image coordinates), as for stable_move.
            dy: distance along the y-axis (image coordinates), as for stable_move.

        Returns:
            FibsemStagePosition: the stage position after the move.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        return self._movement_service().fm_stable_move(dx=dx, dy=dy)

    # The device reads are the stage movement service's (`StageMovement.current_device`,
    # `is_at_device`, `target_position`, ...); these keep their names and signatures.

    def _get_device(self, device: str) -> StageDeviceSettings:
        """The configuration for `device`. See `StageMovement._device`."""
        return self._movement_service()._device(device)

    def get_device_origin(self, device: str) -> FibsemStagePosition:
        """Where the stage travels for `device` to see the sample. See
        `StageMovement.device_origin`."""
        return self._movement_service().device_origin(device)

    def is_at_device(
        self, device: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> bool:
        """Is the stage at `device`? See `StageMovement.is_at_device`."""
        return self._movement_service().is_at_device(device, stage_position)

    def get_current_device(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> Optional[str]:
        """Which device the stage is at, or None. See `StageMovement.current_device`."""
        return self._movement_service().current_device(stage_position)

    def get_device_imaging_state(
        self, device: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> DeviceImagingState:
        """Can `device` see the sample from where the stage is, and if not, why not.
        See `StageMovement.device_imaging_state`."""
        return self._movement_service().device_imaging_state(device, stage_position)

    def _device_translation(self, source: str, target: str) -> FibsemStagePosition:
        """The relative stage move from one device to another. See
        `StageMovement._device_translation`."""
        return self._movement_service()._device_translation(source, target)

    def get_target_position(
        self,
        stage_position: FibsemStagePosition,
        target_orientation: Optional[str] = None,
        target_device: Optional[str] = None,
    ) -> FibsemStagePosition:
        """A stage position converted across the orientation axis, the device axis, or
        both. See `StageMovement.target_position`."""
        return self._movement_service().target_position(
            stage_position,
            target_orientation=target_orientation,
            target_device=target_device,
        )

    def to_device(
        self,
        stage_position: FibsemStagePosition,
        device: str,
        orientation: Optional[str] = None,
    ) -> FibsemStagePosition:
        """*stage_position* as `device` sees it: where `move_to_device` would arrive.
        See `StageMovement.to_device`."""
        return self._movement_service().to_device(
            stage_position, device, orientation=orientation
        )

    def describe_device_imaging_state(
        self,
        device: str,
        state: Optional[DeviceImagingState] = None,
        stage_position: Optional[FibsemStagePosition] = None,
    ) -> str:
        """One sentence a person can act on, for each imaging state.

        The vocabulary is the two axes' (FIB-858): the stage travels between
        *devices* and is re-posed between *orientations*, and the failing term names
        the verb -- so a refusal built from this says which of the two to do, in
        which order, rather than announcing a false generality like "not in a valid
        orientation" for a stage that is 48.8 mm from the instrument.

        Pass `state` when it has already been asked, so a message and the decision it
        explains cannot be about two different moments.
        """
        if state is None:
            state = self.get_device_imaging_state(device, stage_position)

        instrument = (
            "the fluorescence microscope" if device == "FM" else f"the {device} device"
        )

        if state is DeviceImagingState.NO_DEVICE:
            return "This system has no fluorescence microscope."
        if state is DeviceImagingState.READY:
            return f"{instrument.capitalize()} can image the sample from here."

        orientation = self.get_stage_orientation(stage_position)
        held = (
            f"held in the {orientation} orientation"
            if orientation != "NONE"
            else "held in an unrecognised orientation"
        )
        allowed = self._get_device(device).available_orientations
        images_from = (
            f"images from the {' or '.join(allowed)} orientation"
            if allowed
            else "images from any orientation"
        )

        if state is DeviceImagingState.NEEDS_TRAVEL:
            return (
                f"The stage is {held}, which {instrument} images from, but it is "
                f"away from the {device} device: travel there "
                f"(move_to_device('{device}'))."
            )
        if state is DeviceImagingState.NEEDS_REPOSE:
            return (
                f"The stage is at the {device} device but {held}; {instrument} "
                f"{images_from}. Re-pose via the beams "
                f"(move_to_device('{device}', orientation='{allowed[0]}'))."
            )
        return (
            f"The stage is {held}, away from the {device} device; {instrument} "
            f"{images_from}. Re-pose at the beams and travel out "
            f"(move_to_device('{device}'))."
        )

    def _warn_on_fluorescence_geometry(self) -> None:
        """Warn, at connect, about FM geometry that will misbehave quietly later.

        Both cases produce no error at all in operation -- the imaging-state
        conjunction is simply never (or always) true somewhere it should not be --
        so the one loud moment available is connection, while the configuration is
        in front of the person who wrote it.
        """
        if self.fm is None:
            return

        devices = self.system.stage.devices

        # An offset mount that enabled the FM but declared no geometry inherits the
        # default -- the objective under the grid, sharing the beams' origin -- so
        # every place-term answer is about somewhere its FM is not.
        if not self._fm_is_a_pose() and devices == DEFAULT_STAGE_DEVICES:
            logging.warning(
                "A fluorescence microscope is enabled but its device entry declares "
                "no `origin`, so the FM defaults to the beams' origin. An FM on an "
                "offset mount must declare where it is, as the `origin` of its "
                "stage device entry."
            )

    def move_to_device(self, device: str, orientation: Optional[str] = None) -> None:
        """Travel to `device`, re-posing on the way when the pose has to change.

        One call that owns the safe order -- retract the objective, re-pose at the
        beams, travel out -- so a rotation never happens with the stage parked under
        an objective. The rotation guard (FIB-841) stays underneath as the last-line
        assert; the route this composes never trips it.

        `orientation` names the pose to arrive in. Omitted, the pose is carried
        across untouched whenever the target device can image from it -- that is the
        point of a traverse, and the reason this must not pass the current
        orientation's *name* through `move_to_orientation`: doing so would snap r and
        t to nominal and quietly discard a milling angle somebody dialled in. When
        the pose does have to change (an offset FM images in the FIB pose; asking
        for it from SEM used to be a refusal), the device's first declared
        acquisition orientation is used.
        """
        self._movement_service().move_to_device(device, orientation=orientation)

    @property
    def current_grid(self) -> str:
        try:
            grid = self.sample_handling.current_grid
            if grid is None:
                return "NONE"
            return grid.name
        except Exception:
            return "NONE"

    @property
    def manufacturer(self) -> str:
        """The registry name of the driver this class belongs to (its ``driver_name``)."""
        name = getattr(type(self), "driver_name", None)
        if name is None:
            raise NotImplementedError(
                f"{type(self).__name__} sets no driver_name: set it to its driver's "
                "DRIVER.manufacturer."
            )
        return name
