"""Imaging as a service: one place for a microscope's imaging, by beam type.

The beam's own operations (``acquire``, ``autocontrast``, the instrument's own
``auto_focus``, live view) are commands on the beam, which stay their
implementation. The service passes them through by ``beam_type``, so a caller
needs only ``microscope.imaging``, and adds what coordinates them over time, the
same way on every backend: ``auto_focus`` is the one place that chooses between
the instrument's autofocus and the image-based sweep, and ``focus_sweep`` is the
sweep (`fibsem.autofunctions.autofocus.run_auto_focus`), which a ``stop`` from
another caller ends.

No driver needs its own imaging code yet, so every backend builds this class.
Being a service is what lets an agent find and run these routines over the App
Server, under ``/microscope/services/imaging``.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

# At import, not under TYPE_CHECKING: the App Server decodes a command's arguments
# from its type hints (an AutoFocusSettings sent as a dict).
from fibsem.autofunctions.autofocus import AutoFocusResult, AutoFocusSettings
from fibsem.autofunctions.brightness_contrast import (
    BrightnessContrastError,
    BrightnessContrastIteration,
    BrightnessContrastResult,
    BrightnessContrastSettings,
    BrightnessContrastStepper,
    badness,
    measure,
    problems,
    unusable,
)
from fibsem.cancellation import AnyStopEvent, raise_if_cancelled
from fibsem.devices.beam import Beam, nearest_resolution
from fibsem.devices.core import Role, RoleUnfilled, command
from fibsem.services.core import Service, save_beam_conditions
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, ImageSettings

# Two beam currents closer than this, relatively, are the same current: a
# brightness and contrast that worked at one is where to start at the other.
_SAME_CURRENT = 0.05

# The role each beam fills.
_BEAM_ROLES: Dict[BeamType, str] = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}

_I = TypeVar("_I", bound="Imaging")


class Imaging(Service):
    """Imaging over the beams: the beam commands by ``beam_type`` (``acquire``,
    ``autocontrast``, ``start_live``, ``stop_live``); a saved, recorded image
    (``acquire_image``) and reference images of both beams (``reference_images``,
    ``reference_image_sets``); ``auto_focus`` (the instrument's own where the
    beam has one, else the sweep), ``focus_sweep``, ``auto_brightness_contrast``
    (from the image, not the instrument's routine) and ``stop``."""

    electron = Role(Beam, required=False, doc="The electron beam.")
    ion = Role(Beam, required=False, doc="The ion beam.")

    def __init__(self, name: str = "imaging", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._stop_requested = threading.Event()
        # What worked last, per beam and detector: [(current, (brightness,
        # contrast), response)], for auto_brightness_contrast to start from.
        self._last_brightness_contrast: Dict[tuple, List[tuple]] = {}

    def beam(self, beam_type: BeamType) -> Beam:
        """The beam that images as ``beam_type``; a ValueError when this microscope
        has none (the column is disabled)."""
        try:
            return getattr(self, _BEAM_ROLES[beam_type])
        except RoleUnfilled:
            raise ValueError(f"The {beam_type.name} beam is not enabled.") from None

    # -- the beam commands, by beam type ------------------------------------------
    # One line each: the beam does the work, here or under /microscope/devices.

    @command
    def acquire(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        image_settings: Optional[ImageSettings] = None,
    ) -> FibsemImage:
        """Acquire an image with ``beam_type``: with ``image_settings`` when given
        (their own beam type wins), else with the beam's current settings."""
        if image_settings is not None and image_settings.beam_type is not None:
            beam_type = image_settings.beam_type
        return self.beam(beam_type).acquire(image_settings)

    @command
    def autocontrast(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        reduced_area: Optional[FibsemRectangle] = None,
    ) -> None:
        """Run ``beam_type``'s brightness and contrast routine, optionally on a
        rectangle of the frame (0 to 1)."""
        self.beam(beam_type).autocontrast(reduced_area)

    @command
    def start_live(self, beam_type: BeamType = BeamType.ELECTRON) -> None:
        """Image continuously with ``beam_type``'s current settings until
        `stop_live`; the frames come on the beam's ``live_frame``."""
        self.beam(beam_type).start_live()

    @command
    def stop_live(self, beam_type: BeamType = BeamType.ELECTRON) -> None:
        """Stop ``beam_type``'s live view. Safe when it isn't live."""
        self.beam(beam_type).stop_live()

    # -- images for the record -----------------------------------------------------

    @command
    def acquire_image(self, image_settings: ImageSettings) -> FibsemImage:
        """An image as a task takes one: autocontrast first when the settings ask,
        acquire with the settings, save to ``<path>/<filename>_eb`` (or ``_ib``)
        when ``save`` is set, and record it as ``image_acquired``. The saved file
        is the image's ``filepath``."""
        beam_type = image_settings.beam_type
        if image_settings.autocontrast:
            self.autocontrast(beam_type, image_settings.reduced_area)
        image = self.acquire(beam_type, image_settings)
        path = None
        if image_settings.save:
            suffix = "eb" if beam_type is BeamType.ELECTRON else "ib"
            path = image.save(
                path=os.path.join(
                    image_settings.path, f"{image_settings.filename}_{suffix}"
                )
            )
        self.session.record("image_acquired", acquisition_record(image, path))
        return image

    @command
    def reference_images(
        self,
        image_settings: ImageSettings,
        electron: bool = True,
        ion: bool = True,
    ) -> List[Optional[FibsemImage]]:
        """An `acquire_image` from each beam asked for, with the same settings:
        ``[electron, ion]``, None for a beam not asked for. The settings given
        are not changed."""
        images: List[Optional[FibsemImage]] = []
        for beam_type, wanted in ((BeamType.ELECTRON, electron), (BeamType.ION, ion)):
            if not wanted:
                images.append(None)
                continue
            settings = copy.copy(image_settings)
            settings.beam_type = beam_type
            images.append(self.acquire_image(settings))
        return images

    @command
    def reference_image_sets(
        self,
        image_settings: ImageSettings,
        hfws: Sequence[float],
        filename: str = "ref_image",
        electron: bool = True,
        ion: bool = True,
    ) -> List[List[Optional[FibsemImage]]]:
        """`reference_images` at each field of view in ``hfws``, always saved, as
        ``<filename>_res_01``, ``_res_02`` ... in order: one ``[electron, ion]``
        pair per field of view. The settings given are not changed."""
        settings = copy.deepcopy(image_settings)
        settings.save = True
        sets = []
        for i, hfw in enumerate(hfws, start=1):
            settings.hfw = hfw
            settings.filename = f"{filename}_res_{i:02d}"
            sets.append(self.reference_images(settings, electron=electron, ion=ion))
        return sets

    # -- routines ------------------------------------------------------------------

    @command
    def auto_focus(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        reduced_area: Optional[FibsemRectangle] = None,
    ) -> Optional[AutoFocusResult]:
        """Focus ``beam_type``: with the instrument's own autofocus where the beam
        has one, else with the image-based sweep at the beam's field of view.

        Returns the sweep's result, or None after the instrument's autofocus (it
        reports nothing) or a sweep this beam can't run (its working distance can't
        be set).
        """
        beam = self.beam(beam_type)
        if beam.commands["auto_focus"].available:
            beam.auto_focus(reduced_area)
            return None
        logging.debug(
            f"{beam_type.name} has no autofocus of its own; running the focus sweep."
        )
        return self.focus_sweep(
            beam_type=beam_type, settings=AutoFocusSettings(reduced_area=reduced_area)
        )

    @command
    def focus_sweep(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        hfw: Optional[float] = None,
        settings: Optional[AutoFocusSettings] = None,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
        path: Optional[str] = None,
        name: str = "autofocus",
    ) -> Optional[AutoFocusResult]:
        """Focus ``beam_type`` by sweeping its working distance and scoring an image
        at each step (`fibsem.autofunctions.autofocus.run_auto_focus`).

        ``hfw`` is the probe images' field of view, the beam's own when None. A
        ``stop_event`` set, or ``stop`` called, while it runs puts the working
        distance back and raises `OperationCancelledError`. Given a ``path``, the
        result is saved to ``<path>/<name>``. None when the beam's working distance
        can't be set, so there was nothing to sweep.
        """
        from fibsem.autofunctions.autofocus import run_auto_focus

        beam = self.beam(beam_type)
        if hfw is None:
            hfw = beam.hfw.get_value()
        self._stop_requested.clear()
        events = [e for e in (stop_event, self._stop_requested) if e is not None]
        return run_auto_focus(
            self.parent,
            beam_type=beam_type,
            hfw=hfw,
            settings=settings,
            stop_event=AnyStopEvent(*events),
            path=path,
            name=name,
        )

    @command
    def auto_brightness_contrast(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        settings: Optional[BrightnessContrastSettings] = None,
    ) -> BrightnessContrastResult:
        """Set ``beam_type``'s detector brightness and contrast from its own probe
        images, not the instrument's routine: few images, and never left mostly
        black or white.

        Starts from what last worked at this beam current and detector, so most
        runs take one image. An image counts when at most ``max_clipped`` of its
        pixels sit at either limit, it is centred and it isn't flat; until then the
        stepper moves both controls
        (`fibsem.autofunctions.brightness_contrast`), for at most ``max_images``.
        The probe images are at the beam's field of view, with the probe
        resolution and dwell time, which are put back after, and by default scan
        and judge only the centre half of the frame (``reduced_area``).

        Ending short of the targets, the detector is left at the best image's
        controls and the result says it didn't converge; ending on an image that
        is mostly black, mostly white or flat, it raises
        `BrightnessContrastError`, so a workflow stops rather than align or mill on
        it. ``stop`` ends it, as cancelled, with the detector where it was.
        """
        settings = settings or BrightnessContrastSettings()
        beam = self.beam(beam_type)
        brightness, contrast = beam.detector_brightness, beam.detector_contrast
        ranges = [_range(param) for param in (brightness, contrast)]
        key, current = self._detector_key(beam)
        initial = (brightness.get_value(), contrast.get_value())

        start, response = self._remembered(key, current, settings)
        if start is not None:
            for param, value in zip((brightness, contrast), start):
                param.set_value(value)

        probe = ImageSettings(
            beam_type=beam_type,
            resolution=nearest_resolution(
                settings.probe_resolution, beam.resolution.choices
            ),
            dwell_time=settings.probe_dwell_time,
            hfw=beam.hfw.get_value(),
            reduced_area=settings.reduced_area,
            autocontrast=False,
            save=False,
        )
        stepper = BrightnessContrastStepper(settings, response)
        saved = save_beam_conditions(beam, ("resolution", "dwell_time"))
        self._stop_requested.clear()
        iterations: List[BrightnessContrastIteration] = []
        try:
            for i in range(settings.max_images):
                raise_if_cancelled(self._stop_requested)
                stats = measure(beam.acquire(probe).data, settings)
                controls = (brightness.get_value(), contrast.get_value())
                iterations.append(BrightnessContrastIteration(*controls, stats))
                logging.debug(
                    {
                        "msg": "auto_brightness_contrast",
                        "beam_type": beam_type.name,
                        "image": i,
                        "brightness": controls[0],
                        "contrast": controls[1],
                        "stats": stats.to_dict(),
                    }
                )
                if not problems(stats, settings) or i == settings.max_images - 1:
                    break
                step = stepper.next(
                    [_fraction(v, r) for v, r in zip(controls, ranges)], stats
                )
                for param, value, r in zip((brightness, contrast), step, ranges):
                    param.set_value(r[0] + value * (r[1] - r[0]))
        except BaseException:
            for param, value in zip((brightness, contrast), initial):
                param.set_value(value)
            raise
        finally:
            for name, value in saved.items():
                try:
                    getattr(beam, name).write_through(value)
                except Exception:
                    logging.exception(f"Failed to restore the beam {name}")

        best = iterations[-1]
        if problems(best.stats, settings):
            best = min(
                iterations,
                key=lambda it: (
                    unusable(it.stats, settings),
                    badness(it.stats, settings),
                ),
            )
        if best is not iterations[-1]:
            brightness.set_value(best.brightness)
            contrast.set_value(best.contrast)
        result = BrightnessContrastResult(
            brightness=best.brightness,
            contrast=best.contrast,
            stats=best.stats,
            converged=not problems(best.stats, settings),
            iterations=iterations,
            started_from_last=start is not None,
        )
        record = {"beam_type": beam_type.name, **result.to_dict()}
        if unusable(best.stats, settings):
            self.parent.record_event(
                "brightness_contrast", {**record, "status": "failed"}
            )
            raise BrightnessContrastError(
                f"{beam_type.name} brightness and contrast: after {len(iterations)} "
                f"images, the best is still unusable "
                f"({'; '.join(problems(best.stats, settings))})."
            )
        if result.converged:
            self._remember(
                key, current, (best.brightness, best.contrast), stepper.response
            )
        else:
            logging.warning(
                f"{beam_type.name} brightness and contrast short of the targets after "
                f"{len(iterations)} images: {'; '.join(problems(best.stats, settings))}"
            )
        self.parent.record_event(
            "brightness_contrast", {**record, "status": "completed"}
        )
        return result

    def _detector_key(self, beam: Beam) -> tuple:
        """Which detector the beam images with, and its current: what a brightness
        and contrast that worked is good for again."""
        values = [
            beam.beam_type,
            *(
                getattr(beam, name).get_value() if name in beam.parameters else None
                for name in ("detector_type", "detector_mode")
            ),
        ]
        current = beam.current.get_value() if "current" in beam.parameters else None
        return tuple(values), current

    def _remembered(
        self, key: tuple, current: Optional[float], settings: BrightnessContrastSettings
    ) -> tuple:
        """The brightness and contrast that last worked for this detector at this
        current, and the detector response learned then; (None, None) when none."""
        if not settings.start_from_last:
            return None, None
        for seen, controls, response in reversed(
            self._last_brightness_contrast.get(key, [])
        ):
            if _same_current(seen, current):
                return controls, response
        return None, None

    def _remember(
        self, key: tuple, current: Optional[float], controls: tuple, response: Any
    ) -> None:
        entries = [
            entry
            for entry in self._last_brightness_contrast.get(key, [])
            if not _same_current(entry[0], current)
        ]
        entries.append((current, controls, response.tolist()))
        self._last_brightness_contrast[key] = entries

    @command
    def stop(self) -> None:
        """End the sweep or brightness and contrast run in progress, as cancelled."""
        self._stop_requested.set()


def _range(param: Any) -> tuple:
    """A control's (min, max), from its limits; 0 to 1 when it has none."""
    limits = param.limits
    if limits is None:
        return (0.0, 1.0)
    return (float(limits.min), float(limits.max))


def _fraction(value: float, limits: tuple) -> float:
    return (value - limits[0]) / (limits[1] - limits[0])


def _same_current(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= _SAME_CURRENT * max(abs(a), abs(b), 1e-15)


def acquisition_record(image: FibsemImage, path: Optional[str]) -> Dict[str, Any]:
    """What the experiment's record says about an acquisition: how, and where it went.

    ``path`` is the file actually written -- suffix and extension included, which
    the log's ``filename`` never had -- or None when this acquisition was not
    saved here, which is most of them. None does not mean no copy exists: the
    alignment and autofocus diagnostics save their own images, outside this path.
    Not the pixels, and not the full metadata: a saved file embeds that.
    """
    record: Dict[str, Any] = {"path": path, "shape": list(image.data.shape)}
    try:
        settings = image.metadata.image_settings
        record.update(
            beam_type=settings.beam_type.name,
            filename=settings.filename,
            hfw=settings.hfw,
            dwell_time=settings.dwell_time,
            reduced_area=(
                settings.reduced_area.to_dict()
                if settings.reduced_area is not None
                else None
            ),
            pixel_size=image.metadata.pixel_size.x,
            stage_position=image.metadata.microscope_state.stage_position.to_dict(),
        )
    except Exception:  # noqa: BLE001 - an image without full metadata still says where it went
        logging.debug("recording an acquisition without its metadata", exc_info=True)
    return record


def bind_imaging(service: Type[_I], microscope: Any) -> Optional[_I]:
    """Build a microscope's imaging service of class *service* over its beams, or
    None when it has no beam (every column disabled), and ``auto_focus`` then
    raises."""
    beams = microscope.beams
    if not beams:
        return None
    imaging = service(parent=microscope)
    imaging.fill_roles(
        **{_BEAM_ROLES[beam_type]: beam for beam_type, beam in beams.items()}
    )
    imaging.connect()
    return imaging
