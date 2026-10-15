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
from fibsem.cancellation import AnyStopEvent
from fibsem.devices.beam import Beam
from fibsem.devices.core import Role, RoleUnfilled, command
from fibsem.services.core import Service
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, ImageSettings

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
    beam has one, else the sweep), ``focus_sweep`` and ``stop``."""

    electron = Role(Beam, required=False, doc="The electron beam.")
    ion = Role(Beam, required=False, doc="The ion beam.")

    def __init__(self, name: str = "imaging", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._stop_requested = threading.Event()

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
    def stop(self) -> None:
        """End the sweep in progress, as cancelled."""
        self._stop_requested.set()


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
