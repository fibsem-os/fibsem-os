"""Imaging as a service: one place for a microscope's imaging, by beam type.

The beam's own operations (``acquire``, ``autocontrast``, the instrument's own
``auto_focus``, live view) are commands on the beam, which stay their
implementation. The service passes them through by ``beam_type``, so a caller
needs only ``microscope.imaging``, and adds what coordinates them over time, the
same way on every backend: ``auto_focus`` is the one place that chooses between
the instrument's autofocus and the image-based sweep, and ``focus_sweep`` is the
sweep (`fibsem.autofunctions.autofocus.run_auto_focus`). Tiled overviews, beam
(`fibsem.imaging.tiled.acquire_tiled_overview`) and fluorescence
(`fibsem.fm.acquisition.acquire_fm_tiled_overview`), and focus stacks run here
too. A ``stop`` from another caller ends any of these.

No driver needs its own imaging code yet, so every backend builds this class.
Being a service is what lets an agent find and run these routines over the App
Server, under ``/microscope/services/imaging``.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from psygnal import Signal

if TYPE_CHECKING:
    from fibsem.imaging.tiling.result import OverviewResult

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
from fibsem.cancellation import (
    AnyStopEvent,
    OperationCancelledError,
    raise_if_cancelled,
)
from fibsem.devices.beam import Beam, nearest_resolution, scanned_resolution
from fibsem.devices.core import Role, RoleUnfilled, command, query
from fibsem.devices.fm import FM
from fibsem.fm.microscope import AcquisitionRefused
from fibsem.fm.structures import (
    AutoFocusMode,
    ChannelSettings,
    OverviewParameters,
    ZParameters,
)
from fibsem.imaging.tiling.progress import TiledProgress
from fibsem.services.core import Service, save_beam_conditions
from fibsem.structures import (
    BeamType,
    DeviceImagingState,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
)

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
    ``reference_image_sets``) and a focus stacked one (``acquire_focus_stack``); tiled
    overviews (``acquire_overview``, ``estimate_overview``) and fluorescence
    ones (``fm_acquire_overview``, ``fm_estimate_overview``); ``auto_focus``
    (the instrument's own where the beam has one, else the sweep),
    ``focus_sweep``, ``auto_brightness_contrast`` (from the image, not the
    instrument's routine), the fluorescence microscope's ``fm_auto_focus``
    and ``fm_acquire_image`` (an image or a z-stack), and ``stop``, which ends
    any of them."""

    electron = Role(Beam, required=False, doc="The electron beam.")
    ion = Role(Beam, required=False, doc="The ion beam.")
    fm = Role(FM, required=False, doc="The fluorescence microscope.")

    # Tile progress (`TiledProgress`) from a beam or fluorescence tiled overview,
    # however it was started: each tile, the stitch, and how the run ended. Emitted
    # on the acquiring thread. A Signal, not a Parameter: it carries a preview image.
    tiled_progress = Signal(TiledProgress)
    # An overview's mosaic once saved, beam (`FibsemImage`) or fluorescence
    # (`FluorescenceImage`), with the path it was written to. A fluorescence
    # overview emits it before its run reports finished on ``tiled_progress``,
    # a beam overview after. Emitted on the acquiring thread.
    overview_acquired = Signal(object, object)
    # Images someone asked the app's SEM and FIB views to show (`show`): a milling
    # task's result, a route's image. Not every acquisition: alignment and other
    # reduced-area images have no place in the views, and click-to-move and pattern
    # placement work on the image shown until something replaces it on purpose.
    # Live frames are not here either; the views take them from ``live_frame``.
    image_shown = Signal(FibsemImage)

    def __init__(self, name: str = "imaging", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._stop_requested = threading.Event()
        self._depth = 0  # routines running, a sweep inside a focus stack counted
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

    # -- images for the views ------------------------------------------------------

    def show(self, image: FibsemImage) -> None:
        """Send ``image`` to the app's SEM or FIB view, by its own beam type. Only
        what is sent here is shown; an acquisition alone never is."""
        self.image_shown.emit(image)

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
        path = (
            image.save(path=_save_path(image_settings)) if image_settings.save else None
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

    def focus_stacked_image(
        self,
        image_settings: ImageSettings,
        n_steps: int = 3,
        auto_focus: bool = True,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    ) -> FibsemImage:
        """One image focused in ``n_steps`` horizontal strips: each strip is
        acquired on its own (`acquire_image`, saving nothing), after `auto_focus`
        on that strip when ``auto_focus`` is set, and the strips are stacked into
        a full frame. Saved to ``<path>/<filename>_eb`` (or ``_ib``) when the
        settings' ``save`` is set. The metadata is the middle strip's.

        A ``stop_event`` set raises `OperationCancelledError` before the next
        strip. The beam goes back to full frame scanning however it ends. The
        settings given are not changed. `acquire_focus_stack` is this as a command; a
        tiled overview with a focus stack calls it per tile.
        """
        import numpy as np
        from skimage.transform import resize

        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        settings = copy.copy(image_settings)
        settings.save = False
        height = 1.0 / n_steps
        strips: List[FibsemImage] = []
        try:
            for i in range(n_steps):
                raise_if_cancelled(stop_event, "Focus stack cancelled.")
                settings.reduced_area = FibsemRectangle(
                    left=0, top=i * height, width=1, height=height
                )
                if auto_focus:
                    self.auto_focus(settings.beam_type, settings.reduced_area)
                    raise_if_cancelled(stop_event, "Focus stack cancelled.")
                strips.append(self.acquire_image(settings))
        finally:
            self.parent.set_full_frame_scanning_mode(settings.beam_type)

        data = np.vstack([strip.data for strip in strips])
        shape = tuple(settings.resolution[::-1])
        if data.shape != shape:
            data = resize(data, shape, preserve_range=True).astype(np.uint8)
        metadata = copy.deepcopy(strips[(n_steps - 1) // 2].metadata)
        metadata.image_settings.reduced_area = None
        stacked = FibsemImage(data=data, metadata=metadata)
        if image_settings.save:
            stacked.save(_save_path(image_settings))
        return stacked

    @command(long=True)
    def acquire_focus_stack(
        self,
        image_settings: ImageSettings,
        path: str,
        n_steps: int = 3,
        auto_focus: bool = True,
    ) -> Dict[str, Any]:
        """A `focus_stacked_image` saved under ``path``, as
        ``<path>/<filename>_eb`` (or ``_ib``), whatever the settings' own path
        and ``save`` say. ``stop`` ends it before the next strip, ``cancelled``.
        Answers the stacked image."""
        settings = copy.copy(image_settings)
        settings.path = path
        settings.save = True
        try:
            with self._routine():
                image = self.focus_stacked_image(
                    settings, n_steps, auto_focus, stop_event=self._stop_requested
                )
        except OperationCancelledError:
            return {"status": "cancelled", "image": None}
        return {"status": "completed", "image": image}

    # -- overviews -----------------------------------------------------------------

    @command(long=True)
    def acquire_overview(
        self,
        settings: OverviewAcquisitionSettings,
        path: str,
        name: str = "overview",
        centre_position: Optional[FibsemStagePosition] = None,
    ) -> Dict[str, Any]:
        """A tiled overview with ``settings.image_settings.beam_type``, centred
        on ``centre_position`` (None for where the stage is); the stage goes back
        to where it started after (`fibsem.imaging.tiled.acquire_tiled_overview`).
        Saved under ``path``: the stitched image as ``<name>-<time>``, its tiles
        in a folder of that name. ``stop`` ends it before the next tile,
        ``cancelled``, keeping the tiles acquired. Answers the stitched image,
        the tiles' paths, and the overview: the run's `OverviewResult.summary`
        (the settings as used, the layout check, the registration, the timing and
        the record's path)."""
        from fibsem.imaging.tiled import acquire_tiled_overview, stamped_overview_name
        from fibsem.imaging.tiling.progress import MODALITY_BEAM
        from fibsem.imaging.tiling.result import OverviewResult

        result = OverviewResult(MODALITY_BEAM)
        try:
            with self._routine():
                acquire_tiled_overview(
                    self.parent,
                    settings,
                    path,
                    stamped_overview_name(name),
                    centre_position=centre_position,
                    stop_event=self._stop_requested,
                    result=result,
                )
        except OperationCancelledError:
            return _overview_answer(result, image=None)
        self.overview_acquired.emit(result.image, result.mosaic_path)
        return _overview_answer(result, image=result.image)

    @query
    def estimate_overview(
        self, settings: OverviewAcquisitionSettings
    ) -> Dict[str, Any]:
        """What a `acquire_overview` with ``settings`` covers and scans: the tiles
        it would acquire, the beam-on time in seconds, and the field of view in
        metres. Not how long it takes: the stage moves between tiles are not in
        it (`OverviewAcquisitionSettings.scan_time`).

        Also the ``resolution`` the beam scans the tiles at, with the
        ``requested_resolution`` when that is another (the instrument only scans
        the sizes it lists), and the tiles the stage can't reach from where it is
        now, ``unreachable`` as [row, col]: a run with any is refused until they
        are turned off in ``tile_mask``."""
        from fibsem.imaging.tiling.reach import reachable_overview

        settings = copy.deepcopy(settings)
        image_settings = settings.image_settings
        asked = tuple(image_settings.resolution)
        image_settings.resolution = scanned_resolution(
            self.parent, image_settings.beam_type, asked
        )
        _, unreachable = reachable_overview(
            self.parent, settings, self.parent.get_stage_position()
        )
        answer = {
            "tiles": settings.n_enabled_tiles,
            "nrows": settings.nrows,
            "ncols": settings.ncols,
            "scan_time": settings.scan_time,
            "fov": [settings.total_fov_x, settings.total_fov_y],
            "resolution": list(image_settings.resolution),
            "unreachable": [list(tile) for tile in unreachable],
        }
        if tuple(image_settings.resolution) != asked:
            answer["requested_resolution"] = list(asked)
        return answer

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
    ) -> Union[AutoFocusResult, Dict[str, Any], None]:
        """Focus ``beam_type`` by sweeping its working distance and scoring an image
        at each step (`fibsem.autofunctions.autofocus.run_auto_focus`). The result
        has each step's working distance and score, the one it moved to, and
        ``at_edge`` when the best score was at an end of the range searched.

        ``hfw`` is the probe images' field of view, the beam's own when None. A
        ``stop_event`` set, or ``stop`` called, while it runs puts the working
        distance back and raises `OperationCancelledError`. Given a ``path``, the
        result is saved to ``<path>/<name>``.

        Where the beam's working distance can't be set (Tescan's ion beam), there
        is nothing to sweep: the beam's own autofocus runs instead, and the answer
        says so (``focused_with: instrument``, the ``reason`` and the working
        distance after). None when the beam has no autofocus either.
        """
        from fibsem.autofunctions.autofocus import run_auto_focus

        beam = self.beam(beam_type)
        if hfw is None:
            hfw = beam.hfw.get_value()
        events = [e for e in (stop_event, self._stop_requested) if e is not None]
        with self._routine():
            result = run_auto_focus(
                self.parent,
                beam_type=beam_type,
                hfw=hfw,
                settings=settings,
                stop_event=AnyStopEvent(*events),
                path=path,
                name=name,
            )
        if result is not None or not beam.commands["auto_focus"].available:
            return result
        reduced_area = settings.reduced_area if settings is not None else None
        beam.auto_focus(reduced_area)
        return {
            "focused_with": "instrument",
            "reason": (
                f"The {beam_type.name} beam's working distance can't be set on "
                "this system, so its own autofocus ran instead of the sweep."
            ),
            "working_distance": beam.working_distance.get_value(),
        }

    def _has_fm(self) -> bool:
        return "fm" in self._roles

    def _refuse_while_fm_live(self, what: str) -> None:
        """Refuse *what* while the FM device's own live view runs. The FM's
        `acquiring` sees the FM tab's stream, but not live view started on the
        device itself (its ``start_live`` command, as an agent starts it)."""
        if self.fm.is_live:
            raise AcquisitionRefused(
                f"Cannot start {what}: the fluorescence microscope is in use "
                "(live view). Stop it first."
            )

    @command(long=True, available=_has_fm)
    def fm_auto_focus(
        self,
        settings: Optional[AutoFocusSettings] = None,
        path: Optional[str] = None,
        name: str = "autofocus",
        channel: Optional[ChannelSettings] = None,
    ) -> Dict[str, Any]:
        """Focus the fluorescence microscope by sweeping its objective, a coarse
        then a fine pass by default, scoring a frame at each step on its current
        channel (``settings.channel_name`` picks another, and ``channel`` sets
        one outright, as the FM tab's selected channel does)
        (`fibsem.fm.calibration.run_coarse_fine_autofocus`).

        Refused while the FM is in use (live view, the device's own included, a
        tileset, the tab's own sweep), and unless the stage is at the FM's pose:
        from any other pose the objective sweeps where it cannot see the sample,
        so the scores mean nothing. ``stop`` ends it between
        steps, ``cancelled``, with the objective where the passes left it. Given
        a ``path``, the result is saved to ``<path>/<name>``. Answers the best
        objective position and the sweep's scores.
        """
        from fibsem.fm.calibration import run_coarse_fine_autofocus

        fm = self.parent.fm
        self._refuse_while_fm_live("auto-focus")
        state = self.parent.get_device_imaging_state("FM")
        if state is not DeviceImagingState.READY:
            raise AcquisitionRefused(
                "Cannot start auto-focus. "
                f"{self.parent.explain_imaging_state('FM', state)}"
            )
        with self._routine(), fm.acquiring("auto-focus"):
            result = run_coarse_fine_autofocus(
                fm,
                autofocus_settings=settings or AutoFocusSettings(),
                channel_settings=channel,
                stop_event=self._stop_requested,
                path=path,
                name=name,
            )
        if result is None or self._stop_requested.is_set():
            return {"status": "cancelled", **(result.to_dict() if result else {})}
        return {"status": "completed", **result.to_dict()}

    @command(long=True, available=_has_fm)
    def fm_acquire_image(
        self,
        channels: List[ChannelSettings],
        path: str,
        name: str = "image",
        zparams: Optional[ZParameters] = None,
    ) -> Dict[str, Any]:
        """A fluorescence image in ``channels`` where the stage and objective
        are, a z-stack about the objective's position when ``zparams`` is given
        (`fibsem.fm.acquisition.acquire_image`), and the objective back after.
        Saved under ``path`` as ``<name>-<time>.ome.tiff``.

        Refused while the FM is in use or can't see the sample from this pose.
        ``stop`` ends it ``cancelled`` before the next frame, saving nothing.
        Answers the image's path and a summary of it.
        """
        from fibsem.fm.acquisition import acquire_image
        from fibsem.imaging.tiled import stamped_overview_name

        fm = self.parent.fm
        self._refuse_while_fm_live("acquisition")
        filename = os.path.join(path, f"{stamped_overview_name(name)}.ome.tiff")
        with self._routine(), fm.acquiring("acquisition"):
            os.makedirs(path, exist_ok=True)
            image = acquire_image(
                fm,
                list(channels),
                zparams=zparams,
                stop_event=self._stop_requested,
                filename=filename,
            )
        if image is None:
            return {"status": "cancelled", "image": None}
        saved = filename if os.path.isfile(filename) else None
        answer = {"summary": _fm_image_summary(image), "path": saved}
        return {"status": "completed", "image": answer}

    @command(long=True, available=_has_fm)
    def fm_acquire_overview(
        self,
        channels: List[ChannelSettings],
        parameters: OverviewParameters,
        path: str,
        name: str = "overview",
        zparams: Optional[ZParameters] = None,
        autofocus_settings: Optional[AutoFocusSettings] = None,
        centre_position: Optional[FibsemStagePosition] = None,
    ) -> Dict[str, Any]:
        """A fluorescence overview in ``channels``, a grid of ``parameters.rows``
        x ``parameters.cols`` camera fields centred on ``centre_position`` (None
        for where the stage is); the stage and the objective go back to where
        they started after
        (`fibsem.fm.acquisition.acquire_fm_tiled_overview`). Saved under
        ``path``: the mosaic as ``<name>-<time>.ome.tiff``, its tiles in a folder
        of that name. ``zparams`` z-stacks each tile when ``parameters.use_zstack``
        is set; ``autofocus_settings`` is the sweep for ``parameters.autofocus_mode``
        (default passes when not given).

        Refused while the FM is in use or can't see the sample from this pose.
        ``stop`` ends it before the next tile, ``cancelled``, keeping the tiles
        acquired. Answers the mosaic's path and a summary of it, the tiles'
        paths, and the layout, as `tiled_overview`'s.
        """
        from fibsem.fm.acquisition import (
            OverviewDestination,
            acquire_fm_tiled_overview,
        )
        from fibsem.imaging.tiled import stamped_overview_name
        from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE
        from fibsem.imaging.tiling.result import OverviewResult

        autofocus_settings = _fm_overview_focus(parameters, autofocus_settings)
        fm = self.parent.fm
        self._refuse_while_fm_live("overview acquisition")
        result = OverviewResult(MODALITY_FLUORESCENCE)
        try:
            with self._routine(), fm.acquiring("overview acquisition"):
                os.makedirs(path, exist_ok=True)
                acquire_fm_tiled_overview(
                    self.parent,
                    channels,
                    parameters,
                    OverviewDestination.create(path, stamped_overview_name(name)),
                    zparams=zparams if parameters.use_zstack else None,
                    autofocus_settings=autofocus_settings,
                    centre_position=centre_position,
                    stop_event=self._stop_requested,
                    on_acquired=self.overview_acquired.emit,
                    result=result,
                )
        except OperationCancelledError:
            return _overview_answer(result, image=None)
        image = {"summary": _fm_image_summary(result.image), "path": result.mosaic_path}
        return _overview_answer(result, image=image)

    @query
    def fm_estimate_overview(
        self,
        channels: List[ChannelSettings],
        parameters: OverviewParameters,
        zparams: Optional[ZParameters] = None,
        autofocus_settings: Optional[AutoFocusSettings] = None,
    ) -> Dict[str, Any]:
        """What an `fm_acquire_overview` with these arguments would do: the tiles,
        the images it takes (channels x z planes per tile), and how long it runs
        in seconds, the stage and objective moves and focus sweeps counted
        (`fibsem.fm.timing.estimate_fm_overview_time`). Also the tiles the stage
        can't reach from where it is now, ``unreachable`` as [row, col], as
        `estimate_overview`'s."""
        from fibsem.fm.timing import (
            calculate_total_images_count,
            estimate_fm_overview_time,
        )
        from fibsem.imaging.tiling.reach import reachable_fluorescence_overview

        autofocus_settings = _fm_overview_focus(parameters, autofocus_settings)
        zparams = zparams if parameters.use_zstack else None
        tiles = parameters.n_enabled_tiles
        _, unreachable = reachable_fluorescence_overview(
            self.parent, parameters, self.parent.get_stage_position()
        )
        return {
            "tiles": tiles,
            "nrows": parameters.rows,
            "ncols": parameters.cols,
            "images": tiles * calculate_total_images_count(list(channels), zparams),
            "duration": estimate_fm_overview_time(
                channels, parameters, zparams, autofocus_settings
            ),
            "unreachable": [list(tile) for tile in unreachable],
        }

    @contextmanager
    def _routine(self) -> Iterator[None]:
        """Run a routine that ``stop`` ends. The stop is cleared when the
        outermost one starts only, so a sweep inside a focus stack or an
        overview tile doesn't drop a stop meant for the whole run."""
        if self._depth == 0:
            self._stop_requested.clear()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    @command
    def auto_brightness_contrast(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        settings: Optional[BrightnessContrastSettings] = None,
        path: Optional[str] = None,
        name: str = "brightness_contrast",
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
        it. The error carries the result. ``stop`` ends it, as cancelled, with the
        detector where it was.

        Given a ``path``, the result (its probe images, ``data.json`` and the
        diagnostic plot) is saved to ``<path>/<name>``, also when it raises
        `BrightnessContrastError`.
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
        iterations: List[BrightnessContrastIteration] = []
        with self._routine():
            try:
                for i in range(settings.max_images):
                    raise_if_cancelled(self._stop_requested)
                    data = beam.acquire(probe).data
                    stats = measure(data, settings)
                    controls = (brightness.get_value(), contrast.get_value())
                    iterations.append(
                        BrightnessContrastIteration(*controls, stats, image=data)
                    )
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
                for param_name, value in saved.items():
                    try:
                        getattr(beam, param_name).write_through(value)
                    except Exception:
                        logging.exception(f"Failed to restore the beam {param_name}")

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
            initial_brightness=initial[0],
            initial_contrast=initial[1],
            usable=not unusable(best.stats, settings),
            beam_type=beam_type.name,
            settings=settings,
        )
        if path is not None:
            try:
                result.save(path, name)
            except Exception:
                logging.exception("Failed to save the brightness and contrast result")
        record = result.to_dict()
        if not result.usable:
            self.parent.record_event(
                "brightness_contrast", {**record, "status": "failed"}
            )
            raise BrightnessContrastError(
                f"{beam_type.name} brightness and contrast: after {len(iterations)} "
                f"images, the best is still unusable "
                f"({'; '.join(problems(best.stats, settings))}).",
                result=result,
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
        """End the sweep or brightness and contrast run in progress, a beam's or
        the FM's, a tiled overview, beam or fluorescence, or a focus stack, as
        cancelled, and an FM z-stack before its next frame."""
        self._stop_requested.set()
        if self._has_fm():
            self.fm.cancel()


def _overview_answer(result: "OverviewResult", image: Any) -> Dict[str, Any]:
    """What an overview command answers: its status, the mosaic (None when it was
    stopped), the tiles' paths, and the run's short record."""
    return {
        "status": result.status,
        "image": image,
        "tile_paths": result.tile_paths,
        "overview": result.summary(),
    }


def _save_path(image_settings: ImageSettings) -> str:
    """Where an image taken with ``image_settings`` is saved:
    ``<path>/<filename>_eb``, or ``_ib`` for the ion beam.

    Raises:
        ValueError: the settings name no folder to save in.
    """
    if image_settings.path is None:
        raise ValueError(
            "The image settings ask to save the image but name no folder: "
            "set image_settings.path."
        )
    suffix = "eb" if image_settings.beam_type is BeamType.ELECTRON else "ib"
    return os.path.join(image_settings.path, f"{image_settings.filename}_{suffix}")


def _fm_overview_focus(
    parameters: OverviewParameters, settings: Optional[AutoFocusSettings]
) -> Optional[AutoFocusSettings]:
    """The focus sweep for a fluorescence overview: the one given, else the
    default passes when ``parameters.autofocus_mode`` focuses at all."""
    if settings is None and parameters.autofocus_mode is not AutoFocusMode.NONE:
        return AutoFocusSettings()
    return settings


def _fm_image_summary(image: Any) -> Dict[str, Any]:
    """What an agent reads about a fluorescence image: its shape, channels, pixel
    size and where it is."""
    md = image.metadata
    position = md.stage_position
    return {
        "kind": "fluorescence_image",
        "shape": list(image.data.shape),
        "dtype": str(image.data.dtype),
        "channels": [channel.name for channel in md.channels],
        "pixel_size": [md.pixel_size_x, md.pixel_size_y],
        "stage_position": position.to_dict() if position is not None else None,
    }


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


def bind_imaging(
    service: Type[_I], microscope: Any, fm: Optional[FM] = None
) -> Optional[_I]:
    """Build a microscope's imaging service of class *service* over its beams and
    *fm*, or None when it has neither (every column disabled and no FM), and
    ``auto_focus`` then raises."""
    beams = microscope.beams
    if not beams and fm is None:
        return None
    imaging = service(parent=microscope)
    roles: Dict[str, Any] = {
        _BEAM_ROLES[beam_type]: beam for beam_type, beam in beams.items()
    }
    if fm is not None:
        roles["fm"] = fm
    imaging.fill_roles(**roles)
    imaging.connect()
    return imaging
