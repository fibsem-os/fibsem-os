"""One correlation, without Qt.

A :class:`CorrelationSession` holds what the correlation tab works on: the FIB
image and the FM stack, the points placed on them, the result of the last run
and the experiment's correlation config. It says whether a run can start and,
when it cannot, what to do next, and it runs one, from the transform the
images' geometry and the previous runs give it. It arms and clears the
refractive-index pre-correction and applies the post-correction. It opens a
saved run and keeps its run folder current (the auto-save). The tab is a view
of it; a script, the agent or a service can use the same object without a
window: set the images, add the points, optionally arm a correction, run, save.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import enum
import logging
import os
from typing import Iterator, List, Optional, Tuple

from fibsem.correlation.config import MIN_FIDUCIAL_PAIRS, CorrelationConfig
from fibsem.correlation.fit import Seed, fit_transform, seed_for
from fibsem.correlation.history import CorrelationRun, save_run
from fibsem.correlation.points import SELECTION, STRUCTURE, CorrelationPoints
from fibsem.correlation.prediction import excluded_indices
from fibsem.correlation.prior import PlacementOffset, placement_offset_from_runs
from fibsem.correlation.refractive_index import ZetaParams, zeta_params_from_settings
from fibsem.correlation.structures import (
    CorrelationInputData,
    CorrelationResult,
    CorrelationState,
    PointStatus,
    PointType,
    load_correlation_file,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import CameraImageTransform, FibsemImage

RUNNING = "Running…"


class RunOutcome(enum.Enum):
    """What :meth:`CorrelationSession.finish_run` did with a run's result."""

    # Adopted, and it describes the points as they are now.
    LIVE = "live"
    # Adopted, but a point was moved while it ran (FIB-321).
    STALE = "stale"
    # Not adopted: the result was cleared or replaced while it ran (FIB-1242).
    SUPERSEDED = "superseded"


@dataclasses.dataclass(frozen=True)
class PendingRun:
    """A run that has started and not delivered: a copy of the points it fits,
    the transform it starts from, and the session's result generation then."""

    data: CorrelationInputData
    seed: Optional[Seed]
    # Each fitted pair's row in the user's lists (the fit skips rejected
    # pairs), so the verdict names the row the user sees.
    rows: List[int]
    generation: int

    def compute(self) -> CorrelationResult:
        """Fit the transform; on any thread, since it reads only its own copy.
        Raises what the fit raises."""
        return fit_transform(self.data, self.seed, rows=self.rows)


class CorrelationSession:
    """The images, the points, the result and the config of one correlation."""

    def __init__(self) -> None:
        # The points and the selection, held once: the tab's lists and canvases
        # draw from them through a Qt point store (FIB-973). Removing the FM
        # surface disarms the pre-correction; any change arms the auto-save.
        self.points = CorrelationPoints()
        self.points.subscribe(self._disarm_without_fm_surface)
        self.points.subscribe(self._arm_on_change)
        # Pre-correlation RI factor (FM surface mode) and the optical
        # parameters it was armed with (see arm_pre_correction).
        self.ri_pre_correction_factor: Optional[float] = None
        self.ri_pre_correction_params: Optional[dict] = None
        # The seed of the last run started, for what the status line says of it.
        self.last_seed: Optional[Seed] = None
        self._result: Optional[CorrelationResult] = None
        # Bumped whenever the result is cleared or replaced; a run started
        # before the bump delivers into a state it no longer describes, so its
        # result is dropped (FIB-1242).
        self._generation = 0
        self._pending: Optional[PendingRun] = None
        # Reads in progress: an open is not an edit, and writing during one
        # rewrote the file being read (FIB-1237).
        self._reads = 0
        self._start_fresh()

    def _start_fresh(self) -> None:
        """What a new correlation starts with; `reset` returns here."""
        self.fib_image: Optional[FibsemImage] = None
        self.fm_image: Optional[FluorescenceImage] = None
        # What a run starts from (see seed): previous runs, ``(label, run)``
        # pairs most specific first, and the camera transform the host assumes
        # for an FM stack that recorded none (FIB-881).
        self.prior_runs: List[Tuple[str, CorrelationRun]] = []
        self.assumed_fm_transform: Optional[CameraImageTransform] = None
        # Ignore the previous run's placement offset for this lamella (FIB-979).
        self.ignore_placement_offset = False
        # Positions as last seeded by the setup section, to detect manual edits
        # before replacing them (None = nothing seeded yet). FIB-302.
        self.seeded_positions: Optional[list] = None
        # Experiment-global config; passed in on open, read back on close
        # (FIB-298). The fit and RI choices on screen are the tab's widgets.
        self.config = CorrelationConfig()
        # The run folder the auto-save writes; a destination, not a promise:
        # it is created on the first write worth making (FIB-320).
        self.project_dir: Optional[str] = None
        # Auto-save is armed by a load or an edit, never by an image load: an
        # open must not write until there is something of the user's to keep
        # (FIB-977, see auto_save).
        self._save_armed = False
        # Last auto-save failure, held so the status line keeps reporting it
        # instead of claiming "Ready." while nothing is saved (FIB-316).
        self.save_error: Optional[str] = None

    def reset(self) -> None:
        """Start a new correlation here, as a new session would: no images,
        points, result, seed, pre-correction, previous runs or run folder, and
        the default config. A run in flight delivers nothing when it finishes.

        What subscribes to the points and the removal policy stay: a view open
        on this session shows the points go.
        """
        self.discard_result()
        self.points.replace_all([])  # disarms the pre-correction
        self._start_fresh()  # emptying the points is not an edit: unarmed again

    # ------------------------------------------------------------------
    # The images
    # ------------------------------------------------------------------

    @property
    def fib_pixel_size(self) -> Optional[float]:
        """FIB pixel size in metres, or None. A result loaded from JSON has no
        images restored, so this is legitimately absent rather than an error."""
        return getattr(
            getattr(getattr(self.fib_image, "metadata", None), "pixel_size", None),
            "x",
            None,
        )

    @property
    def fm_pixel_size_z(self) -> Optional[float]:
        """The FM stack's slice thickness in metres, or None when it does not say."""
        if self.fm_image is None:
            return None
        return getattr(self.fm_image.metadata, "pixel_size_z", None)

    # ------------------------------------------------------------------
    # The points
    # ------------------------------------------------------------------

    def set_data(self, data: CorrelationInputData) -> None:
        """Replace every point with ``data``'s, and its pre-correction factor.

        The images are not taken from ``data``: the session's stay.
        """
        surface = (
            [data.surface_coordinate] if data.surface_coordinate is not None else []
        )
        fm_surface = (
            [data.fm_surface_coordinate]
            if data.fm_surface_coordinate is not None
            else []
        )
        if surface and fm_surface:
            # Only one surface point is allowed — prefer the FM surface
            logging.warning(
                "Both FIB and FM surface coordinates present — keeping the FM surface."
            )
            surface = []
        self.points.replace_all(
            list(data.fib_coordinates)
            + surface
            + list(data.fm_coordinates)
            + list(data.poi_coordinates)
            + fm_surface
        )
        # `set_data(self.data)` hands back the same objects with new values (a
        # z-rescale); a view only rebuilds for different points, so say so.
        self.points.notify_changed(self.points.coordinates)
        # A factor without an FM surface is meaningless — don't arm it
        self.ri_pre_correction_factor = (
            data.ri_pre_correction_factor
            if data.fm_surface_coordinate is not None
            else None
        )
        self.ri_pre_correction_params = (
            data.ri_pre_correction_params
            if self.ri_pre_correction_factor is not None
            else None
        )

    def _disarm_without_fm_surface(self, change: str, _coords) -> None:
        """The pre-correction factor's lifecycle is tied to the FM surface
        point: once it is gone, however it went, the factor is disarmed so it
        cannot silently re-apply on a later run."""
        if change != STRUCTURE or self.points.of_type(PointType.SURFACE_FM):
            return
        self.clear_pre_correction()

    def in_loaded_stack(self, data: CorrelationInputData) -> None:
        """Bring a record's FM z into the loaded stack's planes, before it is shown.

        The one place a recorded z changes planes: a load, a legacy load and a
        seed all come through here, so a run saved from an interpolated stack
        or another acquisition opens at the depth it was picked at (FIB-1239).
        Without a loaded stack there is nothing to convert to, and the values
        are kept as recorded.
        """
        scale = data.move_fm_z_to_step(self.fm_pixel_size_z)
        if scale != 1.0:
            logging.info(
                "FM z recorded at %.1f nm per plane; moved into the loaded "
                "stack's planes (x%.4f)",
                self.fm_pixel_size_z * scale * 1e9,
                scale,
            )

    def adopt_inputs(self, data: CorrelationInputData) -> None:
        """Take a record's points as the session's, in the loaded stack's planes."""
        self.in_loaded_stack(data)
        self.set_data(data)

    @property
    def data(self) -> CorrelationInputData:
        """Everything a run reads, and everything a saved run records."""
        surface = self.points.of_type(PointType.SURFACE)
        fm_surface = self.points.of_type(PointType.SURFACE_FM)
        return CorrelationInputData(
            fib_image=self.fib_image,
            fm_image=self.fm_image,
            fib_coordinates=self.points.of_type(PointType.FIB),
            fm_coordinates=self.points.of_type(PointType.FM),
            poi_coordinates=self.points.of_type(PointType.POI),
            surface_coordinate=surface[0] if surface else None,
            fm_surface_coordinate=fm_surface[0] if fm_surface else None,
            ri_pre_correction_factor=self.ri_pre_correction_factor,
            ri_pre_correction_params=self.ri_pre_correction_params,
        )

    @property
    def fit_data(self) -> CorrelationInputData:
        """What the fit may see: ``data`` without tentative or rejected points.

        A predicted position is a guess and must never feed the fit; a rejected
        one stays on screen and in the file but not in the transform. Pairs are
        by index, so excluding a point excludes its partner too. ``data`` itself
        stays complete -- it is what the auto-save writes, and a prediction the
        user has not got to yet is part of the record.
        """
        return _for_fit(self.data)

    @property
    def fit_rows(self) -> List[int]:
        """The row of each pair the fit will see, in fit order (see ``fit_data``)."""
        data = self.data
        excluded = excluded_indices(data.fib_coordinates, data.fm_coordinates)
        return [i for i in range(len(data.fib_coordinates)) if i not in excluded]

    def positions(self) -> list:
        """Every point, as comparable ``(type, x, y, z)`` tuples.

        The surface points are in here too: they are single points held in their
        own fields rather than lists, and leaving them out meant a re-seed wiped
        a placed surface point — silently disarming any stored RI pre-correction
        with it — without so much as a confirmation (FIB-319).
        """
        d = self.data
        coords = d.fib_coordinates + d.fm_coordinates + d.poi_coordinates
        coords += [c for c in (d.surface_coordinate, d.fm_surface_coordinate) if c]
        return [(c.point_type.value, c.point.x, c.point.y, c.point.z) for c in coords]

    def mark_seeded(self) -> None:
        """Remember the points as they are now as what was last seeded."""
        self.seeded_positions = self.positions()

    def has_manual_edits(self) -> bool:
        """Whether the points differ from what was last seeded."""
        if self.seeded_positions is None:
            return bool(self.positions())
        return self.positions() != self.seeded_positions

    # ------------------------------------------------------------------
    # The refractive-index correction
    # ------------------------------------------------------------------

    def arm_pre_correction(
        self,
        factor: float,
        params: Optional[ZetaParams] = None,
        computed: bool = False,
    ) -> None:
        """Arm ``factor`` as the pre-correlation RI factor a run applies, with
        the optical parameters it is recorded with and whether it is ζ of them
        (``computed``) or was entered. Arms the auto-save.

        Without ``params`` the record is the experiment's RI settings
        (``config.ri``) at tilt 0, since the pre-correction acts along the
        optical axis, and the factor is recorded as entered: from a script,
        ``session.arm_pre_correction(1.3)``.
        """
        if params is None:
            params = dataclasses.replace(
                zeta_params_from_settings(self.config.ri), tilt_deg=0.0
            )
        self.arm_auto_save()
        self.ri_pre_correction_factor = factor
        self.ri_pre_correction_params = {
            "tilt_deg": params.tilt_deg,
            "depth_um": params.depth_um,
            "na": params.NA,
            "n2": params.n2,
            "wavelength_um": params.wavelength_um,
            "computed": computed,
        }

    def clear_pre_correction(self) -> None:
        """Disarm the pre-correction factor. Removing the FM surface does this
        too (see _disarm_without_fm_surface)."""
        self.ri_pre_correction_factor = None
        self.ri_pre_correction_params = None

    def apply_post_correction(self, factor: float) -> bool:
        """Correct the result's POI 1 for refractive index by ``factor`` about
        the FIB surface point, and adopt it. Returns whether the corrected
        result describes the current points.

        Raises ValueError when there is no result with its inputs or no FIB
        surface point, and what the correction raises when it refuses (a
        result already corrected, one never associated with an image); a
        refusal leaves the result as it was.
        """
        result = self._result
        if result is None or result.input_data is None:
            raise ValueError("there is no result with its inputs to correct")
        surface = self.points.of_type(PointType.SURFACE)
        if not surface:
            raise ValueError("there is no FIB surface point to correct about")
        # Record the surface this correction is about, as a *copy*: input_data is
        # the snapshot the transform was fitted to. Storing the live coordinate
        # would let a later drag of the surface line rewrite history — and, worse,
        # launder an already-stale result back to "current" (FIB-315). The
        # correction reads it from there, so it goes in first and comes out
        # again if the correction refuses: a second Apply on a corrected result
        # must not record a surface dragged since.
        recorded = result.input_data.surface_coordinate
        result.input_data.surface_coordinate = copy.deepcopy(surface[0])
        try:
            # Corrects POI 0 and propagates (reads input_data internally). It
            # refuses when it cannot correct px_m, which is the number Continue
            # commits, and when the result is corrected already.
            result.apply_refractive_index_correction(factor)
        except Exception:
            result.input_data.surface_coordinate = recorded
            raise
        self.set_result(result)
        # A FIB surface placed after the run is a transform input, so placing it
        # made the result stale; recording that surface on the result can make
        # it current again.
        return result.matches_inputs(self.fit_data)

    # ------------------------------------------------------------------
    # The seed
    # ------------------------------------------------------------------

    def placement_offset(self) -> Optional[PlacementOffset]:
        """A previous run's placement offset, or None (see FIB-979)."""
        return placement_offset_from_runs(self.prior_runs) if self.prior_runs else None

    def seed(self) -> Seed:
        """The transform a run of the loaded images starts from (see
        :func:`fibsem.correlation.fit.seed_for`), with the previous run its
        rotation and scale came from (``prior``) and any previous run passed
        over for disagreeing with the geometry (``prior_rejected``). Worked
        out on each call; a caller that needs several of these asks once."""
        return seed_for(
            self.fib_image,
            self.fm_image,
            self.prior_runs,
            assumed_fm_transform=self.assumed_fm_transform,
            use_placement_offset=not self.ignore_placement_offset,
        )

    # ------------------------------------------------------------------
    # The run gate
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether a run has started and not yet delivered or failed."""
        return self._pending is not None

    def can_run(self, data: Optional[CorrelationInputData] = None) -> bool:
        """Whether the inputs (``data``, else the session's) can support a run.

        A predicate rather than a read of the run button: the button is
        re-derived when the data changes, so a caller that announces a change
        and then asks the button gets the answer for the state it announced,
        not the one it meant to test.
        """
        d = _for_fit(data if data is not None else self.data)
        return (
            not self.running
            and self.fib_image is not None
            and self.fm_image is not None
            and len(d.fib_coordinates) >= MIN_FIDUCIAL_PAIRS
            and len(d.fm_coordinates) >= MIN_FIDUCIAL_PAIRS
            and len(d.poi_coordinates) >= 1
            and len(d.fib_coordinates) == len(d.fm_coordinates)
        )

    def next_step(self, data: Optional[CorrelationInputData] = None) -> str:
        """One sentence on what to do next, for the status line (FIB-978 §1.4).

        The bar's idle text used to be "Ready." or a count of what was missing;
        this names the step, in the order the flow goes: images, FIB fiducials,
        FM fiducials, the target, run.
        """
        if self.running:
            return RUNNING
        if self.fib_image is None or self.fm_image is None:
            return "Load the FIB and FM images on the Setup tab."
        d = _for_fit(data if data is not None else self.data)
        ok = self.can_run(d)
        # ``d`` is the fit's view, which drops a rejected pair and a predicted
        # FM point's FIB partner too; the sentence counts what the user sees:
        # points on either side, less the rows rejected from the fit, and on
        # the FM side only those placed (a prediction is not a pair yet).
        fib_all = self.points.of_type(PointType.FIB)
        fm_all = self.points.of_type(PointType.FM)
        rejected = {
            i
            for coords in (fib_all, fm_all)
            for i, c in enumerate(coords)
            if c.status == PointStatus.REJECTED
        }
        n_fib = sum(1 for i, _ in enumerate(fib_all) if i not in rejected)
        n_fm = sum(
            1
            for i, c in enumerate(fm_all)
            if i not in rejected and c.status not in PointStatus.TENTATIVE
        )
        n_poi = len(d.poi_coordinates)
        if ok:
            result = self._result
            stale = result is not None and not result.matches_inputs(d)
            other = result.image_mismatch(d) if stale else None
            lead = (
                f"The last result is for other images ({other}); run again"
                if other
                else "The points changed since the last run; run again"
                if stale
                else "Ready to run"
            )
            return f"{lead} with {n_fm} pairs." + _evidence_clause(d)
        if n_fib == 0:
            return (
                "Seed the spot burns on the Setup tab, or click the burns on the "
                "FIB image."
            )
        n_pred = sum(1 for c in fm_all if c.status in PointStatus.TENTATIVE)
        if n_fm < MIN_FIDUCIAL_PAIRS and n_pred:
            return f"Drag the predicted rings onto their burns: {n_fm} of {MIN_FIDUCIAL_PAIRS} pairs placed."
        if n_fm == 0:
            return "Project FM from FIB, or click the burns on the FM image."
        if n_fm < MIN_FIDUCIAL_PAIRS or n_fib < MIN_FIDUCIAL_PAIRS:
            return f"{min(n_fib, n_fm)} of {MIN_FIDUCIAL_PAIRS} pairs placed."
        if n_fib != n_fm:
            return f"FIB and FM fiducial counts differ ({n_fib} and {n_fm})."
        if n_poi == 0:
            return "Place the target on the FM image."
        return "Ready."  # unreachable: every gate above mirrors can_run

    # ------------------------------------------------------------------
    # The run and its result
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[CorrelationResult]:
        return self._result

    def set_result(self, result: CorrelationResult) -> None:
        """Adopt ``result`` (a delivered run, a loaded one or a corrected one).
        Arms the auto-save."""
        self._save_armed = True
        self._result = result

    def supersede_run(self) -> None:
        """Make a run in flight deliver nothing: what it was started on has
        been replaced (FIB-1242). Its seed goes too, so the status line of a
        result loaded or adopted next does not describe it."""
        self._generation += 1
        self.last_seed = None

    def discard_result(self) -> bool:
        """Drop the result, and supersede a run in flight. Returns whether
        there was a result to drop.

        Whenever the points are replaced wholesale by a new source, any result
        belongs to the previous points; kept, it is drawn over the new points,
        exported, and saved against coordinates it was never computed from
        (FIB-318).
        """
        self.supersede_run()
        if self._result is None:
            return False
        self._result = None
        return True

    def start_run(self) -> PendingRun:
        """Start a run of the points as they are now, from :meth:`seed`.
        Compute it with :meth:`PendingRun.compute`, on any thread, then hand
        the outcome to :meth:`finish_run` or :meth:`fail_run`.

        Does not check :meth:`can_run`: a run of points that cannot support
        one fails, and says why.
        """
        if self.running:
            raise RuntimeError("a correlation run is already in flight")
        self.last_seed = self.seed()
        self._pending = PendingRun(
            data=copy.deepcopy(self.fit_data),
            seed=self.last_seed,
            rows=self.fit_rows,
            generation=self._generation,
        )
        return self._pending

    def finish_run(self, result: CorrelationResult) -> RunOutcome:
        """Adopt a run's result, judged against the points as they are now.

        Returns LIVE when the result still describes the current points, STALE
        when a point was dragged while it ran (FIB-321), and SUPERSEDED,
        adopting nothing, for a run whose result was cleared or replaced while
        it ran (FIB-1242).
        """
        pending, self._pending = self._pending, None
        if pending is not None and pending.generation != self._generation:
            return RunOutcome.SUPERSEDED
        self.set_result(result)
        if result.matches_inputs(self.fit_data):
            return RunOutcome.LIVE
        return RunOutcome.STALE

    def fail_run(self) -> None:
        """The run in flight failed; another may start."""
        self._pending = None

    def abandon_run(self, pending: PendingRun) -> None:
        """``pending`` will never be delivered: whoever was computing it has
        gone (a closed window). If it is still the run in flight, it delivers
        nothing should it arrive, and another may start. A later run is left
        alone."""
        if self._pending is not pending:
            return
        self.supersede_run()
        self._pending = None

    def run(self) -> CorrelationResult:
        """Run the correlation here, from :meth:`seed`, and keep its result.

        Raises ValueError with the next step when the points cannot support a
        run, and what the fit raises when it fails.
        """
        if not self.can_run():
            raise ValueError(self.next_step())
        pending = self.start_run()
        try:
            result = pending.compute()
        except Exception:
            self.fail_run()
            raise
        self.finish_run(result)
        return result

    # ------------------------------------------------------------------
    # Opening a saved run
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def reading(self) -> Iterator[None]:
        """Read a file in without writing one: no auto-save during the read,
        and none after it until an edit or a run arms it. Nests.

        Opening a run used to rewrite it straight away -- ``set_data`` arms the
        save and the load's own ``data_changed`` fired it -- so the file took the
        names of whatever images happened to be open, and opening a folder with
        the wrong reference overwrote the record of the right one (FIB-1237).
        """
        self._reads += 1
        try:
            yield
        finally:
            self._reads -= 1
            if not self._reads:
                self._save_armed = False

    def adopt_result(
        self, result: CorrelationResult, *, adopt_inputs: bool = True
    ) -> bool:
        """Adopt a saved result, and its points too unless ``adopt_inputs`` is
        False. Returns whether it still describes the session's points.

        ``adopt_inputs=False`` when the caller has already loaded fresher
        coordinates: ``result.input_data`` is a record of what the transform was
        fitted to, not the current truth, so applying it would silently discard
        every edit made after that run (FIB-295).
        """
        self.supersede_run()  # a run in flight delivers nothing (FIB-1242)
        if adopt_inputs and result.input_data:
            # Deep-copy: the points become editable, and sharing the Coordinate
            # objects with the result's snapshot would make every edit mutate it
            # too, permanently defeating matches_inputs (FIB-315).
            self.adopt_inputs(copy.deepcopy(result.input_data))
        # Continue commits result.poi[0].px_m to the protocol editor, so a result
        # that predates the current points must not arm it.
        live = result.matches_inputs(self.fit_data)
        self.set_result(result)
        return live

    def adopt_state(self, state: CorrelationState) -> None:
        """Take a saved correlation: its points first, then its result, if any.

        Points are taken before the result so that, when a result is present,
        its staleness is judged against the freshly-loaded coordinates and its
        own snapshot is not replayed over them -- the FIB-295 ordering, now the
        only ordering because both come from one file. Without a result the
        session's is dropped: it belongs to the previous points (FIB-318).
        """
        self.adopt_inputs(state.input_data)
        if state.result is None:
            self.discard_result()
        else:
            self.adopt_result(state.result, adopt_inputs=False)

    def load(self, path: str) -> CorrelationState:
        """Open any correlation JSON -- consolidated file or either legacy file --
        without arming the auto-save. Returns what it read.

        Dispatches on the file's shape (:func:`load_correlation_file`), so the
        caller never picks the wrong loader -- the FIB-263 crash can't recur.
        Raises, changing nothing, when the file cannot be read.
        """
        state = load_correlation_file(path)  # before the read: a bad file
        with self.reading():  # changes nothing, the armed save included
            self.adopt_state(state)
        return state

    def load_data(self, path: str) -> CorrelationState:
        """Open a legacy coordinates file (``correlation_data.json``): its
        points, and no result. Otherwise as :meth:`load`."""
        state = CorrelationState(input_data=CorrelationInputData.load(path))
        with self.reading():
            self.adopt_state(state)
        return state

    def load_result(
        self, path: str, *, adopt_inputs: bool = True
    ) -> Tuple[CorrelationResult, bool]:
        """Open a legacy result file (``correlation_result.json``) and adopt it
        (:meth:`adopt_result`). Returns the result and whether it still
        describes the session's points. Otherwise as :meth:`load`."""
        result = CorrelationResult.load(path)
        with self.reading():
            live = self.adopt_result(result, adopt_inputs=adopt_inputs)
        return result, live

    def save(self, path: str) -> str:
        """Write the whole correlation (points and result) to ``path``, making
        its folder first, and return the path. An explicit save names where it
        goes: the dialogs offer the run folder, which is created on demand and
        may not exist yet (FIB-320)."""
        self.state.save(ensure_dir_for(path))
        return path

    # ------------------------------------------------------------------
    # The auto-save
    # ------------------------------------------------------------------

    def set_project_dir(self, path: str, *, keep_armed: bool = False) -> None:
        """Save into ``path`` from the next load or edit on (see ``auto_save``).

        ``keep_armed`` keeps a save already due, so it goes to the new folder:
        a folder typed in after an edit takes the edit with it.
        """
        self.project_dir = path
        if not keep_armed:
            self._save_armed = False

    def arm_auto_save(self) -> None:
        """Let the next ``auto_save`` write, though nothing has changed: a
        committed result is recorded in its run folder even when it was loaded
        and not edited (FIB-1237)."""
        self._save_armed = True

    def _arm_on_change(self, change: str, _coords) -> None:
        if change != SELECTION:
            self._save_armed = True

    @property
    def state(self) -> CorrelationState:
        """The whole correlation as a run records it: the points and the result."""
        return CorrelationState(input_data=self.data, result=self._result)

    def auto_save_due(self) -> bool:
        """Whether ``auto_save`` would write now.

        Nothing is written until there is a result to write, because a run folder
        is no longer inert bookkeeping -- the setup section seeds from it. Saving
        from the first data change made every *open* a run: history counted opens
        rather than correlations, and a session the user cancelled came back as
        the next open's starting coordinates (FIB-320). Once the folder exists it
        keeps its contents current, result or not.

        And nothing is written until a load or an edit has *armed* the save.
        Loading an image announces a change before any coordinates exist; on a
        folder that already holds a ``correlation.json`` the first such save used
        to overwrite it with an empty state, and the loader then read the file it
        had just emptied -- opening a saved run destroyed its picks (FIB-977).
        ``set_data`` (every load and seed goes through it), any change to the
        points and an adopted result arm it; ``set_project_dir`` and the end of
        a read disarm it.
        """
        if not self.project_dir or not self._save_armed or self._reads:
            return False
        return self._result is not None or os.path.isdir(self.project_dir)

    def auto_save(self) -> Optional[bool]:
        """Rewrite the run folder's ``correlation.json`` from the session, when
        due. Returns True when written, False when the write failed (the reason
        is in ``save_error`` until a write succeeds) and None when not due.

        The whole state, every time, so the points and the result can never drift
        apart on disk (which the split data/result files could; FIB-264,
        FIB-295). A post-run edit rewrites with the (now-stale) result still
        attached -- by design. The result carries its own ``input_data``
        snapshot, so staleness stays derivable on load (``matches_inputs``) and
        the result remains available to inspect; it is not silently discarded.
        """
        if not self.auto_save_due():
            return None
        try:
            save_run(self.project_dir, self.state)
        except Exception as exc:
            # Never let this pass unseen: it is the only persistence path, so a
            # swallowed failure silently discards the whole session's work
            # (FIB-316).
            logging.exception("Auto-save of correlation project failed")
            self.save_error = str(exc)
            return False
        self.save_error = None
        return True


def ensure_dir_for(path: str) -> str:
    """Create the directory ``path`` is about to be written into; return path.

    Every write site needs this now the run folder is created on demand: the
    auto-save mints it on the first result, and both save dialogs offer a
    default *inside* it -- so an explicitly-passed path can be just as absent
    as a derived one (FIB-320). Called immediately before writing, so an
    abandoned session still leaves nothing behind.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _for_fit(data: CorrelationInputData) -> CorrelationInputData:
    """``data`` reduced to what the fit may see (see ``fit_data``).

    Only tentative and rejected points -- and their index partners -- are
    left out. An unpaired point stays, as it always did: the run gate is
    what refuses mismatched lists, and a result's snapshot must still match
    the inputs it was fitted to.
    """
    excluded = excluded_indices(data.fib_coordinates, data.fm_coordinates)
    if not excluded:
        return data
    reduced = copy.copy(data)
    reduced.fib_coordinates = [
        c for i, c in enumerate(data.fib_coordinates) if i not in excluded
    ]
    reduced.fm_coordinates = [
        c for i, c in enumerate(data.fm_coordinates) if i not in excluded
    ]
    return reduced


def _evidence_clause(d: CorrelationInputData) -> str:
    """How many of the pairs are the user's evidence, when any are not.

    A prediction accepted where the projection put it carries the geometry's
    transform into the fit rather than testing it; the status line says so,
    so a result over mostly accepted points is not read as an independent
    confirmation of the seed.
    """
    accepted = sum(1 for c in d.fm_coordinates if c.status == PointStatus.ACCEPTED)
    if not accepted:
        return ""
    placed = len(d.fm_coordinates) - accepted
    return (
        f" {placed} pair{'s' if placed != 1 else ''} placed by you, "
        f"{accepted} accepted from the projection."
    )
