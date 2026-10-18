"""One correlation, without Qt.

A :class:`CorrelationSession` holds what the correlation tab works on: the FIB
image and the FM stack, the points placed on them, the result of the last run
and the experiment's correlation config. It says whether a run can start and,
when it cannot, what to do next, and it runs one. The tab is a view of it; a
script, the agent or a service can use the same object without a window.

Saving and loading runs, the refractive-index correction and the transform a
run starts from are still the tab's. The pre-correction factor is held here
only because it is an input to the run.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional

from fibsem.correlation.config import MIN_FIDUCIAL_PAIRS, CorrelationConfig
from fibsem.correlation.fit import Seed, fit_transform
from fibsem.correlation.points import CorrelationPoints
from fibsem.correlation.prediction import excluded_indices
from fibsem.correlation.structures import (
    CorrelationInputData,
    CorrelationResult,
    PointStatus,
    PointType,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage

RUNNING = "Running…"


@dataclass(frozen=True)
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
        self.fib_image: Optional[FibsemImage] = None
        self.fm_image: Optional[FluorescenceImage] = None
        # The points and the selection, held once: the tab's lists and canvases
        # draw from them through a Qt point store (FIB-973).
        self.points = CorrelationPoints()
        # Pre-correlation RI factor (FM surface mode) and the optical
        # parameters it was armed with; set by the RI tab's Apply.
        self.ri_pre_correction_factor: Optional[float] = None
        self.ri_pre_correction_params: Optional[dict] = None
        # Experiment-global config; passed in on open, read back on close
        # (FIB-298). The fit and RI choices on screen are the tab's widgets.
        self.config = CorrelationConfig()
        self._result: Optional[CorrelationResult] = None
        # Bumped whenever the result is cleared or replaced; a run started
        # before the bump delivers into a state it no longer describes, so its
        # result is dropped (FIB-1242).
        self._generation = 0
        self._pending: Optional[PendingRun] = None

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

    def fit_rows(self) -> List[int]:
        """The row of each pair the fit will see, in fit order (see ``fit_data``)."""
        data = self.data
        excluded = excluded_indices(data.fib_coordinates, data.fm_coordinates)
        return [i for i in range(len(data.fib_coordinates)) if i not in excluded]

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
        """Adopt ``result`` (a delivered run, a loaded one or a corrected one)."""
        self._result = result

    def supersede_run(self) -> None:
        """Make a run in flight deliver nothing: what it was started on has
        been replaced (FIB-1242)."""
        self._generation += 1

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

    def start_run(self, seed: Optional[Seed] = None) -> PendingRun:
        """Start a run of the points as they are now, from ``seed`` (unseeded
        when None). Compute it with :meth:`PendingRun.compute`, on any thread,
        then hand the outcome to :meth:`finish_run` or :meth:`fail_run`.

        Does not check :meth:`can_run`: a run of points that cannot support
        one fails, and says why.
        """
        if self.running:
            raise RuntimeError("a correlation run is already in flight")
        self._pending = PendingRun(
            data=copy.deepcopy(self.fit_data),
            seed=seed,
            rows=self.fit_rows(),
            generation=self._generation,
        )
        return self._pending

    def finish_run(self, result: CorrelationResult) -> Optional[bool]:
        """Adopt a run's result, judged against the points as they are now.

        Returns whether the result is live (it still describes the current
        points: a point can be dragged while a run is in flight, FIB-321), or
        None, adopting nothing, for a run whose result was cleared or replaced
        while it ran (FIB-1242).
        """
        pending, self._pending = self._pending, None
        if pending is not None and pending.generation != self._generation:
            return None
        self._result = result
        return result.matches_inputs(self.fit_data)

    def fail_run(self) -> None:
        """The run in flight failed; another may start."""
        self._pending = None

    def run(self, seed: Optional[Seed] = None) -> CorrelationResult:
        """Run the correlation here and keep its result.

        Raises ValueError with the next step when the points cannot support a
        run, and what the fit raises when it fails.
        """
        if not self.can_run():
            raise ValueError(self.next_step())
        pending = self.start_run(seed)
        try:
            result = pending.compute()
        except Exception:
            self.fail_run()
            raise
        self.finish_run(result)
        return result


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
