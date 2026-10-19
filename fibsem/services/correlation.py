"""Correlation as a service: FM to FIB on saved images.

The correlation itself is `fibsem.correlation`'s, the same library the
correlation tab runs: a correlation session (the seed, the fit and its
verdict), the point fits and the run folders. The service takes saved image *paths* and points, since a route
can't carry an image, and saves each correlation as a run in the lamella's
Correlation folder, where the tab and the next correlation find it.

Being a service is what lets an agent correlate over the App Server, under
``/services/correlation`` (`fibsem.services.analysis`), with or without a
microscope. Every call reads saved files and moves nothing, so they are queries. Moving the lamella's point of interest to an
answer is the experiment's decision, not the service's.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fibsem.devices.core import query
from fibsem.services.core import Service

#: The overlay saved beside each run's correlation.json.
OVERLAY_FILENAME = "correlation_overlay.png"

#: The method each side's point fit uses when none is named.
DEFAULT_METHOD = {"fib": "Hole", "fm": "Gaussian"}


class Correlation(Service):
    """FM to FIB correlation: ``correlate`` fits the transform and maps the
    points of interest, ``refine_point`` moves one picked point onto the
    feature under it, ``predict`` maps more FM points through a saved run."""

    def __init__(self, name: str = "correlation", **kwargs: Any):
        super().__init__(name, **kwargs)

    @query
    def correlate(
        self,
        fib_image_path: str,
        fm_image_path: str,
        fib_points: List[List[float]],
        fm_points: List[List[float]],
        poi_points: List[List[float]],
        lamella_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fit the FM->FIB transform to fiducial pairs and map the points of
        interest into the FIB image.

        *fib_points* are ``(x, y)`` pixels in the FIB image at *fib_image_path*;
        *fm_points* the same fiducials as ``(x, y, z)`` pixels and slices in the
        FM stack at *fm_image_path*, in the same order (at least
        :data:`~fibsem.correlation.config.MIN_FIDUCIAL_PAIRS` pairs);
        *poi_points* the targets as ``(x, y, z)`` in the FM stack.

        With *lamella_path*, the fit starts from the experiment's previous runs
        (this lamella's first), as the tab's does, and is saved as a new run in
        the lamella's Correlation folder with an overlay to look at. Without it,
        the fit starts from the images' geometry and nothing is saved.

        Answers the RMS error, each point of interest in FIB image pixels and in
        metres from the image centre, the verdict (good, check, bad, with its
        reasons), what the fit started from, and the saved paths.
        """
        from fibsem.correlation.export import save_overlay
        from fibsem.correlation.history import CORRELATION_FILENAME, new_run_dir
        from fibsem.correlation.prior import experiment_runs
        from fibsem.correlation.session import CorrelationSession
        from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
        from fibsem.fm.structures import FluorescenceImage
        from fibsem.structures import FibsemImage

        def coordinates(points, kind, dims):
            out = []
            for point in points:
                if len(point) != dims:
                    raise ValueError(
                        f"{kind.value} points are {dims} numbers each, not {point}"
                    )
                x, y, z = (list(point) + [0.0])[:3]
                out.append(Coordinate(PointXYZ(x, y, z), kind))
            return out

        session = CorrelationSession()
        session.fib_image = FibsemImage.load(fib_image_path)
        session.fm_image = FluorescenceImage.load(fm_image_path)
        session.points.add_many(
            coordinates(fib_points, PointType.FIB, 2)
            + coordinates(fm_points, PointType.FM, 3)
            + coordinates(poi_points, PointType.POI, 3)
        )
        if not session.can_run():
            raise ValueError(
                _refusal(len(fib_points), len(fm_points), len(poi_points))
                or session.next_step()
            )
        if lamella_path:
            session.prior_runs = experiment_runs(
                os.path.dirname(os.path.abspath(lamella_path)), lamella_path
            )
        result = session.run()

        data = session.data
        answer = _summary(result, data, session.last_seed)
        if lamella_path:
            run_path = new_run_dir(lamella_path)
            session.save(os.path.join(run_path, CORRELATION_FILENAME))
            overlay_path = os.path.join(run_path, OVERLAY_FILENAME)
            save_overlay(overlay_path, session.fib_image, data, result)
            answer.update(run_path=run_path, overlay_path=overlay_path)
        return answer

    @query
    def refine_point(
        self,
        image_path: str,
        x: float,
        y: float,
        z: float = 0.0,
        beam: str = "fib",
        method: str = "",
        channel: int = 0,
    ) -> Dict[str, Any]:
        """Move a picked point onto the feature under it, as the tab's point fit
        does, and answer where it lands. Moves nothing: the answer is a proposal.

        *beam* ``fib``: a hole in the FIB image at *image_path* (*method*
        ``Hole``), at pixel ``(x, y)``. *beam* ``fm``: a reflection hole
        (``Hole``) or a fluorescent target (``Gaussian``, the default) in
        *channel* of the FM stack at *image_path*, at ``(x, y)`` pixels and
        slice *z*. Answers the status (ok, unchanged, error), the fitted
        ``(x, y, z)``, how far it moved in pixels, and why it failed.
        """
        from fibsem.correlation.point_fit import refine_point
        from fibsem.correlation.structures import Coordinate, PointType, PointXYZ

        if beam not in DEFAULT_METHOD:
            raise ValueError(f"beam is fib or fm, not {beam!r}")
        method = method or DEFAULT_METHOD[beam]
        kind = PointType.FIB if beam == "fib" else PointType.FM
        coord = Coordinate(PointXYZ(x, y, z), kind)
        if beam == "fib":
            from fibsem.structures import FibsemImage

            result = refine_point(
                coord, "fib", method, fib_image=FibsemImage.load(image_path)
            )
        else:
            from fibsem.fm.structures import FluorescenceImage

            result = refine_point(
                coord,
                "fm",
                method,
                fm_image=FluorescenceImage.load(image_path),
                channel=channel,
            )
        if result is None:
            raise ValueError(f"no {method!r} fit applies to a {beam} point")
        fitted = result.fitted
        return {
            "status": result.status.value,
            "method": result.method,
            "initial": [result.initial.x, result.initial.y, result.initial.z],
            "fitted": [fitted.x, fitted.y, fitted.z] if fitted else None,
            "moved_px": result.delta_px,
            "message": result.message,
        }

    @query
    def predict(self, run_path: str, fm_points: List[List[float]]) -> Dict[str, Any]:
        """Map FM points through the correlation saved in the run folder at
        *run_path*: ``(x, y, z)`` pixels and slices of the stack it was fitted on,
        to FIB image pixels and metres from the image centre. No
        refractive-index correction: points, not a target at depth."""
        from fibsem.correlation.fit import fm_to_fib
        from fibsem.correlation.history import load_run

        try:
            state = load_run(run_path)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        if state.result is None:
            raise ValueError(f"the run at {run_path} has no fitted result")
        return {
            "run_path": run_path,
            "points": [_point(p) for p in fm_to_fib(state.result, fm_points)],
        }


def _point(poi: Any) -> Dict[str, Any]:
    """A mapped point: FIB image pixels, and metres from the image centre."""
    return {
        "image_px": [poi.image_px.x, poi.image_px.y],
        "px_m": [poi.px_m.x, poi.px_m.y],
    }


def _refusal(n_fib: int, n_fm: int, n_poi: int) -> str:
    """Why the session's gate refused ``correlate``'s points, in the query's
    own terms (its parameters), or "" to use the gate's own sentence. The
    gate decides; this only words it, as the query always has."""
    from fibsem.correlation.config import MIN_FIDUCIAL_PAIRS

    if n_fib != n_fm:
        return (
            f"fib_points and fm_points are pairs: {n_fib} FIB points, {n_fm} FM points"
        )
    if n_fib < MIN_FIDUCIAL_PAIRS:
        return f"a correlation needs at least {MIN_FIDUCIAL_PAIRS} fiducial pairs"
    if not n_poi:
        return "poi_points: give at least one point of interest"
    return ""


def _summary(result: Any, data: Any, seed: Any) -> Dict[str, Any]:
    """What a correlation found, for the answer."""
    from fibsem.correlation.verdict import FitEvidence, verdict

    pixel_size = data.fib_image_pixel_size
    judged = None
    if result.diagnostics:
        v = verdict(FitEvidence.from_dict(result.diagnostics))
        judged = {
            "tier": v.tier,
            "headline": v.headline,
            "reasons": [r.text for r in v.reasons],
        }
    if seed.prior is not None:
        started = f"previous run ({seed.prior.source})"
    elif seed.transform is not None:
        started = "geometry"
    else:
        started = "unseeded" + (f": {seed.note}" if seed.note else "")
    return {
        "pairs": len(data.fib_coordinates),
        "rms_error_px": result.rms_error,
        "rms_error_nm": result.rms_error * pixel_size * 1e9 if pixel_size else None,
        "poi": [_point(p) for p in result.poi],
        "verdict": judged,
        "started_from": started,
        "placement_offset_um": result.placement_offset,
    }
