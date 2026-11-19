"""SEM/FIB coincidence from features picked in both images of a saved pair:
each feature's offsets, the ``vertical_move`` that brings them to
coincidence, and where they then sit. Several features give a median that
one mistaken match cannot drag, and name a match that disagrees; which to
trust is left to the caller. The geometry is the pair's own
(`fibsem.alignment.coincidence.geometry_from_images`), the sign that of
`fibsem.alignment.coincidence_target.TargetCoincidence`: this is that
measure with the match given rather than searched for. Nothing moves."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fibsem.alignment.coincidence import (
    DEFAULT_MAX_LATERAL_OFFSET,
    geometry_from_images,
)
from fibsem.conversions import image_to_microscope_image_coordinates_px
from fibsem.devices.stage import tilted_past_vertical
from fibsem.geometry.movement import vertical_move_delta
from fibsem.geometry.projection import scan_rotation_flips
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType, Point

# How far apart the two images' recorded stage positions may be before the
# answer says so.
SAME_POSITION_M = 1e-6
SAME_ANGLE_RAD = math.radians(0.05)
# The most point pairs one call takes.
MAX_PAIRS = 10
# A pair disagrees when its dy or dx is further than this from the others'
# median, or than this many of the coarser image's pixels, whichever is more.
DISAGREE_M = 2e-6
DISAGREE_PX = 4


def _offset(image: FibsemImage, px: Sequence[float], name: str) -> List[float]:
    """Point *px* ([x, y], origin top-left) as metres from the image centre,
    x right, y up, by the image's own pixel size."""
    height, width = image.data.shape[-2:]
    x, y = float(px[0]), float(px[1])
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError(
            f"the {name} point [{x:g}, {y:g}] is outside its {width} x {height} image"
        )
    size = image.metadata.pixel_size
    centred = image_to_microscope_image_coordinates_px(
        Point(x, y), (height, width), subpixel_precision=True
    )
    return [centred.x * size.x, centred.y * size.y]


def _checked(sem: FibsemImage, fib: FibsemImage) -> List[str]:
    """Refuse a pair that is not one electron and one ion image carrying
    pixel sizes and stage positions; answer, in words, how far the stage
    moved between the two when their recorded positions differ."""
    for image, name, beam in (
        (sem, "sem", BeamType.ELECTRON),
        (fib, "fib", BeamType.ION),
    ):
        metadata = image.metadata
        if metadata is None or metadata.image_settings is None:
            raise ValueError(f"the {name} image carries no beam image metadata")
        if metadata.image_settings.beam_type is not beam:
            got = getattr(metadata.image_settings.beam_type, "name", None)
            raise ValueError(
                f"the {name} image is an {got} image; give an {beam.name} image "
                f"as {name}_path"
            )
        size = metadata.pixel_size
        if size is None or not size.x or not size.y:
            raise ValueError(f"the {name} image carries no pixel size")
        state = metadata.microscope_state
        position = None if state is None else state.stage_position
        if position is None or None in (position.x, position.y, position.z, position.t):
            raise ValueError(f"the {name} image does not record its stage position")
    a = sem.metadata.microscope_state.stage_position
    b = fib.metadata.microscope_state.stage_position
    moved = [
        f"{axis} by {abs(getattr(a, axis) - getattr(b, axis)) * 1e6:.2f} um"
        for axis in ("x", "y", "z")
        if abs(getattr(a, axis) - getattr(b, axis)) > SAME_POSITION_M
    ]
    for axis in ("r", "t"):
        first, second = getattr(a, axis), getattr(b, axis)
        if first is None and second is None:
            continue
        if first is None or second is None:
            moved.append(f"{axis} recorded in only one")
            continue
        turn = (first - second + math.pi) % (2 * math.pi) - math.pi
        if abs(turn) > SAME_ANGLE_RAD:
            moved.append(f"{axis} by {math.degrees(abs(turn)):.2f} deg")
    warnings = []
    if moved:
        warnings.append(
            "the two images record different stage positions ("
            + ", ".join(moved)
            + "): the offsets assume one position, so they are off by about "
            "that much unless it is the stage's own jitter"
        )
    flips = [
        scan_rotation_flips(state.scan_rotation or 0.0)
        for state in (
            sem.metadata.microscope_state.electron_beam,
            fib.metadata.microscope_state.ion_beam,
        )
        if state is not None
    ]
    if len(set(flips)) > 1:
        raise ValueError(
            "one beam's scan rotation is turned half a turn from the other's: "
            "the views are not upright together, which this measure assumes"
        )
    return warnings


def _nm(value: float) -> float:
    """Metres to the nanometre."""
    return round(float(value), 9)


def _um(value: float) -> str:
    return f"{abs(value) * 1e6:.1f} um"


def _flag_disagreeing(measured: List[Dict[str, Any]], tolerance: float) -> None:
    """Mark each pair whose dy or dx is off the others by more than
    *tolerance* with ``disagrees``, in words. Two pairs that disagree are
    both marked: there is no telling which is right."""
    if len(measured) == 2:
        a, b = measured
        for axis in ("dy", "dx"):
            if abs(a[axis] - b[axis]) > tolerance:
                for m in measured:
                    m["disagrees"] = (
                        f"the two pairs' {axis} differ by {_um(a[axis] - b[axis])}: "
                        "check which is the same feature, or add a third pair"
                    )
                return
        return
    for axis in ("dy", "dx"):
        median = float(np.median([m[axis] for m in measured]))
        for m in measured:
            if "disagrees" not in m and abs(m[axis] - median) > tolerance:
                m["disagrees"] = (
                    f"its {axis} is {_um(m[axis] - median)} from the others' "
                    "median: it disagrees with the others; check it is the same "
                    "feature"
                )


def coincidence_offset(
    sem: FibsemImage,
    fib: FibsemImage,
    pairs: Sequence[Dict[str, Any]],
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Features picked in both images, each pair ``{sem: [x, y], fib: [x, y],
    label?}`` in full-resolution pixels: per pair its offsets and the dy and
    dx it alone gives, then the ``vertical_move`` (beam_type ION) from the
    median dy of the pairs that agree, the height error it means, and where
    each picked FIB point lands after it. A chamber-vertical move leaves the
    SEM view as it is; in the FIB a feature lands where its SEM offset,
    foreshortened, puts it. A pair off the others by more than the tolerance
    (*tolerance* metres, else 2 um or 4 pixels) is flagged, never dropped
    silently; with no majority agreeing no pair is chosen, and each pair's
    own dy stands. Raises ValueError for images or points this cannot
    measure."""
    if not 1 <= len(pairs) <= MAX_PAIRS:
        raise ValueError(f"give 1 to {MAX_PAIRS} point pairs, not {len(pairs)}")
    moved = _checked(sem, fib)
    try:
        geometry = geometry_from_images(sem, fib)
    except ValueError as exc:
        raise ValueError(f"the pair's metadata gives no coincidence geometry: {exc}")
    stretch = geometry.y_stretch
    sem_size, fib_size = sem.metadata.pixel_size, fib.metadata.pixel_size
    if tolerance is None:
        tolerance = max(DISAGREE_M, DISAGREE_PX * max(sem_size.x, fib_size.x))
    elif tolerance <= 0:
        raise ValueError("tolerance_um is more than 0")

    measured = []
    for i, pair in enumerate(pairs):
        label = str(pair.get("label") or i + 1)
        sem_offset = _offset(sem, pair["sem"], f"sem point of pair {label}")
        fib_offset = _offset(fib, pair["fib"], f"fib point of pair {label}")
        measured.append(
            {
                "label": label,
                "sem_px": [round(float(v), 1) for v in pair["sem"]],
                "fib_px": [round(float(v), 1) for v in pair["fib"]],
                "sem_offset_m": [_nm(v) for v in sem_offset],
                "fib_offset_m": [_nm(v) for v in fib_offset],
                # The FIB offset from where coincidence puts the feature (its
                # SEM offset, foreshortened): what vertical_move (ION) is
                # handed. dx: what no stage move changes.
                "dy": fib_offset[1] - sem_offset[1] / stretch,
                "dx": sem_offset[0] - fib_offset[0],
            }
        )
    _flag_disagreeing(measured, tolerance)
    agreeing = [m for m in measured if "disagrees" not in m]

    answer: Dict[str, Any] = {
        "convention": "offsets in metres from the image centre, x right, y up",
        "pixel_size": {"sem": sem_size.x, "fib": fib_size.x},
        "y_stretch": round(stretch, 4),
        "tolerance_m": _nm(tolerance),
        "vertical_move": None,
    }
    warnings: List[str] = list(moved)
    if len(measured) > 2 and len(agreeing) * 2 <= len(measured):
        agreeing = []
        warnings.append(
            "no majority of the pairs agree within tolerance_m: check each is "
            "the same feature in both images"
        )
    if not agreeing:
        answer["direction"] = (
            "The pairs disagree by more than tolerance_m, so none is chosen for "
            "you: each pair's dy is the move that pair alone gives. Recheck the "
            "features, add a pair, or use the one you trust."
        )
    else:
        dy = float(np.median([m["dy"] for m in agreeing]))
        dx = float(np.median([m["dx"] for m in agreeing]))
        used = [m["label"] for m in agreeing]
        position = sem.metadata.microscope_state.stage_position
        ion = fib.metadata.microscope_state.ion_beam
        move = vertical_move_delta(
            dx=0.0,
            dy=dy,
            scan_rotation=(ion.scan_rotation or 0.0) if ion is not None else 0.0,
            column_tilt=math.radians(fib.metadata.hardware_geometry.fib_column_tilt),
            stage_tilt=position.t,
            turned_over=tilted_past_vertical(position.t),
        )
        median = "" if len(used) == 1 else f" (median of pairs {', '.join(used)})"
        answer["direction"] = (
            f"In the FIB image the feature is {_um(dy)} "
            f"{'above' if dy > 0 else 'below'} where coincidence puts it{median}; "
            f"stage_vertical_move dy={dy:.4g}, beam_type ION, moves it there and "
            "leaves the SEM view as it is."
        )
        answer["vertical_move"] = {
            "dy": _nm(dy),
            "dx": 0.0,
            "beam_type": "ION",
            "from_pairs": used,
        }
        answer["dz"] = _nm(dy * geometry.dz_per_dy)
        answer["dx"] = _nm(dx)
        # The raw stage move stage_vertical_move makes for dy, by the live
        # move's own formula, with the pair's scan rotation and stage tilt.
        answer["stage_move_m"] = {a: _nm(getattr(move, a)) for a in ("x", "y", "z")}
        for m, pair in zip(measured, pairs):
            # The SEM view is unchanged; the FIB scene moves down the image by dy.
            x, y = (float(v) for v in pair["fib"])
            m["after_fib_px"] = [round(x, 1), round(y + dy / fib_size.y, 1)]
        if abs(dx) > DEFAULT_MAX_LATERAL_OFFSET:
            common = " in every agreeing pair" if len(used) > 1 else ""
            warnings.append(
                f"dx is {dx * 1e6:+.1f} um{common}: a height error cannot move x, "
                "so the beams are shifted (beam shift, scan rotation) or the "
                "match is wrong"
            )
    if len(measured) > 1:
        answer["spread_m"] = {
            axis: _nm(max(m[axis] for m in measured) - min(m[axis] for m in measured))
            for axis in ("dy", "dx")
        }
    for m in measured:
        m["dy"], m["dx"] = _nm(m["dy"]), _nm(m["dx"])
    answer["pairs"] = measured
    answer["warnings"] = warnings
    return answer
