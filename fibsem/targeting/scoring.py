"""Composite scoring and per-grid-hole candidate selection."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import numpy as np
from skimage.measure import label, regionprops

from fibsem.structures import Point
from fibsem.targeting.structures import (
    REJECT_SUPERSEDED,
    CandidateTarget,
    LamellaTarget,
)

if TYPE_CHECKING:
    from fibsem.targeting.config import PostProcessingParameters  # noqa: F401

# ---------------------------------------------------------------------------
# Private scoring helpers
# ---------------------------------------------------------------------------


def _find_grid_hole_centers(grid_bar_mask: np.ndarray) -> list[tuple[int, int]]:
    """Label the 0-regions of the fitted grid mask and return their centroids as (row, col)."""
    inverted = (grid_bar_mask == 0).astype(np.uint8)
    labelled = label(inverted)
    return [(int(r.centroid[0]), int(r.centroid[1])) for r in regionprops(labelled)]


def _score_cell_size(
    region_mask: np.ndarray, min_area_px: float, max_area_px: float
) -> float:
    """1.0 at the midpoint of [min_area_px, max_area_px], 0.0 at the boundaries."""
    area_px = float(np.sum(region_mask))
    ideal = (min_area_px + max_area_px) / 2.0
    half_range = (max_area_px - min_area_px) / 2.0
    if half_range <= 0:
        return 1.0
    return float(max(0.0, 1.0 - abs(area_px - ideal) / half_range))


def _score_class_penalty(
    region_mask: np.ndarray, seg_mask: np.ndarray, class_idx: int
) -> float:
    """1.0 - fraction of the candidate region occupied by *class_idx* (lower overlap = higher score)."""
    overlap = np.logical_and(region_mask.astype(bool), seg_mask == class_idx)
    area = float(np.sum(region_mask))
    if area <= 0:
        return 1.0
    return float(1.0 - np.sum(overlap) / area)


def _score_class_coverage(
    region_mask: np.ndarray, seg_mask: np.ndarray, class_idx: int
) -> float:
    """Fraction of the candidate region occupied by *class_idx* (higher coverage = higher score)."""
    overlap = np.logical_and(region_mask.astype(bool), seg_mask == class_idx)
    area = float(np.sum(region_mask))
    if area <= 0:
        return 0.0
    return float(np.sum(overlap) / area)


def _score_centrality(
    cy: float, cx: float, hole_centers: list[tuple[int, int]], hole_radius_px: float
) -> float:
    """1.0 at hole centre, falls linearly to 0.0 at the hole edge (distance = hole_radius_px)."""
    if not hole_centers:
        return 1.0
    dists = [math.hypot(cy - hc[0], cx - hc[1]) for hc in hole_centers]
    d = min(dists)
    return float(max(0.0, 1.0 - d / hole_radius_px)) if hole_radius_px > 0 else 1.0


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------


def score_candidate_targets(
    candidates: list[CandidateTarget],
    seg_mask: np.ndarray,
    grid_bar_mask: np.ndarray,
    pixelsize_m: float,
    post_processing: "PostProcessingParameters",  # noqa: F821
) -> list[CandidateTarget]:
    """Compute a composite quality score for each accepted candidate.

    Only accepted candidates with a ``region_mask`` are scored; rejected ones
    pass through unchanged.  The composite score is a weighted average of
    normalised sub-scores, replacing the prior binary ``score=1.0``.
    """
    weights = post_processing.scoring_weights
    hole_radius_px = (post_processing.grid.hole_width / pixelsize_m) / 2.0

    min_area_px = (
        math.pi * (post_processing.min_cell_diameter_um * 1e-6 / pixelsize_m / 2) ** 2
    )
    max_area_px = (
        math.pi * (post_processing.max_cell_diameter_um * 1e-6 / pixelsize_m / 2) ** 2
    )

    hole_centers = _find_grid_hole_centers(grid_bar_mask)

    for candidate in candidates:
        if not candidate.is_accepted or candidate.region_mask is None:
            continue

        cy = candidate.pixel_coordinate.y
        cx = candidate.pixel_coordinate.x

        s_cell_size = _score_cell_size(candidate.region_mask, min_area_px, max_area_px)
        s_centrality = _score_centrality(cy, cx, hole_centers, hole_radius_px)
        s_contamination_free = _score_class_penalty(candidate.region_mask, seg_mask, 4)
        s_crack_free = _score_class_penalty(candidate.region_mask, seg_mask, 5)
        s_contains_target = _score_class_coverage(candidate.region_mask, seg_mask, 1)

        candidate.sub_scores = {
            "cell_size": s_cell_size,
            "centrality": s_centrality,
            "contamination_free": s_contamination_free,
            "crack_free": s_crack_free,
            "contains_target": s_contains_target,
        }

        components = [
            (weights.cell_size, s_cell_size),
            (weights.centrality, s_centrality),
            (weights.contamination_free, s_contamination_free),
            (weights.crack_free, s_crack_free),
            (weights.contains_target, s_contains_target),
        ]
        weighted_sum = sum(w * s for w, s in components if w > 0)
        total_weight = sum(w for w, _ in components if w > 0)

        candidate.score = (
            float(weighted_sum / total_weight) if total_weight > 0 else 1.0
        )

    return candidates


def select_single_target(
    candidates: list[CandidateTarget],
    grid_bar_mask: np.ndarray,
    image_shape: tuple[int, int],
) -> list[CandidateTarget]:
    """Restrict to the single best candidate in the most central grid hole.

    Finds the grid hole whose centre is closest to the image centre, keeps only
    the highest-scoring accepted candidate in that hole, and marks all others
    REJECT_SUPERSEDED.
    """
    hole_centers = _find_grid_hole_centers(grid_bar_mask)
    if not hole_centers:
        return candidates

    img_cy, img_cx = image_shape[0] / 2.0, image_shape[1] / 2.0
    central_hole_idx = min(
        range(len(hole_centers)),
        key=lambda i: math.hypot(
            img_cy - hole_centers[i][0], img_cx - hole_centers[i][1]
        ),
    )

    in_central: list[CandidateTarget] = []
    for c in candidates:
        if not c.is_accepted:
            continue
        hole_idx = min(
            range(len(hole_centers)),
            key=lambda i: math.hypot(
                c.pixel_coordinate.y - hole_centers[i][0],
                c.pixel_coordinate.x - hole_centers[i][1],
            ),
        )
        if hole_idx == central_hole_idx:
            in_central.append(c)
        else:
            c.rejection_reason = REJECT_SUPERSEDED

    if len(in_central) > 1:
        ranked = sorted(in_central, key=lambda c: c.score, reverse=True)
        for c in ranked[1:]:
            c.rejection_reason = REJECT_SUPERSEDED

    return candidates


def select_best_per_grid_square(
    candidates: list[CandidateTarget],
    grid_bar_mask: np.ndarray,
    max_n: int = 1,
) -> list[CandidateTarget]:
    """Among accepted candidates, keep only the top *max_n* scorers per grid hole.

    Non-winning candidates receive ``rejection_reason=REJECT_SUPERSEDED``
    (score is preserved). Already-rejected candidates are untouched.
    ``max_n=0`` is a no-op (keep all).
    """
    if max_n <= 0:
        return candidates

    hole_centers = _find_grid_hole_centers(grid_bar_mask)
    if not hole_centers:
        return candidates

    buckets: dict[int, list[CandidateTarget]] = defaultdict(list)
    for c in candidates:
        if not c.is_accepted:
            continue
        hole_idx = min(
            range(len(hole_centers)),
            key=lambda i: math.hypot(
                c.pixel_coordinate.y - hole_centers[i][0],
                c.pixel_coordinate.x - hole_centers[i][1],
            ),
        )
        buckets[hole_idx].append(c)

    for group in buckets.values():
        if len(group) <= max_n:
            continue
        ranked = sorted(group, key=lambda c: c.score, reverse=True)
        for c in ranked[max_n:]:
            c.rejection_reason = REJECT_SUPERSEDED

    return candidates


def reapply_grid_square_filter(
    targets: list[LamellaTarget],
    grid_bar_mask: np.ndarray,
    max_n: int,
) -> list[LamellaTarget]:
    """Clear REJECT_SUPERSEDED from all targets and re-apply top-N-per-hole filtering.

    Used by the UI to change ``max_targets_per_grid_square`` interactively without
    re-running the full pipeline.  ``max_n=0`` clears all superseded labels (keep all).
    """
    # Clear existing superseded labels
    for t in targets:
        if t.rejection_reason == REJECT_SUPERSEDED:
            t.rejection_reason = None
            t.enabled = True

    if max_n <= 0:
        return targets

    hole_centers = _find_grid_hole_centers(grid_bar_mask)
    if not hole_centers:
        return targets

    buckets: dict[int, list[LamellaTarget]] = defaultdict(list)
    for t in targets:
        if not t.is_accepted or t.pixel_coordinate is None:
            continue
        hole_idx = min(
            range(len(hole_centers)),
            key=lambda i: math.hypot(
                t.pixel_coordinate.y - hole_centers[i][0],  # type: ignore[union-attr]
                t.pixel_coordinate.x - hole_centers[i][1],
            ),  # type: ignore[union-attr]
        )
        buckets[hole_idx].append(t)

    for group in buckets.values():
        if len(group) <= max_n:
            continue
        ranked = sorted(group, key=lambda t: t.score, reverse=True)
        for t in ranked[max_n:]:
            t.rejection_reason = REJECT_SUPERSEDED
            t.enabled = False

    return targets


def find_alignment_area(
    grid_bar_mask: np.ndarray,
    target_pixel: Point,
    box_width_px: float,
    box_height_px: float,
) -> Optional[Point]:
    """Find a suitable alignment area on the grid bar near the target.

    The returned pixel-space point satisfies all three requirements:
    - lies on the grid bar (grid_bar_mask == 1)
    - within the central third of the image (focus region)
    - does not overlap the target's lamella bounding box

    Returns the centroid of the closest valid bar region as ``Point(x=col, y=row)``,
    or ``None`` if no suitable region exists.
    """
    h, w = grid_bar_mask.shape

    # 1. Central-third mask (focus region: rows [h//3, 2*h//3))
    central_mask = np.zeros((h, w), dtype=bool)
    central_mask[h // 3 : 2 * h // 3, :] = True

    # 2. Exclusion zone centred on the target (lamella bounding box)
    cy, cx = int(target_pixel.y), int(target_pixel.x)
    y_lo = max(0, int(cy - box_height_px / 2))
    y_hi = min(h, int(cy + box_height_px / 2))
    x_lo = max(0, int(cx - box_width_px / 2))
    x_hi = min(w, int(cx + box_width_px / 2))
    exclusion = np.zeros((h, w), dtype=bool)
    exclusion[y_lo:y_hi, x_lo:x_hi] = True

    # 3. Valid alignment pixels: on bar, in focus region, outside exclusion zone
    valid = (grid_bar_mask == 1) & central_mask & ~exclusion

    if not np.any(valid):
        return None

    # 4. Label connected components; return centroid of the one closest to the target
    labelled = label(valid)
    best: Optional[Point] = None
    best_dist = float("inf")
    for region in regionprops(labelled):
        r, c = region.centroid
        dist = math.hypot(r - cy, c - cx)
        if dist < best_dist:
            best_dist = dist
            best = Point(x=float(c), y=float(r))

    return best
