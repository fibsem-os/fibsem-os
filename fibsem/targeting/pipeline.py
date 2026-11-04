"""Blob detection, grid fitting, filtering, and main orchestration pipeline."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import petname
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import closing

try:
    from skimage.morphology import footprint_rectangle
except ImportError:
    # footprint_rectangle was added in scikit-image 0.21; fall back to np.ones
    def footprint_rectangle(shape):
        return np.ones(shape, dtype=np.uint8)


from fibsem import constants, conversions
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.utils import create_pattern_mask
from fibsem.structures import FibsemImage, Point
from fibsem.targeting.config import PostProcessingParameters
from fibsem.targeting.scoring import (
    find_alignment_area,
    score_candidate_targets,
    select_best_per_grid_square,
    select_single_target,
)
from fibsem.targeting.structures import (
    REJECT_EDGE,
    REJECT_GRID_OVERLAP,
    REJECT_PATTERN_OVERLAP,
    REJECT_TOO_LARGE,
    REJECT_TOO_SMALL,
    CandidateTarget,
    LamellaTarget,
    SegmentationTargetResult,
)
from fibsem.targeting.util import (
    generate_grid_mask,
    generate_grid_mask_v2,
    generate_perspective_library,
    generate_rotation_masks,
    match_template,
)

if TYPE_CHECKING:
    from fibsem.segmentation.nnunet_model_v2 import SegmentationModelNNUnetV2
    from fibsem.targeting.structures import GridScreeningRun


def detect_blobbs(mask: np.ndarray, index: int = 1) -> np.ndarray:
    """Detect individual blobs from the segmentation mask."""

    # one-hot encode the mask for the given index (cells are index 1)
    mask = (mask == index).astype(np.uint8)

    # apply threshold
    bw = closing(mask > 0, footprint_rectangle((1, 1)))

    # remove artifacts connected to image border
    cleared = filters.gaussian(bw, sigma=1.5)
    blobbs = label(cleared)

    return blobbs  # type: ignore


def detect_positions(
    blobbs: np.ndarray, threshold_area: int = 500, max_threshold_area: int | None = None
) -> list[CandidateTarget]:
    """
    Description
    -----------
    function takes nnUNet prediction of cells and grid and label of connected cells and outputs
    positions of connected cells.

    Parameters
    ----------
    blobbs : ndarray of shape (512, 768, 1), dtype int
        label of connected cells.
    threshold_area : int
        minimum region area in pixels to include.
    max_threshold_area : int, optional
        maximum region area in pixels to include. No upper limit if None.
    Returns
    -------
    list[CandidateTarget] — all detected blobs; rejected ones have rejection_reason set.
    ``point`` is left as None and filled by the caller (run_detection_pipeline).
    """

    candidates: list[CandidateTarget] = []

    for region in regionprops(blobbs):
        y0, x0 = region.centroid
        candidate = CandidateTarget(pixel_coordinate=Point(x=float(x0), y=float(y0)))
        candidate.region_mask = (blobbs == region.label).astype(np.uint8)
        if region.area < threshold_area:
            candidate.rejection_reason = REJECT_TOO_SMALL
            candidate.score = 0.0
        elif max_threshold_area is not None and region.area > max_threshold_area:
            candidate.rejection_reason = REJECT_TOO_LARGE
            candidate.score = 0.0

        candidates.append(candidate)

    return candidates


def fit_the_grid(
    mask: np.ndarray,
    pixelsize: float,
    working_distance: float = 16130,
    hole_width: float = 90,
    pitch: float = 125,
    bar_width: float = 35,
    milling_angle: float = 11,
):
    """

    Description
    -----------
    function takes nnUNet prediction of cells and grid as input and fits a grid of the given
    parameters into the predicted grid.

    Parameters
    ----------
    mask : np.ndarray of shape (512, 768, 1), dtype int
        nnUNet prediction of cells and grid.
    pixelsize : float
        Pixel size in μm.
    working_distance : float
        distance from grid to detector in μm. The default is 16130.
    hole_width : float
        Hole width of grid in μm. The default is 90.
    pitch : float
        Pitch of grid in μm. The default is 125.
    bar_width : float
        Bar width of grid in μm. The default is 35.
    milling_angle : float
        angle between orientation of grid and FIB in °. The default is 79. (90 - milling angle of 11°)

    Returns
    -------
    np.ndarray of shape (512, 768, 3), dtype int depicting the fitted grid.

    """
    ROTATION_SEARCH_MAX_ANGLE = 90
    ROTATION_SEARCH_STEPSIZE = 1

    # 1. reconstruct the grid
    grid_mask = generate_grid_mask(pixelsize, hole_width, pitch, bar_width)

    # 2. fit the grid into the nnUNet prediction of cells and grid
    pred_mask = mask.copy() == 2
    pred_mask = pred_mask.astype(np.uint8)

    # 2.3 generate a rotation library from the reconstructed grid
    rotation_library = generate_rotation_masks(
        grid_mask,
        max_angle=ROTATION_SEARCH_MAX_ANGLE,
        stepsize=ROTATION_SEARCH_STEPSIZE,
    )

    # 2.4 transpose the rotation library into the perspective of the grid label from the nnUNet
    ret = generate_perspective_library(
        rotation_library, int(working_distance), int(milling_angle)
    )
    perspective_library, compression, distortion, translation = ret

    # 2.4.5 template matching of transposed image and nnUNet prediction of grid
    winner_image = match_template(
        pred_mask, perspective_library, compression, distortion, debug=False
    )

    return winner_image


def mind_the_grid(
    grid: np.ndarray,
    candidates: list[CandidateTarget],
    field_of_view: float = 104,
    lamella_width: float = 11,
    lamella_height: float = 8,
    edge_threshold: float = 50,
) -> list[CandidateTarget]:
    """

    Description
    -----------
    function takes fitted grid and cell blobb positions as input and kicks
    out positions that would lead to cutting into the grid.

    Parameters
    ----------
    grid : np.ndarray of shape (512, 768, 3), of dtype int
        grid fitted to nnUNet prediction using the function fit_the_grid().
    candidates : list[CandidateTarget]
        candidates from detect_positions(); already-rejected entries are skipped.
    field_of_view : float
        Horizontal Field Width. The default is 104.

    Returns
    -------
    list[CandidateTarget] — same list with rejection_reason set for grid/edge rejections.
    """

    height, width = grid.shape
    box_width = (width / field_of_view) * lamella_width
    box_height = (width / field_of_view) * lamella_height

    for candidate in candidates:
        if not candidate.is_accepted:
            continue  # already rejected upstream

        y0 = candidate.pixel_coordinate.y
        x0 = candidate.pixel_coordinate.x

        # cut out fib boxes around blobb centroid positions
        y_lo = int(y0 - box_height / 2)
        x_lo = int(x0 - box_width / 2)
        y_hi = int(y_lo + box_height)
        x_hi = int(x_lo + box_width)

        lamella_area = grid[y_lo:y_hi, x_lo:x_hi]

        # check if bbox is too close to edge of image
        if (
            y_lo - edge_threshold < 0
            or x_lo - edge_threshold < 0
            or y_hi + edge_threshold > height
            or x_hi + edge_threshold > width
        ):
            candidate.rejection_reason = REJECT_EDGE
            candidate.score = 0.0
            continue

        # check if fib box overlaps with grid
        if np.any(lamella_area):
            candidate.rejection_reason = REJECT_GRID_OVERLAP
            candidate.score = 0.0
            continue

    return candidates


def filter_positions_by_pattern_overlap(
    candidates: list[CandidateTarget],
    image: FibsemImage,
    winner_image: np.ndarray,
    edge_mask_padding: int = 50,
    reject_threshold: float = 0.2,
    milling_stages: Optional[list[FibsemMillingStage]] = None,
) -> tuple[np.ndarray | None, list[CandidateTarget]]:
    """Filter candidates by milling-pattern / grid-bar overlap.

    Only processes candidates where ``is_accepted`` is True; already-rejected
    candidates pass through untouched.  Accepted candidates receive a
    ``score = 1.0 - ratio_overlap``; rejected ones additionally have
    ``rejection_reason = REJECT_PATTERN_OVERLAP``.

    Returns
    -------
    pattern_mask : np.ndarray or None
        Combined milling-pattern mask (None when no milling stages are given).
    candidates : list[CandidateTarget]
        Same list, mutated in-place with score / rejection_reason updates.
    """
    winner_image_mask = winner_image.astype(bool)

    if milling_stages is None:
        milling_stages = []

    if image.metadata is None or image.metadata.pixel_size is None:
        raise ValueError(
            "Image metadata with pixel size is required for pattern overlap filtering."
        )

    image_shape = image.data.shape
    pixel_size = image.metadata.pixel_size.x

    # mask out the edges of the image
    pad = edge_mask_padding
    edge_mask = np.ones_like(winner_image_mask)
    edge_mask[pad:-pad, pad:-pad] = 0
    winner_image_mask = np.logical_or(winner_image_mask, edge_mask)

    debug_mask: np.ndarray | None = (
        None  # accumulated across all candidates for visualisation
    )

    for candidate in candidates:
        if not candidate.is_accepted:
            continue  # already rejected upstream

        pt = candidate.point  # microscope coords filled by run_detection_pipeline

        # build a fresh per-candidate mask for overlap scoring
        candidate_mask: np.ndarray | None = None
        for stage in milling_stages:
            stage.pattern.point = pt
            m1 = create_pattern_mask(stage.pattern, image_shape, pixel_size)
            candidate_mask = (
                m1 if candidate_mask is None else np.logical_or(candidate_mask, m1)
            )
            debug_mask = m1 if debug_mask is None else np.logical_or(debug_mask, m1)

        # no milling stages — nothing to check, accept with full score
        if candidate_mask is None:
            continue

        # compute the overlap between this candidate's pattern and the grid/edge mask
        overlap = np.logical_and(candidate_mask, winner_image_mask)
        n_overlap = np.sum(overlap)
        n_pattern_pixels = np.sum(candidate_mask)
        ratio_overlap = (
            float(n_overlap / n_pattern_pixels) if n_pattern_pixels > 0 else 0.0
        )
        candidate.score = 1.0 - ratio_overlap

        if ratio_overlap > reject_threshold:
            candidate.rejection_reason = REJECT_PATTERN_OVERLAP

    return debug_mask, candidates


def run_detection_pipeline(
    model: SegmentationModelNNUnetV2,
    image: FibsemImage,
    mask: Optional[np.ndarray] = None,
    post_processing: Optional[PostProcessingParameters] = None,
    show: bool = False,
) -> SegmentationTargetResult:
    """Run the full detection pipeline and return a ``SegmentationTargetResult``.

    Every detected blob is included in ``result.candidates`` regardless of
    rejection status.  Rejected candidates have ``enabled=False`` and a non-None
    ``rejection_reason``; accepted candidates have ``enabled=True``.
    """
    if post_processing is None:
        post_processing = PostProcessingParameters()

    if image.metadata is None or image.metadata.pixel_size is None:
        raise ValueError(
            "Image metadata with pixel size is required for detection pipeline."
        )

    grid = post_processing.grid
    pixelsize_m = image.metadata.pixel_size.x
    image_shape = image.data.shape
    field_of_view = image_shape[1] * pixelsize_m

    if mask is None:
        mask = model.inference(image.data, rgb=False)

    # 1. Detect blobs and assign size-based rejection reasons
    blobbs = detect_blobbs(copy.deepcopy(mask))
    min_area_px = int(
        np.pi * (post_processing.min_cell_diameter_um * 1e-6 / pixelsize_m / 2) ** 2
    )
    max_area_px = int(
        np.pi * (post_processing.max_cell_diameter_um * 1e-6 / pixelsize_m / 2) ** 2
    )
    candidates = detect_positions(
        blobbs, threshold_area=min_area_px, max_threshold_area=max_area_px
    )

    # 2. Fill microscope-space coordinates for all candidates
    for candidate in candidates:
        candidate.point = conversions.image_to_microscope_image_coordinates2(
            candidate.pixel_coordinate, image_shape, pixelsize_m
        )

    # 3. Fit grid and apply grid/edge filter
    grid_bar_mask = fit_the_grid(
        copy.deepcopy(mask),
        pixelsize=pixelsize_m * constants.METRE_TO_MICRON,
        hole_width=grid.hole_width * constants.METRE_TO_MICRON,
        pitch=grid.pitch * constants.METRE_TO_MICRON,
        bar_width=grid.bar_width * constants.METRE_TO_MICRON,
        milling_angle=grid.milling_angle,
    )
    candidates = mind_the_grid(
        grid=grid_bar_mask,
        candidates=candidates,
        field_of_view=field_of_view * constants.METRE_TO_MICRON,
        lamella_width=grid.lamella_width * constants.METRE_TO_MICRON,
        lamella_height=grid.lamella_height * constants.METRE_TO_MICRON,
    )

    # 4. Apply pattern-overlap filter
    pattern_mask, candidates = filter_positions_by_pattern_overlap(
        candidates=candidates,
        image=image,
        winner_image=grid_bar_mask,
        edge_mask_padding=post_processing.edge_mask_padding,
        reject_threshold=post_processing.reject_threshold,
        milling_stages=post_processing.milling_stages,
    )

    # 4.5. Score accepted candidates with composite quality function
    candidates = score_candidate_targets(
        candidates=candidates,
        seg_mask=mask,
        grid_bar_mask=grid_bar_mask,
        pixelsize_m=pixelsize_m,
        post_processing=post_processing,
    )

    # 4.6. Restrict candidates (restrict_to_single_target overrides max_targets_per_grid_square)
    weights = post_processing.scoring_weights
    if weights.restrict_to_single_target:
        candidates = select_single_target(candidates, grid_bar_mask, image.data.shape)
    elif weights.max_targets_per_grid_square > 0:
        candidates = select_best_per_grid_square(
            candidates, grid_bar_mask, max_n=weights.max_targets_per_grid_square
        )

    if show:
        from fibsem.targeting.plotting import _plot_detection_pipeline

        _plot_detection_pipeline(
            image,
            blobbs,
            mask,
            grid_bar_mask,
            [[c.pixel_coordinate.y, c.pixel_coordinate.x] for c in candidates],
            [
                [c.pixel_coordinate.y, c.pixel_coordinate.x]
                for c in candidates
                if c.is_accepted
            ],
            pattern_mask=pattern_mask,
        )

    # 5. Build LamellaTarget for every candidate
    stage_position = image.metadata.stage_position
    all_targets: list[LamellaTarget] = []
    for candidate in candidates:
        all_targets.append(
            LamellaTarget(
                name=petname.generate(2),  # type: ignore
                stage_position=copy.deepcopy(stage_position),
                poi=candidate.point,  # type: ignore
                pixel_coordinate=candidate.pixel_coordinate,
                score=candidate.score,
                rejection_reason=candidate.rejection_reason,
                enabled=candidate.is_accepted,
                sub_scores=candidate.sub_scores,
            )
        )

    # 6. Find alignment area for each accepted target (on grid bar, central third, non-overlapping)
    # box_width_px  = grid.lamella_width  / pixelsize_m
    # box_height_px = grid.lamella_height / pixelsize_m
    # for target in all_targets:
    #     if not target.is_accepted or target.pixel_coordinate is None:
    #         continue
    #     aa_px = find_alignment_area(
    #         grid_bar_mask=grid_bar_mask,
    #         target_pixel=target.pixel_coordinate,
    #         box_width_px=box_width_px,
    #         box_height_px=box_height_px,
    #     )
    #     if aa_px is not None:
    #         target.alignment_area = conversions.image_to_microscope_image_coordinates2(
    #             aa_px, image_shape, pixelsize_m
    #         )

    return SegmentationTargetResult(
        name=image.metadata.image_settings.filename,
        image=image,
        seg_mask=mask,
        grid_mask=grid_bar_mask,
        candidates=all_targets,
    )


def generate_screened_positions(
    model: SegmentationModelNNUnetV2,
    run: GridScreeningRun,
    progress_callback: Optional[Callable[[dict], None]] = None,
    result_callback: Optional[Callable[[SegmentationTargetResult], None]] = None,
) -> list[SegmentationTargetResult]:
    """Generate lamella positions for all images in a GridScreeningRun."""

    results: list[SegmentationTargetResult] = []

    for i, image in enumerate(run.images):
        if progress_callback:
            progress_callback(
                {
                    "current": i,
                    "total": len(run.images),
                    "image_name": image.metadata.image_settings.filename,
                }
            )
        try:
            result = run_detection_pipeline(
                model, image, post_processing=run.params.post_processing
            )
            result.name = f"result-{i:02d}"
        except Exception as e:
            logging.error(
                f"Error generating lamella positions for image {i}: {e}", exc_info=True
            )
            continue

        results.append(result)
        if result_callback:
            result_callback(result)

    if progress_callback and run.images:
        progress_callback(
            {"current": len(run.images), "total": len(run.images), "image_name": "Done"}
        )

    return results
