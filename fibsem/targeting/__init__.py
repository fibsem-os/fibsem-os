"""fibsem.targeting — automated lamella targeting pipeline."""

from fibsem.targeting.config import (
    GridPreparationParameters,
    GridScreeningParameters,
    ModelParameters,
    PostProcessingParameters,
    SampleGridParameters,
    TargetScoringWeights,
)
from fibsem.targeting.pipeline import (
    detect_blobbs,
    detect_positions,
    filter_positions_by_pattern_overlap,
    fit_the_grid,
    generate_screened_positions,
    mind_the_grid,
    run_detection_pipeline,
)
from fibsem.targeting.scoring import (
    _find_grid_hole_centers,
    _score_cell_size,
    _score_centrality,
    _score_class_coverage,
    _score_class_penalty,
    find_alignment_area,
    reapply_grid_square_filter,
    score_candidate_targets,
    select_best_per_grid_square,
)
from fibsem.targeting.structures import (
    REJECT_EDGE,
    REJECT_GRID_OVERLAP,
    REJECT_PATTERN_OVERLAP,
    REJECT_SUPERSEDED,
    REJECT_TOO_LARGE,
    REJECT_TOO_SMALL,
    REJECT_USER,
    TARGET_CLASS_MAP,
    TARGET_COLOR_MAP,
    TARGET_COLOR_RGB_MAP,
    CandidateTarget,
    GridScreeningRun,
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

__all__ = [
    # rejection constants
    "REJECT_EDGE",
    "REJECT_GRID_OVERLAP",
    "REJECT_PATTERN_OVERLAP",
    "REJECT_SUPERSEDED",
    "REJECT_TOO_LARGE",
    "REJECT_TOO_SMALL",
    "REJECT_USER",
    # color / class maps
    "TARGET_CLASS_MAP",
    "TARGET_COLOR_MAP",
    "TARGET_COLOR_RGB_MAP",
    # dataclasses
    "CandidateTarget",
    "GridScreeningRun",
    "LamellaTarget",
    "SegmentationTargetResult",
    # config
    "GridPreparationParameters",
    "GridScreeningParameters",
    "ModelParameters",
    "PostProcessingParameters",
    "SampleGridParameters",
    "TargetScoringWeights",
    # pipeline
    "detect_blobbs",
    "detect_positions",
    "filter_positions_by_pattern_overlap",
    "fit_the_grid",
    "generate_screened_positions",
    "mind_the_grid",
    "run_detection_pipeline",
    # scoring
    "find_alignment_area",
    "reapply_grid_square_filter",
    "score_candidate_targets",
    "select_best_per_grid_square",
    # util
    "generate_grid_mask",
    "generate_grid_mask_v2",
    "generate_perspective_library",
    "generate_rotation_masks",
    "match_template",
]
