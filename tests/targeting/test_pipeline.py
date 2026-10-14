"""Unit tests for the fibsem.targeting pipeline, scoring and grid utilities.

These tests cover the pure-Python / numpy parts of the pipeline and do NOT
require a segmentation model or microscope hardware.
"""

import copy

import numpy as np
import pytest

from fibsem import utils
from fibsem.structures import (
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.targeting import (
    REJECT_EDGE,
    REJECT_GRID_OVERLAP,
    REJECT_PATTERN_OVERLAP,
    REJECT_SUPERSEDED,
    REJECT_TOO_LARGE,
    REJECT_TOO_SMALL,
    REJECT_USER,
    CandidateTarget,
    GridScreeningRun,
    LamellaTarget,
    PostProcessingParameters,
    SampleGridParameters,
    SegmentationTargetResult,
    TargetScoringWeights,
    _find_grid_hole_centers,
    _score_cell_size,
    _score_centrality,
    _score_class_coverage,
    _score_class_penalty,
    detect_positions,
    filter_positions_by_pattern_overlap,
    generate_grid_mask,
    generate_grid_mask_v2,
    mind_the_grid,
    score_candidate_targets,
    select_best_per_grid_square,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PIXEL_SIZE = 1e-7  # 100 nm/px — convenient for round numbers
_SHAPE = (512, 768)  # (rows, cols)


def _make_image(shape=_SHAPE, pixel_size=_PIXEL_SIZE) -> FibsemImage:
    """Return a minimal FibsemImage with the metadata the pipeline requires."""
    data = np.zeros(shape, dtype=np.uint8)
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(filename="test"),
        pixel_size=Point(pixel_size, pixel_size),
        microscope_state=MicroscopeState(),
    )
    return FibsemImage(data=data, metadata=metadata)


def _make_target(
    name="test", rejection_reason=None, enabled=True, score=1.0
) -> LamellaTarget:
    return LamellaTarget(
        name=name,
        stage_position=FibsemStagePosition(),
        poi=Point(0.0, 0.0),
        score=score,
        rejection_reason=rejection_reason,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# LamellaTarget
# ---------------------------------------------------------------------------


class TestLamellaTarget:
    def test_is_accepted_default(self):
        t = _make_target()
        assert t.is_accepted is True

    def test_is_accepted_rejection_reason(self):
        t = _make_target(rejection_reason=REJECT_TOO_SMALL)
        assert t.is_accepted is False

    def test_is_accepted_disabled(self):
        t = _make_target(enabled=False)
        assert t.is_accepted is False

    def test_is_accepted_user_rejected(self):
        t = _make_target(rejection_reason=REJECT_USER, enabled=False)
        assert t.is_accepted is False

    def test_roundtrip_accepted(self):
        original = LamellaTarget(
            name="foo-bar",
            stage_position=FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3),
            poi=Point(x=1.5e-6, y=-0.5e-6),
            score=0.85,
            rejection_reason=None,
            enabled=True,
            pixel_coordinate=Point(x=300.0, y=200.0),
            sub_scores={"cell_size": 0.9, "centrality": 0.7},
        )
        restored = LamellaTarget.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.score == pytest.approx(original.score)
        assert restored.rejection_reason is None
        assert restored.enabled is True
        assert restored.pixel_coordinate.x == pytest.approx(original.pixel_coordinate.x)
        assert restored.pixel_coordinate.y == pytest.approx(original.pixel_coordinate.y)
        assert restored.sub_scores == original.sub_scores

    def test_roundtrip_sub_scores_missing_defaults_to_empty(self):
        """Old serialized dicts without sub_scores should deserialize to empty dict."""
        d = {
            "name": "old",
            "stage_position": FibsemStagePosition().to_dict(),
            "poi": Point(0.0, 0.0).to_dict(),
        }
        restored = LamellaTarget.from_dict(d)
        assert restored.sub_scores == {}

    def test_roundtrip_rejected(self):
        original = LamellaTarget(
            name="big-cell",
            stage_position=FibsemStagePosition(),
            poi=Point(0.0, 0.0),
            score=0.0,
            rejection_reason=REJECT_TOO_LARGE,
            enabled=False,
        )
        restored = LamellaTarget.from_dict(original.to_dict())
        assert restored.rejection_reason == REJECT_TOO_LARGE
        assert restored.enabled is False
        assert restored.score == pytest.approx(0.0)

    def test_roundtrip_no_pixel_coordinate(self):
        original = _make_target()
        assert original.pixel_coordinate is None
        restored = LamellaTarget.from_dict(original.to_dict())
        assert restored.pixel_coordinate is None

    def test_from_dict_backward_compat_defaults(self):
        """Old serialized dicts may lack score/rejection_reason/enabled/pixel_coordinate."""
        d = {
            "name": "legacy",
            "stage_position": FibsemStagePosition().to_dict(),
            "poi": Point(0.0, 0.0).to_dict(),
        }
        t = LamellaTarget.from_dict(d)
        assert t.score == pytest.approx(1.0)
        assert t.rejection_reason is None
        assert t.enabled is True
        assert t.pixel_coordinate is None


# ---------------------------------------------------------------------------
# CandidateTarget
# ---------------------------------------------------------------------------


class TestCandidateTarget:
    def test_is_accepted_no_reason(self):
        c = CandidateTarget(pixel_coordinate=Point(10, 20))
        assert c.is_accepted is True

    def test_is_accepted_with_reason(self):
        c = CandidateTarget(
            pixel_coordinate=Point(10, 20), rejection_reason=REJECT_EDGE
        )
        assert c.is_accepted is False


# ---------------------------------------------------------------------------
# SegmentationTargetResult properties
# ---------------------------------------------------------------------------


class TestSegmentationTargetResult:
    def _make_result(self) -> SegmentationTargetResult:
        image = _make_image()
        return SegmentationTargetResult(
            name="frame_001",
            image=image,
            seg_mask=np.zeros(_SHAPE, dtype=int),
            grid_mask=np.zeros(_SHAPE, dtype=int),
            candidates=[
                _make_target("a", rejection_reason=None, enabled=True),
                _make_target(
                    "b", rejection_reason=REJECT_TOO_SMALL, enabled=False, score=0.0
                ),
                _make_target(
                    "c", rejection_reason=REJECT_GRID_OVERLAP, enabled=False, score=0.0
                ),
                _make_target("d", rejection_reason=REJECT_USER, enabled=False),
            ],
        )

    def test_targets_returns_enabled_only(self):
        result = self._make_result()
        names = [t.name for t in result.targets]
        assert names == ["a"]

    def test_rejected_targets_excludes_accepted(self):
        result = self._make_result()
        names = [t.name for t in result.rejected_targets]
        assert "a" not in names
        assert set(names) == {"b", "c", "d"}

    def test_empty_candidates(self):
        result = SegmentationTargetResult(
            name="empty",
            image=_make_image(),
            seg_mask=np.zeros(_SHAPE, dtype=int),
            grid_mask=np.zeros(_SHAPE, dtype=int),
        )
        assert result.targets == []
        assert result.rejected_targets == []


# ---------------------------------------------------------------------------
# GridScreeningRun
# ---------------------------------------------------------------------------


class TestGridScreeningRun:
    def test_created_at_is_written_as_iso_with_offset(self, tmp_path):
        run = GridScreeningRun(name="run")
        run.save(str(tmp_path))

        written = utils.load_yaml(str(tmp_path / "run.yaml"))["created_at"]
        assert written == run.created_at.isoformat()
        assert run.created_at.utcoffset() is not None

        loaded = GridScreeningRun.load(str(tmp_path))
        assert loaded.created_at == run.created_at

    def test_posix_created_at_still_reads(self):
        run = GridScreeningRun.from_dict({"created_at": 1_757_800_000.0}, [])
        assert run.created_at.timestamp() == 1_757_800_000.0

    def test_missing_created_at_reads_as_none(self):
        assert GridScreeningRun.from_dict({}, []).created_at is None


# ---------------------------------------------------------------------------
# detect_positions
# ---------------------------------------------------------------------------


class TestDetectPositions:
    def _blobbs_with_one_region(self, area_px: int, shape=_SHAPE) -> np.ndarray:
        """Return a labelled array with one square blob of approximately area_px pixels."""
        side = int(area_px**0.5)
        blobbs = np.zeros(shape, dtype=int)
        cy, cx = shape[0] // 2, shape[1] // 2
        blobbs[cy : cy + side, cx : cx + side] = 1
        return blobbs

    def test_accepted_blob(self):
        blobbs = self._blobbs_with_one_region(area_px=1000)
        candidates = detect_positions(
            blobbs, threshold_area=500, max_threshold_area=5000
        )
        assert len(candidates) == 1
        assert candidates[0].is_accepted

    def test_too_small_blob(self):
        blobbs = self._blobbs_with_one_region(area_px=100)
        candidates = detect_positions(
            blobbs, threshold_area=500, max_threshold_area=5000
        )
        assert len(candidates) == 1
        assert candidates[0].rejection_reason == REJECT_TOO_SMALL

    def test_too_large_blob(self):
        blobbs = self._blobbs_with_one_region(area_px=9000)
        candidates = detect_positions(
            blobbs, threshold_area=500, max_threshold_area=5000
        )
        assert len(candidates) == 1
        assert candidates[0].rejection_reason == REJECT_TOO_LARGE

    def test_no_max_threshold(self):
        """When max_threshold_area=None large blobs are always accepted."""
        blobbs = self._blobbs_with_one_region(area_px=9000)
        candidates = detect_positions(
            blobbs, threshold_area=500, max_threshold_area=None
        )
        assert len(candidates) == 1
        assert candidates[0].is_accepted

    def test_pixel_coordinate_set(self):
        blobbs = self._blobbs_with_one_region(area_px=1000)
        candidates = detect_positions(blobbs, threshold_area=500)
        assert candidates[0].pixel_coordinate is not None

    def test_empty_mask_returns_empty_list(self):
        blobbs = np.zeros(_SHAPE, dtype=int)
        candidates = detect_positions(blobbs, threshold_area=500)
        assert candidates == []

    def test_point_not_filled(self):
        """detect_positions must NOT fill candidate.point — that's run_detection_pipeline's job."""
        blobbs = self._blobbs_with_one_region(area_px=1000)
        candidates = detect_positions(blobbs, threshold_area=500)
        assert candidates[0].point is None


# ---------------------------------------------------------------------------
# mind_the_grid
# ---------------------------------------------------------------------------


class TestMindTheGrid:
    _FOV = 104.0  # µm HFW
    _LW = 11.0  # µm lamella width
    _LH = 8.0  # µm lamella height
    _EDGE = 50  # px edge threshold

    def _candidate_at(self, y, x) -> CandidateTarget:
        return CandidateTarget(
            pixel_coordinate=Point(x=float(x), y=float(y)),
            point=Point(0.0, 0.0),
        )

    def test_accepted_in_clear_area(self):
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        # Place candidate safely in the middle, away from edges
        c = self._candidate_at(y=256, x=400)
        result = mind_the_grid(
            grid,
            [c],
            field_of_view=self._FOV,
            lamella_width=self._LW,
            lamella_height=self._LH,
            edge_threshold=self._EDGE,
        )
        assert result[0].is_accepted

    def test_rejected_too_close_to_edge(self):
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        # Place very close to the top-left corner
        c = self._candidate_at(y=5, x=5)
        result = mind_the_grid(
            grid,
            [c],
            field_of_view=self._FOV,
            lamella_width=self._LW,
            lamella_height=self._LH,
            edge_threshold=self._EDGE,
        )
        assert result[0].rejection_reason == REJECT_EDGE

    def test_rejected_grid_overlap(self):
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        cy, cx = 256, 400
        # Fill the lamella box area with grid bars
        box_w = int((_SHAPE[1] / self._FOV) * self._LW)
        box_h = int((_SHAPE[1] / self._FOV) * self._LH)
        grid[cy - box_h // 2 : cy + box_h // 2, cx - box_w // 2 : cx + box_w // 2] = 1
        c = self._candidate_at(y=cy, x=cx)
        result = mind_the_grid(
            grid,
            [c],
            field_of_view=self._FOV,
            lamella_width=self._LW,
            lamella_height=self._LH,
            edge_threshold=self._EDGE,
        )
        assert result[0].rejection_reason == REJECT_GRID_OVERLAP

    def test_already_rejected_candidates_are_skipped(self):
        """Candidates rejected before mind_the_grid must not be re-tagged."""
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        c = CandidateTarget(
            pixel_coordinate=Point(x=5.0, y=5.0),  # would trigger REJECT_EDGE
            point=Point(0.0, 0.0),
            rejection_reason=REJECT_TOO_SMALL,
        )
        result = mind_the_grid(grid, [c])
        # rejection_reason must stay as REJECT_TOO_SMALL, not be overwritten
        assert result[0].rejection_reason == REJECT_TOO_SMALL

    def test_returns_same_list(self):
        """mind_the_grid must return the same list object (in-place mutation)."""
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        candidates = [self._candidate_at(256, 400)]
        returned = mind_the_grid(grid, candidates)
        assert returned is candidates


# ---------------------------------------------------------------------------
# filter_positions_by_pattern_overlap (no milling stages)
# ---------------------------------------------------------------------------


class TestFilterPositionsByPatternOverlap:
    def test_no_milling_stages_all_accepted(self):
        image = _make_image()
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        candidates = [
            CandidateTarget(
                pixel_coordinate=Point(x=400.0, y=256.0),
                point=Point(0.0, 0.0),
            )
        ]
        debug_mask, result = filter_positions_by_pattern_overlap(
            candidates, image, grid, milling_stages=[]
        )
        assert debug_mask is None
        assert result[0].is_accepted
        assert result[0].score == pytest.approx(1.0)

    def test_already_rejected_candidates_untouched(self):
        image = _make_image()
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        c = CandidateTarget(
            pixel_coordinate=Point(x=400.0, y=256.0),
            point=Point(0.0, 0.0),
            rejection_reason=REJECT_GRID_OVERLAP,
        )
        _, result = filter_positions_by_pattern_overlap(
            [c], image, grid, milling_stages=[]
        )
        assert result[0].rejection_reason == REJECT_GRID_OVERLAP

    def test_raises_without_metadata(self):
        image = FibsemImage(data=np.zeros(_SHAPE, dtype=np.uint8))
        grid = np.zeros(_SHAPE, dtype=np.uint8)
        with pytest.raises(ValueError, match="metadata"):
            filter_positions_by_pattern_overlap([], image, grid)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


class TestScoreCellSize:
    def test_at_midpoint(self):
        # A mask of area 50: ideal between min=0 and max=100
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :10] = 1  # area = 50
        assert _score_cell_size(mask, 0.0, 100.0) == pytest.approx(1.0)

    def test_at_boundary_returns_zero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, :] = 1  # area = 10 = min_area
        assert _score_cell_size(mask, 10.0, 100.0) == pytest.approx(0.0)

    def test_clamped_below_zero(self):
        mask = np.zeros((5, 5), dtype=np.uint8)  # area = 0; way below min
        assert _score_cell_size(mask, 50.0, 100.0) == pytest.approx(0.0)

    def test_zero_range_returns_one(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        assert _score_cell_size(mask, 5.0, 5.0) == pytest.approx(1.0)


class TestScoreClassPenalty:
    def test_clean_region_returns_one(self):
        region = np.ones((4, 4), dtype=np.uint8)
        seg = np.zeros((4, 4), dtype=np.uint8)  # no contamination
        assert _score_class_penalty(region, seg, 4) == pytest.approx(1.0)

    def test_fully_contaminated_returns_zero(self):
        region = np.ones((4, 4), dtype=np.uint8)
        seg = np.full((4, 4), 4, dtype=np.uint8)
        assert _score_class_penalty(region, seg, 4) == pytest.approx(0.0)

    def test_half_contaminated(self):
        region = np.ones((2, 4), dtype=np.uint8)
        seg = np.zeros((2, 4), dtype=np.uint8)
        seg[:, :2] = 4  # half contaminated
        assert _score_class_penalty(region, seg, 4) == pytest.approx(0.5)

    def test_empty_region_returns_one(self):
        region = np.zeros((4, 4), dtype=np.uint8)
        seg = np.full((4, 4), 4, dtype=np.uint8)
        assert _score_class_penalty(region, seg, 4) == pytest.approx(1.0)


class TestScoreClassCoverage:
    def test_no_cell_returns_zero(self):
        region = np.ones((4, 4), dtype=np.uint8)
        seg = np.zeros((4, 4), dtype=np.uint8)
        assert _score_class_coverage(region, seg, 1) == pytest.approx(0.0)

    def test_full_cell_returns_one(self):
        region = np.ones((4, 4), dtype=np.uint8)
        seg = np.ones((4, 4), dtype=np.uint8)
        assert _score_class_coverage(region, seg, 1) == pytest.approx(1.0)

    def test_half_cell(self):
        region = np.ones((2, 4), dtype=np.uint8)
        seg = np.zeros((2, 4), dtype=np.uint8)
        seg[:, :2] = 1
        assert _score_class_coverage(region, seg, 1) == pytest.approx(0.5)

    def test_empty_region_returns_zero(self):
        region = np.zeros((4, 4), dtype=np.uint8)
        seg = np.ones((4, 4), dtype=np.uint8)
        assert _score_class_coverage(region, seg, 1) == pytest.approx(0.0)


class TestFindGridHoleCenters:
    def test_single_hole(self):
        # 10x10 mask: all bars (1) except a 4x4 hole in the top-left
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[1:5, 1:5] = 0  # hole region rows 1-4, cols 1-4
        centers = _find_grid_hole_centers(mask)
        assert len(centers) == 1
        r, c = centers[0]
        assert 1 <= r <= 4
        assert 1 <= c <= 4

    def test_no_holes_all_bar(self):
        mask = np.ones((8, 8), dtype=np.uint8)
        centers = _find_grid_hole_centers(mask)
        assert centers == []

    def test_two_holes(self):
        mask = np.ones((10, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 0  # hole 1
        mask[2:5, 15:18] = 0  # hole 2
        centers = _find_grid_hole_centers(mask)
        assert len(centers) == 2


class TestScoreCentrality:
    def test_at_center_returns_one(self):
        centers = [(10, 10)]
        score = _score_centrality(10.0, 10.0, centers, hole_radius_px=50.0)
        assert score == pytest.approx(1.0)

    def test_at_edge_returns_zero(self):
        centers = [(10, 10)]
        score = _score_centrality(10.0, 10.0 + 50.0, centers, hole_radius_px=50.0)
        assert score == pytest.approx(0.0)

    def test_beyond_edge_clamped_to_zero(self):
        centers = [(10, 10)]
        score = _score_centrality(10.0, 10.0 + 100.0, centers, hole_radius_px=50.0)
        assert score == pytest.approx(0.0)

    def test_no_centers_returns_one(self):
        assert _score_centrality(5.0, 5.0, [], hole_radius_px=20.0) == pytest.approx(
            1.0
        )

    def test_zero_radius_returns_one(self):
        assert _score_centrality(
            5.0, 5.0, [(5, 5)], hole_radius_px=0.0
        ) == pytest.approx(1.0)


class TestScoreCandidateTargets:
    def _make_candidate(self, x, y, mask_shape=(10, 10), rejected=False):
        region_mask = np.ones(mask_shape, dtype=np.uint8)
        c = CandidateTarget(
            pixel_coordinate=Point(x=float(x), y=float(y)),
            region_mask=region_mask,
        )
        if rejected:
            c.rejection_reason = "too_small"
            c.score = 0.0
        return c

    def test_accepted_candidates_get_composite_score(self):
        accepted = self._make_candidate(5, 5)
        seg_mask = np.zeros((10, 10), dtype=np.uint8)
        seg_mask[3:7, 3:7] = 1  # cell class
        grid_mask = np.ones((10, 10), dtype=np.uint8)
        pp = PostProcessingParameters(
            min_cell_diameter_um=0.001,
            max_cell_diameter_um=100.0,
            grid=SampleGridParameters(hole_width=100e-6),
        )
        result = score_candidate_targets([accepted], seg_mask, grid_mask, 1e-7, pp)
        # Score should be in [0, 1] and not the default 1.0
        assert 0.0 <= result[0].score <= 1.0

    def test_rejected_candidates_unchanged(self):
        rejected = self._make_candidate(5, 5, rejected=True)
        seg_mask = np.zeros((10, 10), dtype=np.uint8)
        grid_mask = np.ones((10, 10), dtype=np.uint8)
        pp = PostProcessingParameters()
        score_before = rejected.score
        score_candidate_targets([rejected], seg_mask, grid_mask, 1e-7, pp)
        assert rejected.score == score_before

    def test_no_region_mask_skipped(self):
        c = CandidateTarget(pixel_coordinate=Point(x=5.0, y=5.0))  # region_mask=None
        seg_mask = np.zeros((10, 10), dtype=np.uint8)
        grid_mask = np.ones((10, 10), dtype=np.uint8)
        pp = PostProcessingParameters()
        score_candidate_targets([c], seg_mask, grid_mask, 1e-7, pp)
        assert c.score == pytest.approx(1.0)  # untouched


class TestSelectBestPerGridSquare:
    def _make_accepted(self, x, y, score):
        c = CandidateTarget(pixel_coordinate=Point(x=float(x), y=float(y)), score=score)
        return c

    def test_lower_scorer_becomes_superseded(self):
        # Two candidates close together (same hole)
        high = self._make_accepted(5, 5, score=0.9)
        low = self._make_accepted(6, 5, score=0.3)
        # Grid mask: all bars except a hole at center
        grid = np.ones((20, 20), dtype=np.uint8)
        grid[2:18, 2:18] = 0  # one big hole
        result = select_best_per_grid_square([high, low], grid)
        assert high.rejection_reason is None
        assert low.rejection_reason == REJECT_SUPERSEDED
        assert low.score == pytest.approx(
            0.3
        )  # score preserved; only rejection_reason is set

    def test_single_candidate_unchanged(self):
        c = self._make_accepted(5, 5, score=0.7)
        grid = np.ones((20, 20), dtype=np.uint8)
        grid[2:18, 2:18] = 0
        select_best_per_grid_square([c], grid)
        assert c.rejection_reason is None

    def test_already_rejected_untouched(self):
        rejected = CandidateTarget(
            pixel_coordinate=Point(x=5.0, y=5.0),
            score=0.0,
            rejection_reason="too_small",
        )
        grid = np.ones((20, 20), dtype=np.uint8)
        grid[2:18, 2:18] = 0
        select_best_per_grid_square([rejected], grid)
        assert rejected.rejection_reason == "too_small"

    def test_no_holes_returns_unchanged(self):
        c = self._make_accepted(5, 5, score=0.5)
        grid = np.ones((10, 10), dtype=np.uint8)  # all bar, no holes
        select_best_per_grid_square([c], grid)
        assert c.rejection_reason is None


# ---------------------------------------------------------------------------
# generate_grid_mask_v2 — shape, dtype, structure, and parity with v1
# ---------------------------------------------------------------------------

# Shared parameter sets to run across multiple tests
_GRID_PARAMS = [
    dict(pixelsize=0.1, hole_width=5.0, pitch=7.0, bar_width=2.0, nrows=5, ncols=4),
    dict(pixelsize=0.05, hole_width=3.0, pitch=5.0, bar_width=2.0, nrows=3, ncols=3),
    dict(pixelsize=0.2, hole_width=10.0, pitch=15.0, bar_width=5.0, nrows=2, ncols=2),
]


class TestGenerateGridMaskV2:
    """Tests for generate_grid_mask_v2 correctness (shape, dtype, structure)."""

    def test_output_dtype_is_uint8(self):
        m = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0)
        assert m.dtype == np.uint8

    def test_output_only_contains_0_and_1(self):
        m = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0)
        assert set(np.unique(m)).issubset({0, 1})

    def test_holes_exist(self):
        """At least some pixels must be 0 (holes)."""
        m = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0)
        assert np.any(m == 0)

    def test_bars_exist(self):
        """At least some pixels must be 1 (bars)."""
        m = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0)
        assert np.any(m == 1)

    def test_output_shape_matches_v1(self):
        for p in _GRID_PARAMS:
            v1 = generate_grid_mask(**p)
            v2 = generate_grid_mask_v2(**p)
            assert v1.shape == v2.shape, f"Shape mismatch for params {p}"

    def test_hole_fraction_reasonable(self):
        """Holes should occupy approximately nrows*ncols*hole_px^2 / (H*W) of the area."""
        pixelsize, hole_width, pitch, bar_width = 0.1, 5.0, 7.0, 2.0
        nrows, ncols = 5, 4
        m = generate_grid_mask_v2(
            pixelsize, hole_width, pitch, bar_width, nrows=nrows, ncols=ncols
        )
        hole_px = hole_width / pixelsize
        pitch_px = pitch / pixelsize
        bar_px = bar_width / pixelsize
        mheight = int(nrows * pitch_px + bar_px)
        mwidth = int(ncols * pitch_px + bar_px)
        expected = (nrows * ncols * hole_px**2) / (mheight * mwidth)
        hole_fraction = np.mean(m == 0)
        assert abs(hole_fraction - expected) < 0.02, (
            f"hole_fraction={hole_fraction:.3f}, expected≈{expected:.3f}"
        )

    def test_border_is_bar(self):
        """All four edges of the mask should be bar (1), not hole."""
        m = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0)
        assert np.all(m[0, :] == 1), "top row should be bar"
        assert np.all(m[-1, :] == 1), "bottom row should be bar"
        assert np.all(m[:, 0] == 1), "left col should be bar"
        assert np.all(m[:, -1] == 1), "right col should be bar"

    def test_nrows_ncols_scale_shape(self):
        """Doubling nrows should approximately double height."""
        base = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0, nrows=2, ncols=2)
        tall = generate_grid_mask_v2(0.1, 5.0, 7.0, 2.0, nrows=4, ncols=2)
        assert tall.shape[0] > base.shape[0]
        assert tall.shape[1] == base.shape[1]


class TestGenerateGridMaskParity:
    """Comparison tests between v1 and v2.

    NOTE: v1 uses cv2.rectangle with thickness=30, so its holes are
    (hole_px - 30) pixels wide — a known hack called out in the source
    ("ADI: This is absurd, will redo").  v2 is the correct implementation
    with full-size holes.  The two outputs therefore differ in pixel values,
    but must agree on output shape, dtype, and high-level structure.
    """

    def _both(self, params: dict):
        return generate_grid_mask(**params), generate_grid_mask_v2(**params)

    def test_shape_matches_for_all_param_sets(self):
        for p in _GRID_PARAMS:
            v1, v2 = self._both(p)
            assert v1.shape == v2.shape, (
                f"Shape mismatch: {v1.shape} vs {v2.shape} for {p}"
            )

    def test_dtype_matches(self):
        v1, v2 = self._both(_GRID_PARAMS[0])
        assert v1.dtype == v2.dtype

    def test_both_have_holes_and_bars(self):
        for p in _GRID_PARAMS:
            v1, v2 = self._both(p)
            assert np.any(v1 == 0) and np.any(v1 == 1), (
                f"v1 missing holes or bars for {p}"
            )
            assert np.any(v2 == 0) and np.any(v2 == 1), (
                f"v2 missing holes or bars for {p}"
            )

    def test_both_have_bar_border(self):
        """Both implementations must have bars on all four edges."""
        for p in _GRID_PARAMS:
            v1, v2 = self._both(p)
            for name, m in [("v1", v1), ("v2", v2)]:
                assert np.all(m[0, :] == 1), f"{name}: top row not bar for {p}"
                assert np.all(m[-1, :] == 1), f"{name}: bottom row not bar for {p}"
                assert np.all(m[:, 0] == 1), f"{name}: left col not bar for {p}"
                assert np.all(m[:, -1] == 1), f"{name}: right col not bar for {p}"

    def test_v2_has_larger_holes_than_v1(self):
        """v2 holes are full hole_width/pixelsize px; v1 holes are shrunk by cv2 thickness."""
        p = _GRID_PARAMS[0]
        v1, v2 = self._both(p)
        assert np.mean(v2 == 0) > np.mean(v1 == 0), (
            "v2 should have a larger hole fraction than v1 (v1 holes are artificially small)"
        )
