"""Tests for the pre-correlation refractive-index correction (FM surface mode).

Covers:
- apply_z_surface_correction (pure math)
- run_correlation_from_data applying the correction transiently
- CorrelationInputData serialization of the new fields (incl. legacy files)
"""
import math

import numpy as np
import pytest

import fibsem.correlation.correlation_v2 as correlation_v2
from fibsem.correlation.correlation_v2 import run_correlation_from_data
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationResult,
    PointType,
    PointXYZ,
    apply_z_surface_correction,
)


def _coord(x=0.0, y=0.0, z=0.0, pt=PointType.FIB) -> Coordinate:
    return Coordinate(point=PointXYZ(x=x, y=y, z=z), point_type=pt)


# ---------------------------------------------------------------------------
# apply_z_surface_correction
# ---------------------------------------------------------------------------


class TestApplyZSurfaceCorrection:

    def test_basic_scaling(self):
        poi = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        out = apply_z_surface_correction(poi, surface_z=10.0, correction_factor=1.5)
        # 10 + (30 - 10) * 1.5 = 40
        assert out[0, 2] == pytest.approx(40.0)
        # x, y untouched
        assert out[0, 0] == pytest.approx(10.0)
        assert out[0, 1] == pytest.approx(20.0)

    def test_signed_poi_above_surface(self):
        """Scaling is signed — works regardless of stack z direction."""
        poi = np.array([[0.0, 0.0, 30.0]], dtype=np.float32)
        out = apply_z_surface_correction(poi, surface_z=50.0, correction_factor=1.5)
        # 50 + (30 - 50) * 1.5 = 20
        assert out[0, 2] == pytest.approx(20.0)

    def test_factor_one_is_noop(self):
        poi = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = apply_z_surface_correction(poi, surface_z=10.0, correction_factor=1.0)
        assert np.allclose(out, poi)

    def test_poi_at_surface_unchanged(self):
        poi = np.array([[1.0, 2.0, 10.0]], dtype=np.float32)
        out = apply_z_surface_correction(poi, surface_z=10.0, correction_factor=2.0)
        assert out[0, 2] == pytest.approx(10.0)

    def test_all_pois_corrected(self):
        poi = np.array(
            [[0.0, 0.0, 20.0], [0.0, 0.0, 30.0], [0.0, 0.0, 5.0]], dtype=np.float32
        )
        out = apply_z_surface_correction(poi, surface_z=10.0, correction_factor=2.0)
        assert out[:, 2] == pytest.approx([30.0, 50.0, 0.0])

    def test_input_not_mutated(self):
        poi = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        original = poi.copy()
        apply_z_surface_correction(poi, surface_z=0.0, correction_factor=2.0)
        assert np.array_equal(poi, original)

    def test_empty_array(self):
        poi = np.zeros((0, 3), dtype=np.float32)
        out = apply_z_surface_correction(poi, surface_z=0.0, correction_factor=2.0)
        assert out.shape == (0, 3)


# ---------------------------------------------------------------------------
# run_correlation_from_data — transient application at the correlate() seam
# ---------------------------------------------------------------------------


def _fake_run_correlation(captured):
    """Stub for correlation_v2.run_correlation that records its POI input."""

    def fake(
        fib_coords,
        fm_coords,
        poi_coords,
        image_props,
        rotation_center,
        path=None,
        fib_image_filename="",
        fm_image_filename="",
    ):
        captured["poi_coords"] = np.array(poi_coords, copy=True)
        n_poi = len(poi_coords)
        n_markers = len(fib_coords)
        return {
            "input": {},
            "output": {
                "transformation": {
                    "scale": 1.0,
                    "rotation_eulers": [0.0, 0.0, 0.0],
                    # NOTE: despite the name, this field stores Rigid3D.q,
                    # the 3x3 rotation matrix (identity here)
                    "rotation_quaternion": np.eye(3).tolist(),
                    "translation_around_rotation_center_zero": [0.0, 0.0, 0.0],
                    "translation_around_rotation_center_custom": [0.0, 0.0, 0.0],
                },
                "error": {
                    "rms_error": 0.0,
                    "mean_absolute_error": [0.0, 0.0],
                    "delta_2d": [[0.0] * n_markers, [0.0] * n_markers],
                    "reprojected_3d": [
                        [0.0] * n_markers,
                        [0.0] * n_markers,
                        [0.0] * n_markers,
                    ],
                },
                "poi": [
                    {"image_px": [0.0, 0.0], "px": [0.0, 0.0], "px_m": [0.0, 0.0]}
                    for _ in range(n_poi)
                ],
            },
        }

    return fake


def _make_input_data(**kwargs) -> CorrelationInputData:
    return CorrelationInputData(
        fib_coordinates=[_coord(x=i, y=i, pt=PointType.FIB) for i in range(4)],
        fm_coordinates=[_coord(x=i, y=i, z=5.0, pt=PointType.FM) for i in range(4)],
        poi_coordinates=[
            _coord(x=1.0, y=2.0, z=30.0, pt=PointType.POI),
            _coord(x=3.0, y=4.0, z=5.0, pt=PointType.POI),
        ],
        **kwargs,
    )


class TestRunCorrelationPreCorrection:

    @pytest.fixture
    def captured(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            correlation_v2, "run_correlation", _fake_run_correlation(captured)
        )
        return captured

    def test_correction_applied_to_all_pois(self, captured):
        data = _make_input_data(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
            ri_pre_correction_factor=1.5,
        )
        result = run_correlation_from_data(data)
        # POI 1: 10 + (30 - 10) * 1.5 = 40; POI 2: 10 + (5 - 10) * 1.5 = 2.5
        assert captured["poi_coords"][:, 2] == pytest.approx([40.0, 2.5])
        assert result.refractive_index_correction_factor == 1.5
        assert result.refractive_index_correction_mode == "pre"

    def test_ghost_poi_uncorrected(self, captured):
        """Ghost markers reproject the ORIGINAL poi through the same transform."""
        data = _make_input_data(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
            ri_pre_correction_factor=1.5,
        )
        result = run_correlation_from_data(data)
        # stub transform is identity → image_px equals the original (x, y)
        assert len(result.poi_uncorrected) == 2
        assert result.poi_uncorrected[0].image_px.x == pytest.approx(1.0)
        assert result.poi_uncorrected[0].image_px.y == pytest.approx(2.0)
        assert result.poi_uncorrected[1].image_px.x == pytest.approx(3.0)
        assert result.poi_uncorrected[1].image_px.y == pytest.approx(4.0)

    def test_no_ghost_without_correction(self, captured):
        data = _make_input_data(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
        )
        result = run_correlation_from_data(data)
        assert result.poi_uncorrected == []

    def test_input_poi_not_mutated(self, captured):
        data = _make_input_data(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
            ri_pre_correction_factor=1.5,
        )
        run_correlation_from_data(data)
        # stored coordinates keep the user-picked z (correction is transient)
        assert data.poi_coordinates[0].point.z == pytest.approx(30.0)
        assert data.poi_coordinates[1].point.z == pytest.approx(5.0)

    def test_rerun_is_idempotent(self, captured):
        data = _make_input_data(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
            ri_pre_correction_factor=1.5,
        )
        run_correlation_from_data(data)
        first = captured["poi_coords"].copy()
        run_correlation_from_data(data)
        assert np.allclose(captured["poi_coords"], first)

    def test_no_factor_no_correction(self, captured):
        data = _make_input_data(
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
        )
        result = run_correlation_from_data(data)
        assert captured["poi_coords"][:, 2] == pytest.approx([30.0, 5.0])
        assert result.refractive_index_correction_factor is None
        assert result.refractive_index_correction_mode is None

    def test_no_fm_surface_no_correction(self, captured):
        data = _make_input_data(ri_pre_correction_factor=1.5)
        result = run_correlation_from_data(data)
        assert captured["poi_coords"][:, 2] == pytest.approx([30.0, 5.0])
        assert result.refractive_index_correction_mode is None

    def test_fib_surface_does_not_trigger_pre_correction(self, captured):
        data = _make_input_data(
            surface_coordinate=_coord(y=100.0, pt=PointType.SURFACE),
            ri_pre_correction_factor=1.5,
        )
        result = run_correlation_from_data(data)
        assert captured["poi_coords"][:, 2] == pytest.approx([30.0, 5.0])
        assert result.refractive_index_correction_mode is None

    def test_both_surfaces_raise(self):
        """FIB and FM surface points are mutually exclusive at the engine too."""
        data = _make_input_data(
            surface_coordinate=_coord(y=100.0, pt=PointType.SURFACE),
            fm_surface_coordinate=_coord(z=10.0, pt=PointType.SURFACE_FM),
        )
        with pytest.raises(ValueError, match="only one surface point"):
            run_correlation_from_data(data)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:

    def test_input_data_roundtrip(self):
        data = _make_input_data(
            fm_surface_coordinate=_coord(x=1.0, y=2.0, z=10.0, pt=PointType.SURFACE_FM),
            ri_pre_correction_factor=1.47,
        )
        data2 = CorrelationInputData.from_dict(data.to_dict())
        assert data2.fm_surface_coordinate is not None
        assert data2.fm_surface_coordinate.point.z == pytest.approx(10.0)
        assert data2.fm_surface_coordinate.point_type is PointType.SURFACE_FM
        assert data2.ri_pre_correction_factor == pytest.approx(1.47)

    def test_legacy_dict_without_new_keys(self):
        """Files written before the FM-surface feature still load."""
        legacy = {
            "fib_coordinates": [],
            "fm_coordinates": [],
            "poi_coordinates": [],
            "surface_coordinate": None,
            "method": "multi-point",
        }
        data = CorrelationInputData.from_dict(legacy)
        assert data.fm_surface_coordinate is None
        assert data.ri_pre_correction_factor is None

    def test_point_type_roundtrip(self):
        c = _coord(pt=PointType.SURFACE_FM)
        c2 = Coordinate.from_dict(c.to_dict())
        assert c2.point_type is PointType.SURFACE_FM

    def test_result_mode_roundtrip(self):
        r = CorrelationResult(
            refractive_index_correction_factor=1.5,
            refractive_index_correction_mode="pre",
        )
        r2 = CorrelationResult.from_dict(r.to_dict())
        assert r2.refractive_index_correction_mode == "pre"

    def test_poi_uncorrected_roundtrip(self):
        from fibsem.correlation.structures import CorrelationPointOfInterest
        from fibsem.structures import Point

        r = CorrelationResult(
            poi_uncorrected=[
                CorrelationPointOfInterest(image_px=Point(x=11.0, y=22.0))
            ],
        )
        r2 = CorrelationResult.from_dict(r.to_dict())
        assert len(r2.poi_uncorrected) == 1
        assert r2.poi_uncorrected[0].image_px.x == pytest.approx(11.0)

    def test_legacy_result_without_poi_uncorrected(self):
        r = CorrelationResult()
        d = r.to_dict()
        d.pop("poi_uncorrected")
        r2 = CorrelationResult.from_dict(d)
        assert r2.poi_uncorrected == []

    def test_input_dataframe_contains_fm_surface(self):
        data = _make_input_data(
            fm_surface_coordinate=_coord(x=1.0, y=2.0, z=10.0, pt=PointType.SURFACE_FM),
        )
        r = CorrelationResult(input_data=data)
        df = r.to_input_dataframe()
        rows = df[df["type"] == PointType.SURFACE_FM.value]
        assert len(rows) == 1
        assert rows.iloc[0]["z_px"] == pytest.approx(10.0)
