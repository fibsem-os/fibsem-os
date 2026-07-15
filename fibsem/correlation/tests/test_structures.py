"""Unit tests for CorrelationResult and CorrelationInputData."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.structures import Point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poi(image_px_x: float = 100.0, image_px_y: float = 200.0) -> CorrelationPointOfInterest:
    return CorrelationPointOfInterest(
        image_px=Point(x=image_px_x, y=image_px_y),
        px=Point(x=0.0, y=0.0),
        px_m=Point(x=0.0, y=0.0),
    )


def _make_coord(x: float = 0.0, y: float = 0.0, z: float = 0.0,
                pt: PointType = PointType.FIB) -> Coordinate:
    return Coordinate(point=PointXYZ(x=x, y=y, z=z), point_type=pt)


def _make_fake_fib_image(shape=(512, 512), pixel_size: float = 1e-8):
    """Duck-typed stand-in for FibsemImage (data + metadata.pixel_size.x)."""
    return SimpleNamespace(
        data=np.zeros(shape, dtype=np.uint8),
        metadata=SimpleNamespace(
            pixel_size=SimpleNamespace(x=pixel_size),
            image_settings=SimpleNamespace(filename="fake.tif"),
        ),
    )


def _make_result() -> CorrelationResult:
    return CorrelationResult(
        poi=[_make_poi(100.0, 200.0)],
        scale=1.5,
        rotation_eulers=[10.0, 20.0, 30.0],
        rotation_quaternion=[0.9, 0.1, 0.2, 0.3],
        translation=[1.0, 2.0, 3.0],
        rms_error=0.42,
        mean_absolute_error=[0.3, 0.5],
        delta_2d=[Point(x=0.1, y=0.2)],
        reprojected_3d=[PointXYZ(x=1.0, y=2.0, z=3.0)],
        refractive_index_correction_factor=1.47,
        refractive_index_correction_mode="post",
    )


def _make_corrected_result(
    poi_list=None, fib_image=None
) -> CorrelationResult:
    """Result with input_data carrying a FIB surface at y=200."""
    surface = _make_coord(y=200.0, pt=PointType.SURFACE)
    data = CorrelationInputData(surface_coordinate=surface, fib_image=fib_image)
    return CorrelationResult(
        poi=poi_list if poi_list is not None else [_make_poi(image_px_y=300.0)],
        input_data=data,
    )


# ---------------------------------------------------------------------------
# apply_refractive_index_correction (post mode: FIB image space)
# ---------------------------------------------------------------------------

class TestApplyRefractiveIndexCorrection:

    def test_basic_correction(self):
        """Depth below surface is scaled by the correction factor."""
        result = _make_corrected_result()
        result.apply_refractive_index_correction(1.5)
        # expected: 200 + (300 - 200) * 1.5 = 350
        assert math.isclose(result.poi[0].image_px.y, 350.0)

    def test_px_m_updated(self):
        """px and px_m coordinates are recalculated from image_px after correction."""
        pixel_size = 1e-8
        fib_image = _make_fake_fib_image(shape=(512, 512), pixel_size=pixel_size)
        result = _make_corrected_result(fib_image=fib_image)
        result.apply_refractive_index_correction(1.5)
        cy = 512 / 2.0  # 256
        corrected_y = 350.0
        assert math.isclose(result.poi[0].px.y, -(corrected_y - cy))
        assert math.isclose(result.poi[0].px_m.y, -(corrected_y - cy) * pixel_size)

    def test_no_input_data_raises(self):
        result = CorrelationResult(poi=[_make_poi()])
        with pytest.raises(ValueError, match="input_data"):
            result.apply_refractive_index_correction(1.5)

    def test_no_poi_raises(self):
        result = _make_corrected_result(poi_list=[])
        with pytest.raises(ValueError, match="POI"):
            result.apply_refractive_index_correction(1.5)

    def test_no_surface_raises(self):
        data = CorrelationInputData()
        result = CorrelationResult(poi=[_make_poi()], input_data=data)
        with pytest.raises(ValueError, match="surface_coordinate"):
            result.apply_refractive_index_correction(1.5)

    def test_factor_and_mode_stored(self):
        result = _make_corrected_result()
        result.apply_refractive_index_correction(1.47)
        assert result.refractive_index_correction_factor == 1.47
        assert result.refractive_index_correction_mode == "post"

    def test_ghost_snapshot_stored(self):
        """The uncorrected POI 1 position is kept for the ghost overlay."""
        result = _make_corrected_result()
        result.apply_refractive_index_correction(1.5)
        assert len(result.poi_uncorrected) == 1
        assert math.isclose(result.poi_uncorrected[0].image_px.y, 300.0)
        assert math.isclose(result.poi[0].image_px.y, 350.0)

    def test_updated_at_advances(self):
        result = _make_corrected_result()
        before = result.updated_at
        time.sleep(0.01)
        result.apply_refractive_index_correction(1.47)
        assert result.updated_at > before

    def test_only_first_poi_modified(self):
        """Only poi[0] is corrected; other POIs are unchanged."""
        poi0 = _make_poi(image_px_y=300.0)
        poi1 = _make_poi(image_px_y=400.0)
        result = _make_corrected_result(poi_list=[poi0, poi1])
        result.apply_refractive_index_correction(2.0)
        assert math.isclose(result.poi[0].image_px.y, 400.0)   # corrected
        assert math.isclose(result.poi[1].image_px.y, 400.0)   # unchanged


# ---------------------------------------------------------------------------
# CorrelationResult serialization
# ---------------------------------------------------------------------------

class TestCorrelationResultSerialization:

    def test_roundtrip(self):
        r = _make_result()
        r2 = CorrelationResult.from_dict(r.to_dict())
        assert r2.scale == r.scale
        assert r2.rms_error == r.rms_error
        assert r2.rotation_eulers == r.rotation_eulers
        assert r2.refractive_index_correction_factor == r.refractive_index_correction_factor
        assert r2.refractive_index_correction_mode == r.refractive_index_correction_mode
        assert len(r2.poi) == 1
        assert math.isclose(r2.poi[0].image_px.x, 100.0)
        assert math.isclose(r2.poi[0].image_px.y, 200.0)
        assert len(r2.delta_2d) == 1
        assert math.isclose(r2.delta_2d[0].x, 0.1)
        assert len(r2.reprojected_3d) == 1
        assert math.isclose(r2.reprojected_3d[0].z, 3.0)

    def test_save_load(self, tmp_path):
        r = _make_result()
        path = str(tmp_path / "result.json")
        r.save(path)
        r2 = CorrelationResult.load(path)
        assert r2.scale == r.scale
        assert r2.rms_error == r.rms_error
        assert r2.refractive_index_correction_factor == r.refractive_index_correction_factor
        assert len(r2.poi) == 1

    def test_input_data_embedded_and_restored(self):
        data = CorrelationInputData(
            fib_coordinates=[_make_coord(1.0, 2.0, 3.0, PointType.FIB)],
            fm_coordinates=[_make_coord(4.0, 5.0, 6.0, PointType.FM)],
            poi_coordinates=[_make_coord(7.0, 8.0, 9.0, PointType.POI)],
        )
        r = _make_result()
        r.input_data = data
        r2 = CorrelationResult.from_dict(r.to_dict())
        assert r2.input_data is not None
        assert len(r2.input_data.fib_coordinates) == 1
        assert math.isclose(r2.input_data.fib_coordinates[0].point.x, 1.0)
        assert len(r2.input_data.fm_coordinates) == 1
        assert len(r2.input_data.poi_coordinates) == 1

    def test_empty_result_roundtrip(self):
        r = CorrelationResult()
        r2 = CorrelationResult.from_dict(r.to_dict())
        assert r2.poi == []
        assert r2.scale == 0.0
        assert r2.rms_error == 0.0
        assert r2.input_data is None
        assert r2.refractive_index_correction_factor is None
        assert r2.refractive_index_correction_mode is None

    def test_updated_at_preserved(self):
        r = _make_result()
        r2 = CorrelationResult.from_dict(r.to_dict())
        assert math.isclose(r2.updated_at, r.updated_at)
