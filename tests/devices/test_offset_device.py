"""Unit tests for fibsem.devices.base — ChamberDeviceGeometry and ChamberDevice.

All tests run without hardware (no microscope connection required).
"""
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from fibsem.devices.base import ChamberDevice, ChamberDeviceGeometry
from fibsem.structures import FibsemStagePosition, RangeLimit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geometry(
    offset_x=0.0,
    offset_y=0.0,
    offset_z=0.0,
    device_r=0.0,
    device_t=0.0,
    x_range=(-100e-3, 100e-3),
    y_range=(-100e-3, 100e-3),
    z_range=(-10e-3, 10e-3),
    available_orientations=None,
):
    if available_orientations is None:
        available_orientations = ["FM", "FIB"]
    return ChamberDeviceGeometry(
        offset=FibsemStagePosition(x=offset_x, y=offset_y, z=offset_z),
        available_range={
            "x": RangeLimit(x_range[0], x_range[1]),
            "y": RangeLimit(y_range[0], y_range[1]),
            "z": RangeLimit(z_range[0], z_range[1]),
        },
        default_orientation=FibsemStagePosition(r=device_r, t=device_t),
        available_orientations=available_orientations,
    )


def _make_parent(
    stage_r=0.0,
    stage_t=0.0,
    rotation_reference=0.0,
    rotation_180=180.0,
    pretilt_deg=0.0,
    sem_column_tilt_deg=0.0,
    fib_column_tilt_deg=52.0,
    orientation="FM",
):
    parent = MagicMock()
    parent.get_stage_position.return_value = FibsemStagePosition(r=stage_r, t=stage_t)
    parent.system.stage.rotation_reference = rotation_reference
    parent.system.stage.rotation_180 = rotation_180
    parent.system.stage.shuttle_pre_tilt = pretilt_deg
    parent.system.electron.column_tilt = sem_column_tilt_deg
    parent.system.ion.column_tilt = fib_column_tilt_deg
    parent.get_stage_orientation.return_value = orientation
    # transform tests: get_target_position returns the input position unchanged
    # (rotation behaviour is tested in microscope tests; here we test offset logic only)
    parent.get_target_position.side_effect = lambda pos, target_orientation: pos
    return parent


class _ConcreteDevice(ChamberDevice):
    """Minimal concrete subclass for testing."""
    pass


# ---------------------------------------------------------------------------
# ChamberDeviceGeometry serialization
# ---------------------------------------------------------------------------

class TestChamberDeviceGeometrySerialisation:
    def test_roundtrip(self):
        geom = _make_geometry(
            offset_x=48.8e-3, offset_y=1e-3, offset_z=0.0,
            device_r=np.deg2rad(180), device_t=np.deg2rad(-52),
            x_range=(40e-3, 60e-3),
            available_orientations=["FM", "FIB"],
        )
        d = geom.to_dict()
        geom2 = ChamberDeviceGeometry.from_dict(d)

        assert geom2.offset.x == pytest.approx(geom.offset.x)
        assert geom2.offset.y == pytest.approx(geom.offset.y)
        assert geom2.available_range["x"].min == pytest.approx(geom.available_range["x"].min)
        assert geom2.available_range["x"].max == pytest.approx(geom.available_range["x"].max)
        assert geom2.default_orientation.r == pytest.approx(geom.default_orientation.r)
        assert geom2.default_orientation.t == pytest.approx(geom.default_orientation.t)
        assert geom2.available_orientations == geom.available_orientations

    def test_no_column_tilt_field(self):
        """column_tilt must not be a field on ChamberDeviceGeometry."""
        geom = _make_geometry()
        assert not hasattr(geom, "column_tilt")


# ---------------------------------------------------------------------------
# transform_to_device_frame
# ---------------------------------------------------------------------------

class TestTransformToDeviceFrame:
    def test_zero_offset(self):
        """Zero offset: transform returns whatever get_target_position returns."""
        geom = _make_geometry(offset_x=0.0, offset_y=0.0)
        device = _ConcreteDevice(geometry=geom, parent=_make_parent())

        fibsem_pos = FibsemStagePosition(x=10e-6, y=5e-6, z=0.0, r=0.0, t=0.0)
        result = device.transform_to_device_frame(fibsem_pos)

        assert result.x == pytest.approx(fibsem_pos.x)
        assert result.y == pytest.approx(fibsem_pos.y)

    def test_x_offset_added(self):
        """Device is +48.8mm from FIBSEM: stage must move +48.8mm to put the
        same feature under the device."""
        offset_x = 48.8e-3
        geom = _make_geometry(offset_x=offset_x, offset_y=0.0)
        device = _ConcreteDevice(geometry=geom, parent=_make_parent())

        fibsem_pos = FibsemStagePosition(x=5e-3, y=0.0, z=0.0, r=0.0, t=0.0)
        result = device.transform_to_device_frame(fibsem_pos)

        assert result.x == pytest.approx(fibsem_pos.x + offset_x)
        assert result.y == pytest.approx(0.0)

    def test_offset_added_after_get_target_position(self):
        """Offset is added to the result of get_target_position, not the input."""
        geom = _make_geometry(offset_x=5e-3, offset_y=2e-3)
        parent = _make_parent()
        rotated = FibsemStagePosition(x=10e-3, y=3e-3, z=0.0, r=0.0, t=0.0)
        parent.get_target_position.side_effect = lambda pos, target_orientation: rotated

        device = _ConcreteDevice(geometry=geom, parent=parent)
        result = device.transform_to_device_frame(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0)
        )
        assert result.x == pytest.approx(rotated.x + geom.offset.x)
        assert result.y == pytest.approx(rotated.y + geom.offset.y)

    def test_z_offset_added(self):
        geom = _make_geometry(offset_x=0.0, offset_z=1e-3)
        device = _ConcreteDevice(geometry=geom, parent=_make_parent())

        fibsem_pos = FibsemStagePosition(x=0.0, y=0.0, z=3.5e-3, r=0.0, t=0.0)
        result = device.transform_to_device_frame(fibsem_pos)
        assert result.z == pytest.approx(fibsem_pos.z + 1e-3)

    def test_no_geometry_raises(self):
        device = _ConcreteDevice(geometry=None, parent=_make_parent())
        with pytest.raises(ValueError):
            device.transform_to_device_frame(FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0))


# ---------------------------------------------------------------------------
# stable_move — base class raises NotImplementedError
# ---------------------------------------------------------------------------

class TestStableMoveBase:
    def test_raises_not_implemented(self):
        geom = _make_geometry()
        device = _ConcreteDevice(geometry=geom, parent=_make_parent())
        with pytest.raises(NotImplementedError):
            device.stable_move(dx=0.0, dy=1e-6)


# ---------------------------------------------------------------------------
# is_accessible
# ---------------------------------------------------------------------------

class TestIsAccessible:
    def test_within_range(self):
        # Device offset: +50mm. Available range: 40–60 mm.
        # device_x = fibsem_x + 50mm, so fibsem_x must be in (-10mm, 10mm) for accessibility.
        geom = _make_geometry(offset_x=50e-3, x_range=(40e-3, 60e-3))
        device = _ConcreteDevice(geometry=geom, parent=_make_parent())

        # fibsem_x = 20mm → device_x = 70mm → outside range
        outside = FibsemStagePosition(x=20e-3, y=0.0, z=0.0, r=0.0, t=0.0)
        assert not device.is_accessible(outside)

        # fibsem_x = 5mm → device_x = 55mm → inside range
        inside = FibsemStagePosition(x=5e-3, y=0.0, z=0.0, r=0.0, t=0.0)
        assert device.is_accessible(inside)


# ---------------------------------------------------------------------------
# has_valid_orientation
# ---------------------------------------------------------------------------

class TestHasValidOrientation:
    def test_valid_orientation_returns_true(self):
        geom = _make_geometry(available_orientations=["FM", "FIB"])
        device = _ConcreteDevice(geometry=geom, parent=_make_parent(orientation="FM"))
        assert device.has_valid_orientation() is True

    def test_invalid_orientation_returns_false(self):
        geom = _make_geometry(available_orientations=["FM", "FIB"])
        device = _ConcreteDevice(geometry=geom, parent=_make_parent(orientation="SEM"))
        assert device.has_valid_orientation() is False

    def test_no_geometry_returns_true(self):
        device = _ConcreteDevice(geometry=None, parent=_make_parent(orientation="SEM"))
        assert device.has_valid_orientation() is True


# ---------------------------------------------------------------------------
# FluorescenceMicroscope — inherits ChamberDevice, overrides stable_move
# ---------------------------------------------------------------------------

class TestFluorescenceMicroscopeInheritance:
    def _make_fm(self, geometry=None, parent=None):
        from fibsem.fm.microscope import FluorescenceMicroscope

        class _FM(FluorescenceMicroscope):
            def acquire_image(self, *a, **kw): ...

        return _FM(parent=parent, geometry=geometry)

    def test_fm_is_chamber_device(self):
        from fibsem.fm.microscope import FluorescenceMicroscope
        assert issubclass(FluorescenceMicroscope, ChamberDevice)

    def test_fm_default_geometry_is_none(self):
        fm = self._make_fm()
        assert fm.geometry is None

    def test_fm_geometry_set_on_init(self):
        geom = _make_geometry(offset_x=48.8e-3)
        fm = self._make_fm(geometry=geom)
        assert fm.geometry is geom

    def test_fm_has_column_tilt_attribute(self):
        fm = self._make_fm()
        assert hasattr(fm, "column_tilt")
        assert fm.column_tilt == 0.0

    def test_fm_stable_move_zero_tilt(self):
        """At zero pretilt, zero column_tilt, zero stage_tilt, dy=10µm → y_move=10µm, z_move=0."""
        parent = _make_parent(
            stage_r=0.0, stage_t=0.0,
            pretilt_deg=0.0, sem_column_tilt_deg=0.0,
            rotation_reference=0.0,
        )
        captured = {}
        parent.move_stage_relative.side_effect = lambda pos: captured.update({"pos": pos})
        parent.get_stage_position.side_effect = [
            FibsemStagePosition(r=0.0, t=0.0),
            FibsemStagePosition(r=0.0, t=0.0),
        ]

        fm = self._make_fm(parent=parent)
        fm.column_tilt = 0.0
        fm.stable_move(dx=0.0, dy=10e-6)

        assert captured["pos"].y == pytest.approx(10e-6)
        assert captured["pos"].z == pytest.approx(0.0, abs=1e-15)

    def test_fm_has_valid_orientation_no_geometry(self):
        parent = _make_parent(orientation="SEM")
        fm = self._make_fm(parent=parent)
        fm._allow_unknown_orientations = False
        fm.valid_orientations = ["FM", "FIB"]
        assert fm.has_valid_orientation() is False

    def test_fm_has_valid_orientation_with_geometry(self):
        geom = _make_geometry(available_orientations=["FM", "FIB"])
        parent = _make_parent(orientation="FM")
        fm = self._make_fm(parent=parent, geometry=geom)
        fm._allow_unknown_orientations = False
        assert fm.has_valid_orientation() is True


# ---------------------------------------------------------------------------
# FluorescenceConfiguration geometry field
# ---------------------------------------------------------------------------

class TestFluorescenceConfigurationGeometry:
    def _minimal_config_dict(self, geometry_dict=None):
        from fibsem.fm.structures import ChannelSettings, ZParameters, OverviewParameters, ZStackOrder
        return {
            "channel_settings": [ChannelSettings(
                name="ch1", excitation_wavelength=488,
                emission_wavelength=520, power=10.0,
                exposure_time=0.1, color="green", gain=1.0,
            ).to_dict()],
            "z_parameters": ZParameters(
                zmin=-5e-6, zmax=5e-6, zstep=1e-6,
                order=ZStackOrder.CHANNEL,
            ).to_dict(),
            "overview_parameters": OverviewParameters(rows=2, cols=2, overlap=0.1).to_dict(),
            "geometry": geometry_dict,
        }

    def test_no_geometry_loads_as_none(self):
        from fibsem.fm.structures import FluorescenceConfiguration
        cfg = FluorescenceConfiguration.from_dict(self._minimal_config_dict())
        assert cfg.geometry is None

    def test_geometry_roundtrip(self):
        from fibsem.fm.structures import FluorescenceConfiguration
        geom = _make_geometry(offset_x=48.8e-3)
        cfg = FluorescenceConfiguration.from_dict(self._minimal_config_dict(geom.to_dict()))
        assert cfg.geometry is not None
        assert cfg.geometry.offset.x == pytest.approx(48.8e-3)

    def test_old_config_without_geometry_key(self):
        from fibsem.fm.structures import FluorescenceConfiguration
        d = self._minimal_config_dict()
        del d["geometry"]
        cfg = FluorescenceConfiguration.from_dict(d)
        assert cfg.geometry is None
