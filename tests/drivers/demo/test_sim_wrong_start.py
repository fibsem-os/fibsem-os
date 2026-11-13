"""A simulated session can start wrong, for something to notice and put right:
the scan rotation the beams boot at (the FM camera's mount is in
tests/fm/test_sim_camera_mount.py)."""

import numpy as np
import pytest

from fibsem import utils
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import BeamType


def _demo(**sample):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage.sim_error = None
    microscope.system.sim["sample"] = {"enabled": True, **sample}
    microscope._connection.setup_sample_scene()
    return microscope


def test_the_beams_boot_at_the_scan_rotation_asked_for():
    microscope = _demo(scan_rotation_at_start={"electron": 180, "ion": 0})
    try:
        assert microscope.get_scan_rotation(BeamType.ELECTRON) == pytest.approx(np.pi)
        assert microscope.get_scan_rotation(BeamType.ION) == 0.0
        # and it is the beam's own setting: putting it right is one write
        microscope.set_scan_rotation(0.0, BeamType.ELECTRON)
        assert microscope.get_scan_rotation(BeamType.ELECTRON) == 0.0
    finally:
        microscope.disconnect()


def test_a_session_starts_right_unless_asked():
    scene = SampleScene.from_config({})
    assert scene.scan_rotation_at_start == {}
    assert scene.fm_camera_mount is None
    microscope = _demo()
    try:
        assert microscope.get_scan_rotation(BeamType.ELECTRON) == 0.0
    finally:
        microscope.disconnect()


def test_an_unknown_mount_is_refused():
    with pytest.raises(ValueError, match="mount"):
        SampleScene.from_config({"fm_camera_mount": "sideways"})
