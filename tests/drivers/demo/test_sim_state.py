"""The sample's state is kept in a file, so a later session carries on from it."""

import json
import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.sim import state as sim_state
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml")


def _session(path, **sample):
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    microscope.system.sim["sample"] = {
        "enabled": True,
        "seed": 3,
        "noise_sigma": 0.0,
        "noise_fraction": 0.0,
        "charging": 0.0,
        "coincidence_offset": 0.0,
        "state_file": str(path),
        **sample,
    }
    microscope._connection.setup_sample_scene()
    return microscope


def _mill(microscope, gap=2e-6):
    for sign in (1, -1):
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=10e-6,
                height=4e-6,
                depth=5e-6,
                centre_x=0.0,
                centre_y=sign * (gap / 2 + 2e-6),
            )
        )
    microscope.run_milling()


def _image(microscope, beam_type=BeamType.ELECTRON) -> np.ndarray:
    settings = ImageSettings(
        resolution=(384, 256),
        hfw=40e-6,
        dwell_time=1e-6,
        save=False,
        beam_type=beam_type,
    )
    return microscope.acquire_image(image_settings=settings).data.astype(np.float32)


def test_a_new_session_finds_the_sample_as_the_last_one_left_it(tmp_path):
    path = tmp_path / "sample.json"
    first = _session(path)
    try:
        untouched = _image(first)
        assert not path.exists()  # nothing has been done to the sample yet
        _mill(first)
        first._connection.sample_scene.deposit(30.0)
        milled = _image(first)
        left = sim_state.dump(first._connection.sample_scene)
    finally:
        first.disconnect()
    assert np.abs(milled - untouched).mean() > 1.0

    second = _session(path)
    try:
        scene = second._connection.sample_scene
        assert len(scene.milled) == len(left["milled"]) > 0
        again = _image(second)
        assert json.dumps(sim_state.dump(scene)) == json.dumps(left)
        # the same trenches, give or take the image's own noise
        assert np.abs(again - milled).mean() < 0.2 * np.abs(milled - untouched).mean()
        # and it carries on: a second pass is the sample's second, not its first
        _mill(second, gap=1e-6)
        assert [p["pass"] for p in scene.mill_passes] == [0, 1]
    finally:
        second.disconnect()
    assert len(json.loads(path.read_text())["passes"]) == 2


def test_another_sample_starts_fresh(tmp_path):
    path = tmp_path / "sample.json"
    first = _session(path)
    try:
        _mill(first)
    finally:
        first.disconnect()
    other = _session(path, seed=4)
    try:
        scene = other._connection.sample_scene
        assert scene.milled == []
        assert sim_state.read(scene, str(path)) is None
        # the file is the new sample's from its first change
        _mill(other)
        assert sim_state.read(scene, str(path))["sample"]["seed"] == 4
    finally:
        other.disconnect()


def test_no_state_file_keeps_nothing(tmp_path):
    microscope = _session(tmp_path / "unused.json", state_file=None)
    try:
        _mill(microscope)
        assert microscope._connection.sample_scene.on_change is None
    finally:
        microscope.disconnect()
    assert list(tmp_path.iterdir()) == []


def test_spot_burns_and_imaging_wear_come_back(tmp_path):
    path = tmp_path / "sample.json"
    first = _session(path, imaging_damage=1.0)
    try:
        scene = first._connection.sample_scene
        first.set_beam_current(7.6e-9, BeamType.ION)
        for _ in range(4):
            first.acquire_image(
                image_settings=ImageSettings(
                    resolution=(768, 512),
                    hfw=20e-6,
                    dwell_time=3e-6,
                    save=False,
                    beam_type=BeamType.ION,
                )
            )
        worn = [seen for seen in scene.exposures.values() if seen["region"] is not None]
        assert worn, "imaging at 7.6 nA should have worn the sample"
        left = sim_state.dump(scene)
    finally:
        first.disconnect()
    second = _session(path, imaging_damage=1.0)
    try:
        scene = second._connection.sample_scene
        assert len(scene.exposures) == len(left["exposures"])
        assert [seen["depth"] for seen in scene.exposures.values()] == [
            seen["depth"] for seen in left["exposures"]
        ]
        # a worn field of view is still the region the sample holds, so more
        # imaging deepens that one rather than adding another
        for seen in scene.exposures.values():
            if seen["region"] is not None:
                assert any(seen["region"] is region for region in scene.milled)
    finally:
        second.disconnect()


def test_a_file_that_cannot_be_read_starts_fresh(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text("{ not json")
    microscope = _session(path)
    try:
        scene = microscope._connection.sample_scene
        assert scene.milled == []
        assert sim_state.read(scene, str(path)) is None
    finally:
        microscope.disconnect()


@pytest.mark.parametrize("value", [np.arange(4.0).reshape(2, 2), (1.0, (2, 3)), None])
def test_values_come_back_as_they_were(value):
    back = sim_state._value(json.loads(json.dumps(sim_state._plain(value))))
    if isinstance(value, np.ndarray):
        assert np.array_equal(back, value)
    else:
        assert back == value


def _holder_session(path):
    from tests.drivers.demo.test_sim_scene_holder import (
        TFS_SHUTTLE_CONFIG,
        _calibrate,
        _slot_position,
    )

    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["sample"] = {
        "enabled": True,
        "seed": 3,
        "cell_type": "none",
        "noise_sigma": 0.0,
        "noise_fraction": 0.0,
        "coincidence_offset": 0.0,
        "grids_from_holder": True,
        "state_file": str(path),
    }
    microscope._connection.setup_sample_scene()
    _calibrate(microscope, [(-1.5e-3, "Grid-A"), (1.5e-3, "Grid-B")])
    microscope.move_stage_absolute(_slot_position(microscope, -1.5e-3))
    return microscope


def test_grids_on_the_holder_keep_their_trenches(tmp_path):
    """The holder's grids are laid out at the first image, which clears what was
    milled: a saved sample on those same grids is put back after."""
    path = tmp_path / "sample.json"
    first = _holder_session(path)
    try:
        _image(first)
        _mill(first)
        assert len(first._connection.sample_scene.milled) == 2
    finally:
        first.disconnect()
    second = _holder_session(path)
    try:
        scene = second._connection.sample_scene
        assert len(scene.milled) == 2  # restored at connect
        _image(second)  # the grids are placed here
        assert len(scene.grids) == 2
        assert len(scene.milled) == 2
    finally:
        second.disconnect()
