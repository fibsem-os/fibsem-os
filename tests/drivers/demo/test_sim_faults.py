"""Faults a sample is set to have: each fires once, on the event it was set
for, and is logged."""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.sim_faults import Faults, SimulatedFault
from fibsem.drivers.demo.sim_scene import SampleScene
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
)

CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml")


def _microscope(*faults):
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    microscope.system.sim["sample"] = {
        "enabled": True,
        "seed": 3,
        "charging": 0.0,
        "beam_optics": False,
        "faults": list(faults),
    }
    microscope._connection.setup_sample_scene()
    return microscope


def _image(microscope, beam_type=BeamType.ELECTRON) -> np.ndarray:
    settings = ImageSettings(
        resolution=(256, 170),
        hfw=60e-6,
        dwell_time=1e-9,
        save=False,
        beam_type=beam_type,
    )
    return microscope.acquire_image(image_settings=settings).data.astype(np.float32)


def _mill(microscope):
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=10e-6, height=4e-6, depth=10e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()


def test_a_fault_fires_once_on_its_count_and_is_logged():
    faults = Faults([{"kind": "blank_image", "at": 2, "beam": "ion"}])
    assert faults.due("image", "ion") == []
    assert faults.due("image", "electron") == []
    assert [f.kind for f in faults.due("image", "ion")] == ["blank_image"]
    assert faults.due("image", "ion") == []
    assert [(e["kind"], e["at"], e["beam"]) for e in faults.log] == [
        ("blank_image", 2, "ion")
    ]


def test_a_count_given_as_a_pair_is_drawn_from_the_seed():
    entries = [{"kind": "stage_fails", "at": [3, 9]}]
    ats = {Faults(entries, seed).faults[0].at for seed in range(20)}
    assert ats <= set(range(3, 10)) and len(ats) > 2
    assert Faults(entries, 5).faults[0].at == Faults(entries, 5).faults[0].at


@pytest.mark.parametrize(
    "entry",
    [
        {"kind": "smoke"},
        {"kind": "charging", "on": "mill"},
        {"kind": "beam_off", "x": 1},
    ],
)
def test_a_fault_the_simulator_does_not_have_is_refused(entry):
    with pytest.raises(ValueError):
        SampleScene.from_config({"faults": [entry]})


def test_the_beam_turns_off_until_it_is_turned_on_again():
    microscope = _microscope({"kind": "beam_off", "at": 2})
    try:
        first = _image(microscope)
        assert _image(microscope).mean() < 0.2 * first.mean()
        assert not microscope.is_on(BeamType.ELECTRON)
        assert _image(microscope).mean() < 0.2 * first.mean()
        microscope.turn_on(BeamType.ELECTRON)
        assert _image(microscope).mean() > 0.8 * first.mean()
        fired = microscope._connection.sample_scene.record()["faults"]
        assert [f["kind"] for f in fired] == ["beam_off"]
    finally:
        microscope.disconnect()


def test_a_blank_and_a_noisy_frame_are_one_frame_each():
    microscope = _microscope(
        {"kind": "blank_image", "at": 2}, {"kind": "noisy_image", "at": 3}
    )
    try:
        first = _image(microscope)
        assert _image(microscope).mean() < 0.2 * first.mean()
        noisy = _image(microscope)
        after = _image(microscope)
        assert (
            np.abs(np.diff(noisy, axis=1)).mean()
            > 2 * np.abs(np.diff(after, axis=1)).mean()
        )
        assert after.mean() == pytest.approx(first.mean(), rel=0.1)
    finally:
        microscope.disconnect()


def test_the_current_drops_and_stays_dropped():
    microscope = _microscope({"kind": "current_drop", "beam": "ion", "amount": 0.25})
    try:
        before = microscope.get_beam_current(BeamType.ION)
        _image(microscope, BeamType.ION)
        assert microscope.get_beam_current(BeamType.ION) == pytest.approx(0.25 * before)
    finally:
        microscope.disconnect()


def test_a_pass_drifts_more_and_one_with_no_beam_mills_nothing():
    microscope = _microscope(
        {"kind": "mill_drift", "at": 1, "amount": 1e-6},
        {"kind": "beam_off", "on": "mill", "at": 2},
    )
    try:
        scene = microscope._connection.sample_scene
        _mill(microscope)
        assert scene.mill_passes[0]["drift"][1] == pytest.approx(1e-6, abs=0.2e-6)
        _mill(microscope)
        assert len(scene.mill_passes) == 1 and len(scene.milled) == 1
    finally:
        microscope.disconnect()


def test_a_stage_move_fails_where_it_was_and_another_lands_off():
    microscope = _microscope(
        {"kind": "stage_fails", "at": 1}, {"kind": "stage_off", "at": 2, "amount": 4e-6}
    )
    try:
        start = microscope.get_stage_position()
        with pytest.raises(SimulatedFault):
            microscope.move_stage_relative(FibsemStagePosition(x=10e-6))
        assert microscope.get_stage_position().x == pytest.approx(start.x)
        microscope.move_stage_relative(FibsemStagePosition(x=10e-6))
        assert microscope.get_stage_position().x == pytest.approx(start.x + 10e-6)
        actual = microscope._connection.scene_stage_position()
        assert actual.x == pytest.approx(start.x + 14e-6)
    finally:
        microscope.disconnect()


def test_a_valve_that_does_not_open_lays_nothing_down():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    try:
        scene = microscope._connection.sample_scene
        scene.faults = [{"kind": "gis_closed"}]
        gis = microscope.devices["gis"]
        microscope.coating.deposit(2)
        assert scene.gis_thickness == 0
        assert [f["kind"] for f in scene.faults_fired()] == ["gis_closed"]
        microscope.coating.deposit(2)
        assert scene.gis_thickness > 0
        assert not gis.opened.get_value()
    finally:
        microscope.disconnect()
