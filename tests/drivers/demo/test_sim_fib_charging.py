"""FIB charging in the simulator: charged parts darken and the scan lines
slip, differently every frame."""

import numpy as np

from fibsem.drivers.demo.sim.scene import SampleScene


def _disc(shape=(342, 512), radius=40):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2 < radius**2).astype(
        np.float32
    )


def test_a_charged_part_goes_dark_and_each_frame_differs():
    charge = _disc()
    flat = np.full(charge.shape, 150.0, dtype=np.float32)
    one = SampleScene.ion_charge_up(flat, charge, 1.0, np.random.default_rng(1))
    two = SampleScene.ion_charge_up(flat, charge, 1.0, np.random.default_rng(2))
    inside, outside = charge > 0.5, charge == 0
    assert one[inside].mean() < 0.8 * one[outside].mean()
    assert np.abs(one - two)[inside].mean() > 1.0


def test_the_scan_lines_slip_over_a_charged_part():
    charge = _disc()
    rows = np.tile(np.arange(512, dtype=np.float32), (342, 1))
    out = SampleScene.ion_charge_up(rows, charge, 1.0, np.random.default_rng(3))
    # an image that is its own column number shows each row's slip
    slip = (rows - out)[:, 100:400].mean(axis=1)
    assert np.ptp(slip) > 1.0


def test_no_charging_leaves_the_image_alone():
    flat = np.full((64, 64), 150.0, dtype=np.float32)
    out = SampleScene.ion_charge_up(flat, _disc((64, 64), 10), 0.0, None)
    assert out is flat
    assert SampleScene().ion_charge_strength(1e-6) == 0.0
    assert SampleScene.from_config({}).ion_charge_strength(1e-6) > 0.0
