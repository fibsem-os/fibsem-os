"""Correlation inputs and canvas clicks shared by the correlation tests."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.fm.structures import (
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


def fm_image() -> FluorescenceImage:
    """A blank 20-slice, one-channel FM stack."""
    meta = FluorescenceImageMetadata(
        acquisition_datetime=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=300e-9,
        resolution=(200, 200),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                excitation_wavelength=488.0,
                emission_filter=EmissionFilter.band("520 nm", 520.0),
                power=0.3,
                exposure_time=0.05,
                gain=1.5,
                offset=50.0,
            )
        ],
    )
    return FluorescenceImage(data=np.zeros((1, 20, 200, 200), np.uint16), metadata=meta)


def fiducial_pairs():
    """FM points and their FIB images under a rotation and scale, with click noise."""
    fm = np.array(
        [[40, 50, 4], [150, 40, 9], [160, 150, 14], [50, 160, 6], [100, 100, 11]],
        float,
    )
    a = np.deg2rad(20)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fib = (rot @ fm[:, :2].T).T * 1.5 + [60.0, 30.0]
    fib += np.random.default_rng(0).normal(0, 0.5, fib.shape)
    return (
        [Coordinate(PointXYZ(x, y, 0.0), PointType.FIB) for x, y in fib],
        [Coordinate(PointXYZ(x, y, z), PointType.FM) for x, y, z in fm],
    )


def press_release(overlay, x, y, *, drag_to=None):
    """Drive a real press → (optional motion) → release on the overlay."""
    ax = overlay._ax

    def _ev(px, py):
        sx, sy = ax.transData.transform((px, py))
        return SimpleNamespace(
            inaxes=ax, button=1, x=sx, y=sy, xdata=px, ydata=py, dblclick=False
        )

    overlay._on_press(_ev(x, y))
    if drag_to is not None:
        overlay._on_motion(_ev(*drag_to))
    overlay._on_release(_ev(*(drag_to or (x, y))))
