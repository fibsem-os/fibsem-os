"""A correlation tab hands back what it put on its session when it is destroyed.

The session can outlive the window that shows it (``AppSession.correlation``).
A window closed while its run was in flight left the session running forever,
so nothing could run again; and its removal policy, which asks the user before
a fiducial's partner goes, stayed installed and was called on a deleted widget
by the next removal from a script.

Real worker thread, real fit, offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5 import sip

from fibsem.correlation.session import CorrelationSession
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.fm.structures import (
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage


def _fm_image():
    meta = FluorescenceImageMetadata(
        acquisition_date=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=300e-9,
        resolution=(200, 200),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                excitation_wavelength=488.0,
                emission_filter=EmissionFilter("520 nm", low=520.0),
                power=0.3,
                exposure_time=0.05,
                gain=1.5,
                offset=50.0,
            )
        ],
    )
    return FluorescenceImage(data=np.zeros((1, 20, 200, 200), np.uint16), metadata=meta)


def _pairs():
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


@pytest.fixture
def session():
    session = CorrelationSession()
    session.fib_image = FibsemImage.generate_blank_image(resolution=(400, 300))
    session.fm_image = _fm_image()
    fib, fm = _pairs()
    session.set_data(
        CorrelationInputData(
            fib_coordinates=fib,
            fm_coordinates=fm,
            poi_coordinates=[Coordinate(PointXYZ(90, 90, 8), PointType.POI)],
        )
    )
    return session


def _tab(session):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget(session=session)


def test_a_tab_destroyed_mid_run_lets_the_session_run_again(qapp, session):
    tab = _tab(session)
    tab._run()
    worker = tab._worker
    assert session.running

    sip.delete(tab)  # the window goes before the run is delivered

    assert not session.running
    assert session.can_run()
    worker.wait()
    qapp.processEvents()
    assert session.result is None  # nobody was there to adopt it
    assert session.run() is session.result  # and the next run starts


def test_a_destroyed_tab_no_longer_decides_what_a_removal_removes(qapp, session):
    tab = _tab(session)
    fib = session.points.of_type(PointType.FIB)[0]
    asked = []
    tab._ask_pair_removal = lambda coord, partner: asked.append(coord) or ""
    assert not session.points.remove(fib)  # the tab asks, and is cancelled
    assert asked == [fib]

    sip.delete(tab)
    removed = session.points.remove(fib)

    assert removed  # the default: what is asked for, without asking
    assert asked == [fib]
    assert len(session.points.of_type(PointType.FM)) == 5  # the partner stays


def test_a_second_tab_keeps_its_own_policy_when_the_first_goes(qapp, session):
    first = _tab(session)
    second = _tab(session)
    asked = []
    second._ask_pair_removal = lambda coord, partner: asked.append(coord) or ""

    sip.delete(first)
    fib = session.points.of_type(PointType.FIB)[0]

    assert not session.points.remove(fib)
    assert asked == [fib]
    second.close()
