"""A correlation tab hands back what it put on its session when it is destroyed.

The session can outlive the window that shows it (``AppSession.correlation``).
A window closed while its run was in flight left the session running forever,
so nothing could run again; and its removal policy, which asks the user before
a fiducial's partner goes, stayed installed and was called on a deleted widget
by the next removal from a script.

Real worker thread, real fit, offscreen.
"""

import os

from tests.fixtures.correlation import fiducial_pairs, fm_image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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
from fibsem.structures.image import FibsemImage


@pytest.fixture
def session():
    session = CorrelationSession()
    session.fib_image = FibsemImage.generate_blank_image(resolution=(400, 300))
    session.fm_image = fm_image()
    fib, fm = fiducial_pairs()
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
