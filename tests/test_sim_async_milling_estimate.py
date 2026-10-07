"""The simulator's milling estimate while an asynchronous mill runs (FIB-1119).

`start_milling` on the simulator never ends on its own, so while such a mill runs its
estimate adds `SIM_ASYNC_MILLING_EXTRA_TIME`. That used to be an `isinstance(Demo)`
override inside the coincidence milling strategy, which is timed by the estimate.
"""

import pytest

from fibsem import utils
from fibsem.microscopes import simulator
from fibsem.structures import MillingState


@pytest.fixture()
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def _pattern_estimate(microscope) -> float:
    return microscope.estimate_milling_time()


def test_an_asynchronous_mill_reports_the_extra_time(microscope):
    idle = _pattern_estimate(microscope)
    microscope.start_milling()
    assert (
        microscope.estimate_milling_time()
        == idle + simulator.SIM_ASYNC_MILLING_EXTRA_TIME
    )


def test_stopping_drops_the_extra_time(microscope):
    idle = _pattern_estimate(microscope)
    microscope.start_milling()
    microscope.stop_milling()
    assert microscope.get_milling_state() is MillingState.IDLE
    assert microscope.estimate_milling_time() == idle


def test_a_synchronous_mill_is_timed_by_patterns_alone(microscope):
    # a stale asynchronous flag must not stretch a later timed mill
    microscope.start_milling()
    microscope.run_milling(milling_current=1e-9, milling_voltage=30e3)
    assert microscope.milling.progress.cached.total < (
        simulator.SIM_ASYNC_MILLING_EXTRA_TIME
    )
