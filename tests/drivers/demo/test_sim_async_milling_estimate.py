"""The simulator's milling estimate while an asynchronous mill runs (FIB-1119).

`start_milling` on the simulator never ends on its own, so while such a mill runs its
estimate adds `SIM_ASYNC_MILLING_EXTRA_TIME`. That used to be an `isinstance(Demo)`
override inside the coincidence milling strategy, which is timed by the estimate.
"""

from fibsem.drivers.demo import simulator
from fibsem.structures.settings import MillingState
from tests.fixtures.microscopes import demo_microscope

microscope = demo_microscope(scope="function")


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
    assert microscope.milling.state.get_value() is MillingState.IDLE
    assert microscope.estimate_milling_time() == idle


def test_a_synchronous_mill_is_timed_by_patterns_alone(microscope):
    # a stale asynchronous flag must not stretch a later timed mill
    microscope.start_milling()
    ticks = []
    microscope.milling.progress.changed.connect(
        lambda p: ticks.append(p) if not p.status.is_terminal else None
    )
    microscope.run_milling()
    assert ticks[-1].estimated_time < simulator.SIM_ASYNC_MILLING_EXTRA_TIME
