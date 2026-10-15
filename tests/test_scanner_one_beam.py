"""One scanner bound to both beams scans for one at a time.

The binding is allowed: a unit may switch columns. A beam's live view holds the
scanner (``Resources.hold``) for its whole session, and the other beam's ``acquire``
and ``start_live`` are refused at once with ``ResourceBusy``, naming where it is
live, rather than interleaving frames. The live beam's own acquire goes on.
"""

import time

import pytest

from fibsem import utils
from fibsem.devices import ResourceBusy, Resources
from fibsem.drivers.demo.devices import DemoScanGenerator
from fibsem.structures import BeamType


@pytest.fixture
def microscope(monkeypatch):
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(setup_logging=False, manufacturer="Demo")
    scanner = DemoScanGenerator("unit", resources=Resources())
    for beam in microscope.beams.values():
        beam.fill_roles(scanner=scanner)
    yield microscope
    for beam in microscope.beams.values():
        beam.stop_live()
    microscope.disconnect()


def _until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_a_resource_held_by_one_is_refused_to_another():
    resources = Resources()
    resources.hold("scanner", "electron")
    resources.hold("scanner", "electron")  # again, by the same holder: fine
    with pytest.raises(ResourceBusy) as refused:
        resources.hold("scanner", "ion")
    assert (refused.value.resource, refused.value.holder) == ("scanner", "electron")
    resources.release("scanner", "ion")  # not the holder: nothing changes
    assert resources.holder("scanner") == "electron"
    resources.release("scanner", "electron")
    assert resources.holder("scanner") is None


def test_the_other_beam_is_refused_while_one_is_live(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    fib = microscope.beams[BeamType.ION]
    sem.start_live()
    _until(lambda: sem.scanner.sim_frames > 0)

    with pytest.raises(ResourceBusy, match="unit is live on electron"):
        fib.acquire()
    with pytest.raises(ResourceBusy, match="unit is live on electron"):
        fib.start_live()
    assert not fib.is_live
    assert sem.acquire().data is not None  # the live beam's own image goes on

    sem.stop_live()
    _until(lambda: not sem.is_live)
    assert fib.acquire().data is not None


def test_a_beam_without_a_live_view_doesnt_hold_the_scanner(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    fib = microscope.beams[BeamType.ION]
    sem.acquire()
    assert sem.scanner.resources.holder("scanner") is None
    assert fib.acquire().data is not None
