"""A scanner's own limits, and what a beam bound to it offers.

A scanner says what its unit can do with ``_dwell_time_limits`` and
``_resolution_limits``; ``acquire`` refuses a frame outside them. While it is bound to
a beam, the beam's ``dwell_time`` limits are the overlap of the column's and the
scanner's, and its ``resolution`` choices are the ones the scanner can scan. The values
stay the beam's.
"""

import logging

import pytest

from fibsem import utils
from fibsem.drivers.demo.devices import DemoScanGenerator
from fibsem.structures import BeamType, RangeLimit

DWELL = RangeLimit(min=100e-9, max=10e-6)
RESOLUTIONS = ((256, 256), (2048, 2048))


class LimitedScanner(DemoScanGenerator):
    dwell = DWELL
    frames = RESOLUTIONS

    def _dwell_time_limits(self):
        return self.dwell

    def _resolution_limits(self):
        return self.frames


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(setup_logging=False, manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def test_a_scanner_without_limits_takes_any_positive_frame():
    scanner = DemoScanGenerator("plain")
    assert scanner.dwell_time_limits is None
    assert scanner.resolution_limits is None
    assert scanner.acquire((3, 2), 1e-12).shape == (2, 3)


@pytest.mark.parametrize(
    "resolution, dwell_time, message",
    [
        ((512, 512), 10e-9, "dwell time"),
        ((512, 512), 1e-3, "dwell time"),
        ((128, 512), 1e-6, "resolution 128x512"),
        ((4096, 1024), 1e-6, "resolution 4096x1024"),
    ],
)
def test_a_scanner_refuses_a_frame_outside_its_limits(resolution, dwell_time, message):
    scanner = LimitedScanner("limited")
    with pytest.raises(ValueError, match=message):
        scanner.acquire(resolution, dwell_time)
    assert scanner.sim_frames == 0


def test_a_scanner_scans_a_frame_within_its_limits():
    scanner = LimitedScanner("limited")
    assert scanner.acquire((256, 2048), 100e-9).shape == (2048, 256)


def test_a_bound_beam_offers_only_what_both_can_do(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    own_dwell = sem.dwell_time.limits
    own_resolutions = list(sem.resolution.choices)
    changed = []
    sem.metadata_changed.connect(lambda name, metadata: changed.append(name))

    sem.fill_roles(scanner=LimitedScanner("limited"))

    assert sem.dwell_time.limits == RangeLimit(
        min=max(own_dwell.min, DWELL.min), max=min(own_dwell.max, DWELL.max)
    )
    expected = [
        r for r in own_resolutions if 256 <= r[0] <= 2048 and 256 <= r[1] <= 2048
    ]
    assert expected and len(expected) < len(own_resolutions)
    assert [tuple(r) for r in sem.resolution.choices] == [tuple(r) for r in expected]
    assert {"dwell_time", "resolution"} <= set(changed)
    # the other beam, with no scanner, keeps its own
    fib = microscope.beams[BeamType.ION]
    assert list(fib.resolution.choices) == own_resolutions


def test_limits_that_dont_overlap_offer_the_scanners(microscope, caplog):
    sem = microscope.beams[BeamType.ELECTRON]
    own = sem.dwell_time.limits
    far = RangeLimit(min=own.max * 10, max=own.max * 100)
    scanner = LimitedScanner("far")
    scanner.dwell = far
    logging.getLogger().addHandler(caplog.handler)

    sem.fill_roles(scanner=scanner)

    assert sem.dwell_time.limits == far
    assert "don't overlap" in caplog.text
