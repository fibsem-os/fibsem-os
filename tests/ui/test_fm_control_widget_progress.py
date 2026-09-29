"""What `FMControlWidget` renders for each fluorescence progress report (FIB-401).

This decode path has never had a test. Its only existing test file covers teardown, and
nothing in `tests/` asserted a rendered label from it -- every check during steps 1 and 2
of this issue was a scratch harness. Typing the payload is the moment to fix that, since
the branch structure is now legible enough to enumerate.

Deliberately about the *rendered text*, not about which branch ran. The bugs this path
has actually had were all wording ones -- a sweep with no channel rendering
"Acquiring  (1/1)...", a focus sweep labelling its objective positions "Z-level", a
second pass looking like the first bar starting over -- and none of them would have been
caught by asserting on control flow.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.fm.progress import (  # noqa: E402
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget  # noqa: E402


class _Host:
    """Only the three widgets the decode path touches.

    The whole `FMControlWidget` needs a microscope and builds a canvas; the decode is a
    method that reads a report and writes to a label and a bar, so it is borrowed onto a
    host carrying real ones.
    """

    _on_acquisition_progress = FMControlWidget._on_acquisition_progress

    def __init__(self):
        from PyQt5.QtWidgets import QLabel, QProgressBar

        self.progressBar_current_acquisition = QProgressBar()
        self.progressText = QLabel()


@pytest.fixture
def host(qapp):
    h = _Host()
    yield h
    h.progressBar_current_acquisition.deleteLater()
    h.progressText.deleteLater()


def _report(status, **fields):
    return FluorescenceAcquisitionProgress(status=status, **fields)


class TestAChannelAcquisition:
    def test_it_names_the_channel_and_counts_them(self, host):
        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_CHANNELS,
                channel="GFP",
                channel_index=2,
                total_channels=3,
            )
        )

        assert host.progressText.text() == "Acquiring GFP (2/3)..."


class TestAZStack:
    def test_the_bar_counts_planes(self, host):
        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
                channel="DAPI",
                channel_index=1,
                total_channels=1,
                zlevel=3,
                total_zlevels=7,
            )
        )

        assert host.progressBar_current_acquisition.format() == "Z-level 3/7"
        assert host.progressBar_current_acquisition.value() == int(3 / 7 * 100)


class TestAFocusSweep:
    def test_it_is_not_labelled_as_a_z_stack(self, host):
        """A sweep steps the objective through a search range. Calling those positions
        "Z-level" names the z-stack, which is not running."""
        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
                channel="DAPI",
                zlevel=3,
                total_zlevels=7,
            )
        )

        assert host.progressBar_current_acquisition.format() == "Focus 3/7"
        assert "Z-level" not in host.progressBar_current_acquisition.format()
        assert host.progressText.text() == "Focusing on DAPI..."

    def test_a_multi_pass_sweep_says_which_pass(self, host):
        """Otherwise a coarse sweep followed by a fine one looks like the same bar
        inexplicably starting over."""
        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
                channel="DAPI",
                zlevel=3,
                total_zlevels=7,
                pass_index=2,
                total_passes=2,
            )
        )

        assert host.progressBar_current_acquisition.format() == "Focus 3/7 · pass 2/2"

    def test_a_single_pass_sweep_does_not(self, host):
        """ "pass 1/1" is noise on the common case."""
        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
                channel="DAPI",
                zlevel=3,
                total_zlevels=7,
                pass_index=1,
                total_passes=1,
            )
        )

        assert host.progressBar_current_acquisition.format() == "Focus 3/7"

    def test_a_sweep_with_no_channel_does_not_render_an_empty_name(self, host):
        """The regression this branch ordering exists for: handled before the channel
        branch, a nameless sweep says "Focusing..." rather than rendering
        "Acquiring  (1/1)..." or leaving the previous message sitting there."""
        host.progressText.setText("Acquiring DAPI (1/2)...")

        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
                channel="",
                zlevel=1,
                total_zlevels=5,
            )
        )

        assert host.progressText.text() == "Focusing..."


class TestTheEndOfAnAcquisition:
    def test_finished_hides_the_bar(self, host):
        host.progressBar_current_acquisition.show()
        host.progressText.show()

        host._on_acquisition_progress(_report(FluorescenceAcquisitionStatus.FINISHED))

        assert not host.progressBar_current_acquisition.isVisible()
        assert not host.progressText.isVisible()

    def test_finished_returns_before_drawing_a_count(self, host):
        """It carries none, so anything drawn from it would be drawn from defaults."""
        host._on_acquisition_progress(
            _report(
                FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
                zlevel=2,
                total_zlevels=4,
            )
        )
        before = host.progressBar_current_acquisition.format()

        host._on_acquisition_progress(_report(FluorescenceAcquisitionStatus.FINISHED))

        assert host.progressBar_current_acquisition.format() == before
