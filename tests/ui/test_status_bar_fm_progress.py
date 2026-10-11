"""A fluorescence acquisition's progress in the status bar's line (FIB-397, FIB-1188).

It reported only into the Fluorescence tab, so a fluorescence task left the main
window's status bar idle and the app looking hung. The signal lives on the optional
fluorescence device rather than the microscope, most of its routines never say they
have finished, and an FM overview's tiles report through it as well as through the
tiler -- each of which is one test here.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.fm.progress import (  # noqa: E402
    FluorescenceAcquisitionProgress as Report,
)
from fibsem.fm.progress import (  # noqa: E402
    FluorescenceAcquisitionStatus as Status,
)
from fibsem.imaging.tiling.progress import (  # noqa: E402
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.ui.widgets import status_bar  # noqa: E402

INSTRUCTION = "Create or load an experiment to begin."


@pytest.fixture
def bar(qapp, monkeypatch):
    monkeypatch.setattr(status_bar, "OUTCOME_MS", 20)
    monkeypatch.setattr(status_bar, "FM_STALE_MS", 40)
    bar = status_bar.FibsemStatusBar()
    bar.set_instruction(INSTRUCTION)
    yield bar
    bar.deleteLater()


def _wait(ms=120):
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def test_a_z_stack_counts_its_planes(bar):
    bar._on_fm_progress(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=12, total_zlevels=21)
    )
    assert bar.text == "Fluorescence DAPI · plane 12 of 21 57%"
    assert bar.fraction == pytest.approx(12 / 21, abs=1e-3)


def test_channels_count_the_channels(bar):
    bar._on_fm_progress(
        Report(
            Status.ACQUIRING_CHANNELS, channel="GFP", channel_index=2, total_channels=3
        )
    )
    assert bar.text == "Fluorescence GFP · channel 2 of 3 66%"


def test_autofocus_counts_its_steps_and_passes(bar):
    bar._on_fm_progress(
        Report(
            Status.ACQUIRING_AUTOFOCUS,
            channel="DAPI",
            zlevel=4,
            total_zlevels=15,
            pass_index=1,
            total_passes=2,
        )
    )
    assert bar.text == "Fluorescence autofocus · DAPI · pass 1 of 2 · step 4 of 15 26%"


def test_finished_says_done_then_gives_the_line_back(bar):
    bar._on_fm_progress(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=21, total_zlevels=21)
    )
    bar._on_fm_progress(Report(Status.FINISHED))
    assert bar.text == "Fluorescence done"
    _wait()
    assert bar.text == INSTRUCTION


def test_a_routine_that_never_finishes_does_not_linger(bar):
    """The autofocus sweep and the device's own z-stack stop without a FINISHED: the
    line goes once the reports stop (FIB-374's failure, on this signal)."""
    bar._on_fm_progress(
        Report(Status.ACQUIRING_AUTOFOCUS, channel="DAPI", zlevel=15, total_zlevels=15)
    )
    assert bar.showing == "progress"
    _wait()
    assert bar.text == INSTRUCTION


def test_reports_keep_the_line_alive(bar):
    for z in range(1, 6):
        bar._on_fm_progress(
            Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=z, total_zlevels=21)
        )
        _wait(20)  # each well inside the expiry
    assert "plane 5 of 21" in bar.text


def test_the_expiry_does_not_take_someone_elses_line(bar):
    bar._on_fm_progress(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=1, total_zlevels=21)
    )
    bar.set_progress("Milling: Rough Mill", "stage 1 of 2")
    _wait()
    assert bar.text == "Milling: Rough Mill stage 1 of 2"


def test_a_stray_finished_says_nothing(bar):
    bar._on_fm_progress(Report(Status.FINISHED))
    assert bar.text == INSTRUCTION


def test_an_fm_overview_keeps_its_tile_line(bar):
    """The tiles are the line; each tile's channels and planes would take turns
    with it."""
    bar._on_tiled_progress(
        TiledProgress(
            TiledStatus.TILE_COLLECTED,
            modality=MODALITY_FLUORESCENCE,
            completed=3,
            total=9,
        )
    )
    tiles = bar.text
    bar._on_fm_progress(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=4, total_zlevels=21)
    )
    assert bar.text == tiles
    bar._on_tiled_progress(
        TiledProgress(TiledStatus.FINISHED, modality=MODALITY_FLUORESCENCE)
    )
    _wait()
    bar._on_fm_progress(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=4, total_zlevels=21)
    )
    assert bar.text.startswith("Fluorescence DAPI"), "after the run, FM speaks again"


# --- a real microscope, with and without the device ---------------------------------


def _microscope(config: str):
    import fibsem.config as cfg
    from fibsem import utils

    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, config), setup_logging=False
    )
    return microscope


def test_the_bar_listens_to_the_fluorescence_device(bar):
    """Through the device's own signal, which `set_microscope` subscribes to when the
    system has one: the Arctis simulator does."""
    microscope = _microscope("sim-arctis-configuration.yaml")
    assert microscope.fm is not None
    bar.set_microscope(microscope)
    microscope.fm.acquisition_progress_signal.emit(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=2, total_zlevels=4)
    )
    assert bar.text == "Fluorescence DAPI · plane 2 of 4 50%"

    bar.set_microscope(None)
    microscope.fm.acquisition_progress_signal.emit(
        Report(Status.ACQUIRING_ZSTACK, channel="DAPI", zlevel=3, total_zlevels=4)
    )
    assert bar.text == INSTRUCTION, "a disconnect lets the device go too"
    microscope.disconnect()


def test_a_microscope_without_fluorescence_connects(bar):
    """The signal is on the optional device: a system without one must not raise."""
    microscope = _microscope("microscope-configuration.yaml")
    microscope.fm = None  # whatever the default configuration builds, take it away
    bar.set_microscope(microscope)
    bar.set_microscope(None)
    microscope.disconnect()
