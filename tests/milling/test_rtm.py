"""RTM as a milling monitor (FIB-1178): frames from the milling service, read
by the RTM monitor while a step mills, simulated on the Demo."""

import numpy as np
import pytest

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.monitors import RtmMonitor, get_monitors
from fibsem.milling.monitors.rtm import RtmConfig
from fibsem.milling.rtm import RtmFrame, is_monitored
from fibsem.structures import CrossSectionPattern, FibsemRectangleSettings
from tests.milling.test_fm_intensity_and_sem_peek import (  # noqa: F401
    _stage,
    milling_in_real_time,
)
from tests.test_milling_run import (  # noqa: F401
    RUNNING,
    microscope,
    scripted,
)


def _rtm(**kwargs):
    return RtmConfig(**kwargs).made_for(RtmMonitor)


def _rectangle(**kwargs):
    return FibsemRectangleSettings(
        width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0, **kwargs
    )


def test_a_frame_puts_each_value_at_its_position():
    frame = RtmFrame(
        sequence=1,
        timestamp=0.0,
        interval=1.0,
        resolution=(4, 3),
        values={0: np.array([1.0, 3.0]), 2: np.array([5.0])},
        positions={0: np.array([[0, 0], [3, 2]]), 2: np.array([[1, 1]])},
    )

    image = frame.to_image()

    assert image.shape == (3, 4)
    assert (image[0, 0], image[2, 3], image[1, 1]) == (1.0, 3.0, 5.0)
    assert np.isnan(image).sum() == 9
    assert frame.means() == {0: 2.0, 2: 5.0}


def test_cleaning_cross_sections_and_exclusions_are_not_read():
    assert is_monitored(_rectangle())
    assert not is_monitored(
        _rectangle(cross_section=CrossSectionPattern.CleaningCrossSection)
    )
    assert not is_monitored(_rectangle(is_exclusion=True))


def test_rtm_reads_the_patterns_as_they_mill_on_the_demo(
    tmp_path, milling_in_real_time
):
    demo = milling_in_real_time()
    frames = []
    demo.milling.rtm_frame.connect(frames.append)

    answer = demo.milling.run_stage(_stage(_rtm(interval=0.1)), path=str(tmp_path))

    assert len(frames) >= 3
    assert [f.sequence for f in frames] == list(range(1, len(frames) + 1))
    width, height = frames[0].resolution
    for frame in frames:
        assert frame.values
        for index, positions in frame.positions.items():
            assert positions.shape == (frame.values[index].size, 2)
            assert ((positions >= 0) & (positions < (width, height))).all()
    first, last = frames[0].means(), frames[-1].means()
    assert all(last[i] < first[i] for i in first), "values fall as it mills"
    assert demo.milling.last_rtm_frame is frames[-1]

    (line,) = answer["stages"]
    (step,) = line["steps"]
    (record,) = step["monitors"]
    assert record["name"] == "RTM"
    assert record["facts"]["Reads"] == len(frames)
    assert "error" not in record


def test_high_mode_reads_more_points_than_low(microscope):
    milling = microscope.milling
    milling._set_drawn([_rectangle()])
    sizes = {}
    for mode in ("low", "high"):
        milling.rtm_start(mode)
        sizes[mode] = milling.rtm_read().values[0].size
        milling.rtm_stop()
    assert sizes["high"] > sizes["low"]
    assert milling.rtm_read() is None, "nothing is read once RTM stops"


def test_a_mode_the_driver_lacks_is_refused(microscope):
    with pytest.raises(ValueError, match="no RTM in 'ultra'"):
        microscope.milling.rtm_start("ultra")


def test_rtm_is_offered_only_where_the_driver_reads_it(microscope, scripted):
    assert "RTM" in microscope.milling.list_options()["monitors"]

    service = scripted([RUNNING, RUNNING])
    assert not service.has_rtm
    assert "RTM" not in service.list_options()["monitors"]
    assert "RTM" in get_monitors(), "still known, so a saved stage keeps it"

    with service.stage_run(_stage(_rtm())) as stage_run:
        status = service.run()
    (step,) = stage_run.steps
    assert status == "completed", "the step mills without it"
    (record,) = step.to_dict()["monitors"]
    assert record["error"] == "RuntimeError: This instrument has no RTM."


def test_a_stage_saves_its_rtm_monitor():
    stage = _stage(_rtm(mode="high", interval=0.5))
    loaded = FibsemMillingStage.from_dict(stage.to_dict())
    assert loaded.monitors == stage.monitors
    assert stage.to_dict()["monitors"][0]["name"] == "RTM"
