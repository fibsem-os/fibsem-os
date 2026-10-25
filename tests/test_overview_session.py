"""The overview session: what both overview tabs share, without Qt (FIB-1449)."""

import pytest

from fibsem import utils
from fibsem.imaging.overview_session import (
    BEAM,
    FLUORESCENCE,
    LOCK,
    POSITIONS,
    SELECTION,
    OverviewSession,
)
from fibsem.structures.settings import FibsemStagePosition
from fibsem.workflows.session import AppSession


def test_each_change_says_which_part_changed(microscope):
    session = OverviewSession(microscope)
    seen = []
    session.changed.connect(seen.append)
    session.set_positions([FibsemStagePosition(name="a")], flagged=["a"])
    session.select("a")
    session.set_interactive(False, "the other overview is acquiring")
    assert seen == [POSITIONS, SELECTION, LOCK]
    assert [p.name for p in session.positions] == ["a"]
    assert session.flagged == {"a"} and session.selected == "a"
    assert session.move_refusal() == (
        "Cannot move the stage while the other overview is acquiring."
    )
    session.set_interactive(True)
    session.set_running(True)
    assert session.move_refusal() == "Cannot move the stage during an acquisition."


def test_a_move_reads_back_where_the_stage_stopped(microscope):
    session = OverviewSession(microscope)
    start = microscope.get_stage_position()
    target = FibsemStagePosition(
        x=start.x + 20e-6, y=start.y, z=start.z, r=start.r, t=start.t
    )
    try:
        session.move_to(target)
        assert session.read_stage_position().x == pytest.approx(target.x, abs=1e-7)
    finally:
        microscope.safe_absolute_stage_movement(start)


def test_the_app_session_keeps_one_per_kind_until_the_microscope_changes(microscope):
    app = AppSession()
    with pytest.raises(ValueError):
        app.overview(BEAM)
    app.microscope = microscope
    beam = app.overview(BEAM)
    assert app.overview(BEAM) is beam
    if microscope.fm is None:
        with pytest.raises(ValueError):
            app.overview(FLUORESCENCE)
    other, _ = utils.setup_session(manufacturer="Demo")
    app.microscope = other
    assert app.overview(BEAM) is not beam
    assert app.overview(BEAM).microscope is other
