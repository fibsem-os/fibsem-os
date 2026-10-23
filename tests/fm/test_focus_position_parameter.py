"""The objective's saved focus position is a device parameter: the FM tab and
the agent save and read the same one (Patrick, 2026-10-23)."""

import pytest

from fibsem.drivers.demo.devices import bind_demo_fm
from fibsem.fm.microscope import ObjectiveLens


@pytest.fixture
def objective():
    return bind_demo_fm()["objective"]


def test_the_demo_starts_with_a_saved_focus_that_can_be_cleared(objective):
    from fibsem.drivers.demo.simulator import SIM_OBJECTIVE_FOCUS_POSITION

    assert objective.focus_position.get_value() == SIM_OBJECTIVE_FOCUS_POSITION
    assert objective.focus_position.settable
    objective.focus_position.write_through(None)
    assert objective.focus_position.get_value() is None


def test_save_focus_position_saves_where_the_objective_is(objective):
    assert objective.save_focus_position() == objective.position.get_value()
    assert objective.focus_position.get_value() == objective.position.get_value()


def test_the_fm_api_reads_and_hears_the_device_parameter(objective):
    lens = ObjectiveLens(objective)
    heard = []
    lens.focus_position_changed.connect(heard.append)
    objective.focus_position.set_value(1.5e-3)  # as the App Server writes it
    assert lens.focus_position == pytest.approx(1.5e-3)
    lens.focus_position = None
    assert objective.focus_position.get_value() is None
    assert heard == [pytest.approx(1.5e-3), None]
