"""The milling loop, the alignment-area check and the point of interest are the
shared task's: a volume task asks them on its position as a lamella task does."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from fibsem.applications.volume.manager import VolumeTaskManager
from fibsem.applications.volume.tasks import VolumeTask
from fibsem.applications.volume.tasks.milling import slice_milling
from fibsem.structures.settings import FibsemRectangle, Point
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    MILLING,
    POINT_OF_INTEREST,
    DecisionOutcome,
    prepare_values,
)
from fibsem.workflows.task_config import TaskConfig


@dataclass
class _AskingConfig(TaskConfig):
    task_type: ClassVar[str] = "VOLUME_TEST_ASKS"
    display_name: ClassVar[str] = "Asks"
    task_name: str = "Asks"


class _Asks(VolumeTask):
    """Mills once through the shared loop and checks the alignment area."""

    config_cls = _AskingConfig
    asks = (MILLING, ALIGNMENT_AREA)

    def _run(self) -> None:
        self.milled = self.update_milling_config_ui(slice_milling(), msg="Mill.")
        self._validate_alignment_area()


def test_an_automated_volume_task_mills_once_and_records_what_it_used(
    microscope, experiment, position, milled
):
    manager = VolumeTaskManager(microscope=microscope, experiment=experiment)
    task = _Asks(microscope, _AskingConfig(), position, experiment, manager)
    area = position.alignment_area

    task.run()

    assert len(milled) == 1
    kinds = [p.kind for p in position.proposals]
    assert kinds == [MILLING, MILLING, ALIGNMENT_AREA]
    assert all(
        p.current.outcome is DecisionOutcome.Unreviewed for p in position.proposals
    )
    assert position.alignment_area == area


def test_a_point_of_interest_decided_on_a_position_moves_its_slices():
    from fibsem.applications.volume.position import VolumePosition
    from fibsem.applications.volume.tasks.milling import (
        VolumeFIBMillingTaskConfig,
        slice_at,
    )

    position = VolumePosition(name="aspen", poi=Point(0, 1e-6))
    config = VolumeFIBMillingTaskConfig()
    before = slice_at(config, position, 1).stages[0].pattern.point

    synced = prepare_values(
        None, position, POINT_OF_INTEREST, {"poi": Point(2e-6, 3e-6)}
    ).apply()

    after = slice_at(config, position, 1).stages[0].pattern.point
    assert synced == []
    assert (after.x - before.x, after.y - before.y) == pytest.approx((2e-6, 2e-6))


def test_a_position_keeps_its_alignment_area():
    from fibsem.applications.volume.position import VolumePosition

    position = VolumePosition(name="aspen")
    position.alignment_area = FibsemRectangle(0.1, 0.2, 0.3, 0.4)

    back = VolumePosition.from_dict(position.to_dict())

    assert back.alignment_area == position.alignment_area
