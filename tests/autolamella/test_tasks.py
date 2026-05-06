from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTask,
    MillTrenchTaskConfig,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition


def _make_trench_task(microscope: FibsemMicroscope, tmp_path: Path) -> MillTrenchTask:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    config = MillTrenchTaskConfig()
    return MillTrenchTask(microscope=microscope, config=config, lamella=lamella)


@pytest.fixture
def compustage_microscope() -> FibsemMicroscope:
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope._update_orientations()
    return microscope


def test_get_stage_position_for_orientation_none_returns_same_position(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """When orientation is None the stage position is returned unchanged."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    pos = FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, r=np.radians(0), t=np.radians(-23))
    result = task._get_stage_position_for_orientation(pos, None)
    assert result is pos


def test_get_stage_position_for_orientation_delegates_to_get_target_position(
    compustage_microscope: FibsemMicroscope, tmp_path: Path
) -> None:
    """When orientation is set the result comes from get_target_position."""
    task = _make_trench_task(compustage_microscope, tmp_path)
    pos = FibsemStagePosition(r=np.radians(0), t=np.radians(-23))
    result = task._get_stage_position_for_orientation(pos, "SEM")
    sem = compustage_microscope.get_orientation("SEM")
    assert sem.t is not None and result.t is not None
    assert np.isclose(result.t, sem.t, atol=1e-6)


@pytest.mark.parametrize("orientation", ["SEM", "FIB", "MILLING", None])
def test_mill_trench_task_config_orientation_accepts_none_and_strings(
    orientation: Optional[Literal["SEM", "FIB", "MILLING"]],
) -> None:
    """MillTrenchTaskConfig accepts all valid orientation values including None."""
    config = MillTrenchTaskConfig(orientation=orientation)
    assert config.orientation == orientation
