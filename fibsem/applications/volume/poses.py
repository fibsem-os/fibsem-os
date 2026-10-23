"""A volume position's poses: setting one from the microscope, the stage position
a move to one goes to, and moving the milling pose. The positions panel and the
App Server both call these; each returns why it refused, or "" once done."""

import copy
from typing import Optional, Tuple

from fibsem.applications.volume.position import SEM_POSE, VolumePosition
from fibsem.structures import FibsemStagePosition
from fibsem.workflows.item import MILLING_POSE

__all__ = ["POSES", "move_milling_pose", "pose_stage_position", "set_pose_now"]

POSES = (MILLING_POSE, SEM_POSE)


def _unknown(pose_name: str) -> str:
    if pose_name in POSES:
        return ""
    return f"A volume position's poses are {', '.join(POSES)}, not {pose_name!r}."


def set_pose_now(experiment, microscope, name: str, pose_name: str) -> str:
    """Set ``name``'s pose ``pose_name`` to the microscope state now. Saved."""
    why = _unknown(pose_name)
    if why:
        return why
    return experiment.set_pose(name, pose_name, microscope.get_microscope_state())


def pose_stage_position(
    position: VolumePosition, pose_name: str
) -> Tuple[Optional[FibsemStagePosition], str]:
    """Where a move to ``position``'s pose ``pose_name`` goes, or why it can't."""
    why = _unknown(pose_name)
    if why:
        return None, why
    if MILLING_POSE not in position.poses:
        return None, f"{position.name} has no milling pose."
    return position.pose(pose_name).stage_position, ""


def move_milling_pose(
    experiment, name: str, stage_position: FibsemStagePosition
) -> str:
    """Move ``name``'s milling pose to ``stage_position``, the rest of its state
    kept. Refused once a slice is cut, as setting it is. Saved."""
    position = experiment.get_position_by_name(name)
    if position is None:
        return f"No position named {name!r}."
    if MILLING_POSE not in position.poses:
        return f"{name} has no milling pose."
    state = copy.deepcopy(position.poses[MILLING_POSE])
    state.stage_position = stage_position
    return experiment.set_pose(name, MILLING_POSE, state)
