"""A volume experiment: its positions, and the protocol every slice runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import petname
import yaml

from fibsem.applications.volume.position import VolumePosition
from fibsem.applications.volume.protocol import VolumeProtocol
from fibsem.structures import FibsemRectangle, MicroscopeState, Point
from fibsem.workflows.deciding import ProposalDecisions
from fibsem.workflows.experiment import EXPERIMENT_FILENAME, BaseExperiment
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.saved_format import check_app
from fibsem.workflows.status import TaskStatus
from fibsem.workflows.task_config import TaskConfig

PROTOCOL_FILENAME = "protocol.yaml"


class VolumeExperiment(ProposalDecisions, BaseExperiment):
    """Positions milled and imaged slice by slice. The protocol is saved beside
    the experiment, as ``protocol.yaml``."""

    app: ClassVar[str] = "volume"

    def __init__(
        self,
        path: Path,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_protocol: Optional[VolumeProtocol] = None,
    ) -> None:
        super().__init__(path=path, name=name, metadata=metadata)
        self.positions: List[VolumePosition] = []
        self.task_protocol: VolumeProtocol = (
            task_protocol if task_protocol is not None else VolumeProtocol.default()
        )

    # -- items -----------------------------------------------------------------

    def items(self) -> List[Any]:
        return list(self.positions)

    def position_path(self, position: VolumePosition) -> Path:
        """Where a position's files go: ``positions/<name>/`` under the
        experiment. Not created here; a task creates its folder when it writes."""
        return Path(self.path) / "positions" / position.name

    def item_path(self, item: Any) -> Path:
        if isinstance(item, VolumePosition):
            return self.position_path(item)
        return super().item_path(item)

    def get_position_by_name(self, name: str) -> Optional[VolumePosition]:
        return next((p for p in self.positions if p.name == name), None)

    def next_position_name(self) -> str:
        """A new position's name, fixed for good: its number, then two words,
        ``03-brave-otter``, as a lamella's. Never one a removed position's
        folder still has."""
        number = 1 + max(
            [len(self.positions)]
            + [int(p.name[:2]) for p in self.positions if p.name[:2].isdigit()]
        )
        while True:
            name = f"{number:02d}-{petname.generate(2)}"
            if (
                self.get_position_by_name(name) is None
                and not self.position_path(VolumePosition(name=name)).exists()
            ):
                return name

    def add_position(
        self,
        name: str,
        mill_pose: MicroscopeState,
        poi: Optional[Point] = None,
        iterations: int = 10,
    ) -> VolumePosition:
        """A new position at ``mill_pose``, cut into ``iterations`` slices from
        ``poi``. Names are unique within the experiment."""
        if self.get_position_by_name(name) is not None:
            raise ValueError(f"There is already a position named {name!r}.")
        position = VolumePosition(
            name=name,
            poses={MILLING_POSE: mill_pose},
            poi=poi if poi is not None else Point(),
            iterations=iterations,
        )
        self.positions.append(position)
        self.save()
        return position

    # -- edits ---------------------------------------------------------------------
    # What a route or a widget calls. Each returns why it refused, with nothing
    # changed, or "" once it is made and saved.

    def remove_position(self, name: str) -> str:
        """Take ``name`` out of the experiment. Its folder stays on disk."""
        position = self.get_position_by_name(name)
        if position is None:
            return f"No position named {name!r}."
        self.positions.remove(position)
        self.save()
        return ""

    def patch_position(
        self,
        name: str,
        iterations: Optional[int] = None,
        poi: Optional[Point] = None,
        alignment_area: Optional[FibsemRectangle] = None,
        description: Optional[str] = None,
    ) -> str:
        """Set any of ``name``'s slices, point of interest, alignment area and
        description, all or none, and save once. Never fewer slices than it
        has cut; the point of interest only before its first slice; the area
        inside the frame."""
        position = self.get_position_by_name(name)
        if position is None:
            return f"No position named {name!r}."
        cut = self.slices_cut(position)
        if iterations is not None:
            if isinstance(iterations, bool) or not isinstance(iterations, int):
                return "The slices are a whole number."
            if iterations < 1:
                return "A position needs at least one slice."
            if iterations < cut:
                return f"{name} has already cut {cut} slices."
        if poi is not None:
            if not isinstance(poi, Point):
                return (
                    f"The point of interest must be a point, not {type(poi).__name__}."
                )
            if cut:
                return f"{name} has cut slices from its point of interest."
        if alignment_area is not None and (
            not isinstance(alignment_area, FibsemRectangle)
            or not alignment_area.is_valid_reduced_area
        ):
            return "The alignment area must be a rectangle inside the frame."
        if description is not None and not isinstance(description, str):
            return "The description is text."

        if iterations is not None:
            position.iterations = iterations
        if poi is not None:
            position.poi = Point(poi.x, poi.y)
        if alignment_area is not None:
            position.alignment_area = alignment_area
        if description is not None:
            position.description = description.strip()
        self.save()
        return ""

    def add_task(self, task_type: str, task_name: str) -> TaskConfig:
        """Add a new task of a registered *task_type*, named *task_name*, to
        the protocol (``VolumeProtocol.add_new``). Raises ValueError for an
        unknown type or a name the protocol already has."""
        config = self.task_protocol.add_new(task_type, task_name)
        logging.info(
            f"Added task {config.task_name} ({task_type}) to experiment {self.name}"
        )
        return config

    @property
    def turns_beams_off(self) -> bool:
        return self.task_protocol.options.turn_beams_off

    def remove_task(self, task_name: str) -> None:
        """Take a task out of the protocol and its plan. Task history stays:
        it is the record of what ran."""
        self.task_protocol.remove(task_name)
        logging.info(f"Removed task {task_name} from experiment {self.name}")

    def slices_cut(self, position: VolumePosition) -> int:
        """How many slices ``position`` has cut: a per-slice task completed."""
        return len(
            {
                state.iteration
                for state in position.task_history
                if state.iteration is not None and state.status is TaskStatus.Completed
            }
        )

    # -- saved form --------------------------------------------------------------

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["positions"] = [p.to_dict() for p in self.positions]
        return data

    def save_protocol(self) -> None:
        with open(os.path.join(self.path, PROTOCOL_FILENAME), "w") as f:
            f.write(yaml.safe_dump(self.task_protocol.to_dict(), indent=4))

    @classmethod
    def create(
        cls,
        path: Path,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_protocol: Optional[VolumeProtocol] = None,
    ) -> "VolumeExperiment":
        """A new one in the folder ``name`` under ``path``, saved with its
        protocol."""
        experiment = cls(
            path=path, name=name, metadata=metadata, task_protocol=task_protocol
        )
        os.makedirs(experiment.path, exist_ok=True)
        experiment.save(save_protocol=True)
        return experiment

    @classmethod
    def load(cls, path: Path) -> "VolumeExperiment":
        """The one whose experiment.yaml, or folder, is at ``path``. Another
        application's experiment is refused (``WrongApplication``)."""
        path = Path(path)
        if path.is_dir():
            path = path / EXPERIMENT_FILENAME
        with open(path) as f:
            ddict = yaml.safe_load(f) or {}
        check_app(ddict, cls.app, path)
        protocol_path = path.parent / PROTOCOL_FILENAME
        if protocol_path.exists():
            with open(protocol_path) as f:
                protocol = VolumeProtocol.from_dict(yaml.safe_load(f))
        else:
            logging.warning(
                f"{path.parent.name} has no {PROTOCOL_FILENAME}; using the default."
            )
            protocol = VolumeProtocol.default()
        experiment = cls(
            path=path.parent.parent, name=path.parent.name, task_protocol=protocol
        )
        experiment._read_identity(ddict)
        experiment.positions = [
            VolumePosition.from_dict(p) for p in ddict.get("positions", [])
        ]
        experiment.withdraw_what_was_asked("the app closed before it was answered")
        return experiment
