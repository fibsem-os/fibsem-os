"""A volume experiment: its positions, and the protocol every slice runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import yaml

from fibsem.applications.volume.position import MILL_POSE, VolumePosition
from fibsem.applications.volume.protocol import VolumeProtocol
from fibsem.structures import FibsemRectangle, MicroscopeState, Point
from fibsem.workflows.deciding import ProposalDecisions
from fibsem.workflows.experiment import EXPERIMENT_FILENAME, BaseExperiment
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
            poses={MILL_POSE: mill_pose},
            poi=poi if poi is not None else Point(),
            iterations=iterations,
        )
        self.positions.append(position)
        self.save()
        return position

    # -- edits ---------------------------------------------------------------------
    # What a route or a widget calls. Each returns why it refused, with nothing
    # changed, or "" once it is made and saved.

    def rename_position(self, name: str, new_name: str) -> str:
        """Rename ``name``, and its folder. A position that has run keeps its
        name: its record and files are under it."""
        position = self.get_position_by_name(name)
        new_name = new_name.strip()
        if position is None:
            return f"No position named {name!r}."
        if not new_name:
            return "A position needs a name."
        if new_name == name:
            return ""
        if self.get_position_by_name(new_name) is not None:
            return f"There is already a position named {new_name!r}."
        if position.task_history:
            return f"{name} has run, so it keeps its name."
        old_path = self.position_path(position)
        position.name = new_name
        new_path = self.position_path(position)
        if new_path.exists():
            position.name = name
            return f"{new_path} already exists."
        if old_path.exists():
            os.rename(old_path, new_path)
        self.save()
        return ""

    def remove_position(self, name: str) -> str:
        """Take ``name`` out of the experiment. Its folder stays on disk."""
        position = self.get_position_by_name(name)
        if position is None:
            return f"No position named {name!r}."
        self.positions.remove(position)
        self.save()
        return ""

    def set_iterations(self, name: str, iterations: int) -> str:
        """Cut ``name`` into ``iterations`` slices; never fewer than it has cut."""
        position = self.get_position_by_name(name)
        if position is None:
            return f"No position named {name!r}."
        if iterations < 1:
            return "A position needs at least one slice."
        cut = self.slices_cut(position)
        if iterations < cut:
            return f"{name} has already cut {cut} slices."
        position.iterations = int(iterations)
        self.save()
        return ""

    def set_poi(self, name: str, poi: Point) -> str:
        """Move ``name``'s point of interest, before its first slice: the
        slices step down from it."""
        position = self.get_position_by_name(name)
        if position is None:
            return f"No position named {name!r}."
        if not isinstance(poi, Point):
            return f"The point of interest must be a point, not {type(poi).__name__}."
        if self.slices_cut(position):
            return f"{name} has cut slices from its point of interest."
        position.poi = Point(poi.x, poi.y)
        self.save()
        return ""

    def set_alignment_area(self, name: str, area: FibsemRectangle) -> str:
        """Set where ``name``'s alignment references are taken; inside the frame."""
        position = self.get_position_by_name(name)
        if position is None:
            return f"No position named {name!r}."
        if not isinstance(area, FibsemRectangle) or not area.is_valid_reduced_area:
            return "The alignment area must be a rectangle inside the frame."
        position.alignment_area = area
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
