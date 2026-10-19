"""A volume experiment: its positions, and the protocol every slice runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import yaml

from fibsem.applications.volume.position import MILL_POSE, VolumePosition
from fibsem.applications.volume.protocol import VolumeProtocol
from fibsem.structures import MicroscopeState, Point
from fibsem.workflows.deciding import ProposalDecisions
from fibsem.workflows.experiment import EXPERIMENT_FILENAME, BaseExperiment
from fibsem.workflows.saved_format import check_app

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
        return position

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
