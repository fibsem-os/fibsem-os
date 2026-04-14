"""Adapter functions mapping AutoLamella application dataclasses to DB models.

Each function takes an application dataclass and returns the corresponding
SQLModel DB model. No application code is modified — these are one-way
converters for persistence.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from fibsem.applications.autolamella.db.models import (
    ExperimentDB,
    LamellaDB,
    SessionDB,
    TaskHistoryDB,
    UserDB,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskState,
        AutoLamellaUser,
        Experiment,
        Lamella,
    )


def _now() -> float:
    return datetime.timestamp(datetime.now())


# -----------------------------------------------------------------------------
# user
# -----------------------------------------------------------------------------

def user_to_db(user: "AutoLamellaUser") -> UserDB:
    return UserDB(
        id=user._id,
        username=user.username,
        name=user.name,
        email=user.email,
        organization=user.organization,
        role=user.role,
        is_default=user.is_default,
        preferences=json.dumps(user.preferences),
        created_at=user.created_at,
    )


def user_from_db(row: UserDB) -> "AutoLamellaUser":
    from fibsem.applications.autolamella.structures import AutoLamellaUser
    user = AutoLamellaUser(
        username=row.username,
        name=row.name,
        email=row.email,
        organization=row.organization,
        role=row.role,
        is_default=row.is_default,
        preferences=json.loads(row.preferences),
        created_at=row.created_at,
    )
    user._id = row.id
    return user


# -----------------------------------------------------------------------------
# experiment
# -----------------------------------------------------------------------------

def experiment_from_db(row: ExperimentDB, lamellas: Optional[list] = None) -> "Experiment":
    """Restore an Experiment from a DB row.

    Args:
        row: ExperimentDB row.
        lamellas: list of Lamella objects already restored (e.g. via lamella_from_db).
                  If None, experiment.positions will be empty.
    """
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
        Experiment,
    )
    import os
    path = os.path.dirname(row.path)
    experiment = Experiment(path=path, name=row.name)
    experiment._id = row.id
    experiment.created_at = row.created_at
    experiment.metadata = {"description": row.description}

    protocol_data = json.loads(row.protocol_json)
    if protocol_data:
        experiment.task_protocol = AutoLamellaTaskProtocol.from_dict(protocol_data)

    if lamellas:
        for lamella in lamellas:
            experiment.positions.append(lamella)

    return experiment


# -----------------------------------------------------------------------------
# lamella
# -----------------------------------------------------------------------------

def lamella_from_db(row: LamellaDB, task_history: Optional[list] = None) -> "Lamella":
    """Restore a Lamella from a DB row.

    Args:
        row: LamellaDB row.
        task_history: list of AutoLamellaTaskState objects from the task_history table.
                      If None, lamella.task_history will be empty.
    """
    from fibsem.applications.autolamella.structures import Lamella

    data = json.loads(row.data)
    data["id"] = row.id
    data["petname"] = row.petname
    data["task_history"] = [t.to_dict() for t in task_history] if task_history else []

    return Lamella.from_dict(data)


# -----------------------------------------------------------------------------
# task_history
# -----------------------------------------------------------------------------

def task_state_from_db(row: TaskHistoryDB) -> "AutoLamellaTaskState":
    from fibsem.applications.autolamella.structures import AutoLamellaTaskState, AutoLamellaTaskStatus
    task = AutoLamellaTaskState(
        name=row.name,
        step=row.step,
        task_type=row.task_type,
        lamella_id=row.lamella_id,
        start_timestamp=row.start_timestamp,
        end_timestamp=row.end_timestamp,
        status=AutoLamellaTaskStatus[row.status],
        status_message=row.status_message,
    )
    task.task_id = row.id
    return task


# -----------------------------------------------------------------------------
# session
# -----------------------------------------------------------------------------

def session_to_db(
    user_id: str,
    microscope_name: str = "",
    data: Optional[dict] = None,
) -> SessionDB:
    return SessionDB(
        user_id=user_id,
        microscope_name=microscope_name,
        connected_at=_now(),
        data=json.dumps(data or {}),
    )


# -----------------------------------------------------------------------------
# experiment
# -----------------------------------------------------------------------------

def experiment_to_db(
    experiment: "Experiment",
    user_id: str,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> ExperimentDB:
    protocol = experiment.task_protocol
    protocol_json = json.dumps(protocol.to_dict()) if protocol is not None else "{}"
    protocol_id = protocol._id if protocol is not None else None
    protocol_name = protocol.name if protocol is not None else ""
    protocol_version = protocol.version if protocol is not None else ""

    return ExperimentDB(
        id=experiment._id,
        name=experiment.name,
        path=str(experiment.path),
        description=experiment.description,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        protocol_id=protocol_id,
        protocol_json=protocol_json,
        protocol_name=protocol_name,
        protocol_version=protocol_version,
        created_at=experiment.created_at or _now(),
        updated_at=_now(),
    )


# -----------------------------------------------------------------------------
# lamella
# -----------------------------------------------------------------------------

def lamella_to_db(lamella: "Lamella", experiment_id: str) -> LamellaDB:
    # Key queryable fields as columns; everything else in the JSON blob.
    # task_history is excluded from the blob — it lives in the task_history table.
    data = {
        "path": str(lamella.path),
        "number": lamella.number,
        "alignment_area": lamella.alignment_area.to_dict(),
        "poses": {k: v.to_dict() for k, v in lamella.poses.items()},
        "task_config": {k: v.to_dict() for k, v in lamella.task_config.items()},
        "defect": lamella.defect.to_dict(),
        "milling_angle": lamella.milling_angle,
        "objective_position": lamella.objective_position,
        "poi": lamella.poi.to_dict(),
    }

    return LamellaDB(
        id=lamella._id,
        experiment_id=experiment_id,
        petname=lamella.petname,
        defect_state=lamella.defect.state.name,  # "NONE" | "FAILURE" | "REWORK"
        current_task_id=lamella.task_state.task_id if lamella.task_state else None,
        data=json.dumps(data),
        created_at=_now(),
        updated_at=_now(),
    )


# -----------------------------------------------------------------------------
# task_history
# -----------------------------------------------------------------------------

def task_state_to_db(
    task: "AutoLamellaTaskState",
    experiment_id: str,
) -> TaskHistoryDB:
    return TaskHistoryDB(
        id=task.task_id,
        lamella_id=task.lamella_id,
        experiment_id=experiment_id,
        name=task.name,
        task_type=task.task_type,
        step=task.step,
        status=task.status.name,
        status_message=task.status_message,
        start_timestamp=task.start_timestamp,
        end_timestamp=task.end_timestamp,
    )
