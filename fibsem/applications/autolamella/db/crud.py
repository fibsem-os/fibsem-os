"""CRUD helpers for the AutoLamella database.

Usage pattern:

    from fibsem.applications.autolamella.db.crud import get_engine, create_db_and_tables
    from sqlmodel import Session

    engine = get_engine("/path/to/autolamella.db")
    create_db_and_tables(engine)

    with Session(engine) as session:
        user = get_or_create_user(session, AutoLamellaUser.from_environment())
        db_session = create_session(session, user.id, microscope_name="TESCAN-LYRA3")
        ...
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlmodel import Session, SQLModel, create_engine, select

from fibsem.applications.autolamella.db.adapters import (
    experiment_from_db,
    experiment_to_db,
    lamella_from_db,
    lamella_to_db,
    session_to_db,
    task_state_from_db,
    task_state_to_db,
    user_from_db,
    user_to_db,
)
from fibsem.applications.autolamella.db.models import (
    ExperimentDB,
    LamellaDB,
    ProjectDB,
    SessionDB,
    TaskHistoryDB,
    UserDB,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sqlalchemy import Engine
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskState,
        AutoLamellaUser,
        Experiment,
        Lamella,
    )


def _now() -> float:
    return datetime.timestamp(datetime.now())


# -----------------------------------------------------------------------------
# Engine / setup
# -----------------------------------------------------------------------------

def get_engine(db_path: str) -> "Engine":
    """Create a SQLite engine. Creates the file if it does not exist."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


def create_db_and_tables(engine: "Engine") -> None:
    """Create all tables defined in SQLModel metadata (idempotent)."""
    SQLModel.metadata.create_all(engine)


# -----------------------------------------------------------------------------
# user
# -----------------------------------------------------------------------------

def get_or_create_user(session: Session, user: "AutoLamellaUser") -> UserDB:
    """Return the existing DB row for this user (matched by username), or insert it."""
    row = session.exec(select(UserDB).where(UserDB.username == user.username)).first()
    if row is not None:
        return row
    row = user_to_db(user)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_user(session: Session, user_id: str) -> Optional[UserDB]:
    return session.get(UserDB, user_id)


def get_default_user(session: Session) -> Optional[UserDB]:
    return session.exec(select(UserDB).where(UserDB.is_default == True)).first()


def list_users(session: Session) -> List[UserDB]:
    return list(session.exec(select(UserDB)).all())


def update_user(session: Session, user: "AutoLamellaUser") -> Optional[UserDB]:
    row = session.get(UserDB, user._id)
    if row is None:
        return None
    row.name = user.name
    row.email = user.email
    row.organization = user.organization
    row.role = user.role
    row.is_default = user.is_default
    import json
    row.preferences = json.dumps(user.preferences)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


# -----------------------------------------------------------------------------
# project
# -----------------------------------------------------------------------------

def create_project(session: Session, name: str, owner_user_id: str,
                   description: str = "", organisation: str = "") -> ProjectDB:
    row = ProjectDB(
        name=name,
        description=description,
        organisation=organisation,
        owner_user_id=owner_user_id,
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_project(session: Session, project_id: str) -> Optional[ProjectDB]:
    return session.get(ProjectDB, project_id)


def list_projects(session: Session, user_id: Optional[str] = None) -> List[ProjectDB]:
    stmt = select(ProjectDB)
    if user_id:
        stmt = stmt.where(ProjectDB.owner_user_id == user_id)
    return list(session.exec(stmt).all())


# -----------------------------------------------------------------------------
# session (instrument connection)
# -----------------------------------------------------------------------------

def create_session(session: Session, user_id: str,
                   microscope_name: str = "", data: Optional[dict] = None) -> SessionDB:
    row = session_to_db(user_id=user_id, microscope_name=microscope_name, data=data)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def end_session(session: Session, session_id: str) -> Optional[SessionDB]:
    """Set disconnected_at on the session row."""
    row = session.get(SessionDB, session_id)
    if row is None:
        return None
    row.disconnected_at = _now()
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_session(session: Session, session_id: str) -> Optional[SessionDB]:
    return session.get(SessionDB, session_id)


# -----------------------------------------------------------------------------
# experiment
# -----------------------------------------------------------------------------

def create_experiment(session: Session, experiment: "Experiment", user_id: str,
                      project_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> ExperimentDB:
    row = experiment_to_db(experiment, user_id=user_id,
                           project_id=project_id, session_id=session_id)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_experiment(session: Session, experiment_id: str) -> Optional[ExperimentDB]:
    return session.get(ExperimentDB, experiment_id)


def load_experiment(session: Session, experiment_id: str) -> Optional["Experiment"]:
    """Restore a full Experiment (with lamellas and task histories) from the DB."""
    row = session.get(ExperimentDB, experiment_id)
    if row is None:
        return None
    lamellas = [
        load_lamella(session, lrow.id)
        for lrow in session.exec(
            select(LamellaDB).where(LamellaDB.experiment_id == experiment_id)
        ).all()
    ]
    return experiment_from_db(row, lamellas=lamellas)


def update_experiment(session: Session, experiment: "Experiment") -> Optional[ExperimentDB]:
    row = session.get(ExperimentDB, experiment._id)
    if row is None:
        return None
    row.name = experiment.name
    row.path = str(experiment.path)
    row.description = experiment.description
    if experiment.task_protocol is not None:
        import json
        row.protocol_json = json.dumps(experiment.task_protocol.to_dict())
        row.protocol_name = experiment.task_protocol.name
        row.protocol_version = experiment.task_protocol.version
    row.updated_at = _now()
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_experiments(session: Session, user_id: Optional[str] = None,
                     project_id: Optional[str] = None) -> List[ExperimentDB]:
    stmt = select(ExperimentDB)
    if user_id:
        stmt = stmt.where(ExperimentDB.user_id == user_id)
    if project_id:
        stmt = stmt.where(ExperimentDB.project_id == project_id)
    return list(session.exec(stmt).all())


# -----------------------------------------------------------------------------
# lamella
# -----------------------------------------------------------------------------

def create_lamella(session: Session, lamella: "Lamella", experiment_id: str) -> LamellaDB:
    row = lamella_to_db(lamella, experiment_id=experiment_id)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_lamella(session: Session, lamella_id: str) -> Optional[LamellaDB]:
    return session.get(LamellaDB, lamella_id)


def load_lamella(session: Session, lamella_id: str) -> Optional["Lamella"]:
    """Restore a full Lamella (with task history) from the DB."""
    row = session.get(LamellaDB, lamella_id)
    if row is None:
        return None
    history_rows = session.exec(
        select(TaskHistoryDB)
        .where(TaskHistoryDB.lamella_id == lamella_id)
        .order_by(TaskHistoryDB.start_timestamp)
    ).all()
    task_history = [task_state_from_db(h) for h in history_rows]
    return lamella_from_db(row, task_history=task_history)


def update_lamella(session: Session, lamella: "Lamella") -> Optional[LamellaDB]:
    """Sync the lamella data blob and defect_state column back to the DB."""
    row = session.get(LamellaDB, lamella._id)
    if row is None:
        return None
    import json
    row.defect_state = lamella.defect.state.name
    row.current_task_id = lamella.task_state.task_id if lamella.task_state else None
    row.data = json.dumps({
        "path": str(lamella.path),
        "number": lamella.number,
        "alignment_area": lamella.alignment_area.to_dict(),
        "poses": {k: v.to_dict() for k, v in lamella.poses.items()},
        "task_config": {k: v.to_dict() for k, v in lamella.task_config.items()},
        "defect": lamella.defect.to_dict(),
        "milling_angle": lamella.milling_angle,
        "objective_position": lamella.objective_position,
        "poi": lamella.poi.to_dict(),
    })
    row.updated_at = _now()
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_lamellas(session: Session, experiment_id: str) -> List[LamellaDB]:
    return list(session.exec(
        select(LamellaDB).where(LamellaDB.experiment_id == experiment_id)
    ).all())


# -----------------------------------------------------------------------------
# task_history
# -----------------------------------------------------------------------------

def create_task_history(session: Session, task: "AutoLamellaTaskState",
                        experiment_id: str) -> TaskHistoryDB:
    """Insert a new task_history row (called when a task starts)."""
    row = task_state_to_db(task, experiment_id=experiment_id)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def update_task_history(session: Session, task: "AutoLamellaTaskState") -> Optional[TaskHistoryDB]:
    """Update status, step, end_timestamp (called when a task completes or fails)."""
    row = session.get(TaskHistoryDB, task.task_id)
    if row is None:
        return None
    row.status = task.status.name
    row.status_message = task.status_message
    row.step = task.step
    row.end_timestamp = task.end_timestamp
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_task_history(session: Session, experiment_id: Optional[str] = None,
                      lamella_id: Optional[str] = None) -> List[TaskHistoryDB]:
    stmt = select(TaskHistoryDB).order_by(TaskHistoryDB.start_timestamp)
    if experiment_id:
        stmt = stmt.where(TaskHistoryDB.experiment_id == experiment_id)
    if lamella_id:
        stmt = stmt.where(TaskHistoryDB.lamella_id == lamella_id)
    return list(session.exec(stmt).all())


# -----------------------------------------------------------------------------
# sync
# -----------------------------------------------------------------------------

def sync_experiment(session: Session, experiment: "Experiment", user_id: str,
                    project_id: Optional[str] = None,
                    session_id: Optional[str] = None) -> ExperimentDB:
    """Upsert an experiment and all its lamellas to the DB.

    Safe to call multiple times — creates on first call, updates on subsequent calls.
    Does not touch task_history rows (managed separately during task execution).

    Intended to be called alongside Experiment.save():

        experiment.save()
        with Session(engine) as db_session:
            sync_experiment(db_session, experiment, user_id=user.id)
    """
    existing = session.get(ExperimentDB, experiment._id)

    if existing is None:
        row = experiment_to_db(experiment, user_id=user_id,
                               project_id=project_id, session_id=session_id)
        session.add(row)
    else:
        import json
        existing.name = experiment.name
        existing.path = str(experiment.path)
        existing.description = experiment.description
        if experiment.task_protocol is not None:
            existing.protocol_json = json.dumps(experiment.task_protocol.to_dict())
            existing.protocol_name = experiment.task_protocol.name
            existing.protocol_version = experiment.task_protocol.version
        existing.updated_at = _now()
        if project_id is not None:
            existing.project_id = project_id
        if session_id is not None:
            existing.session_id = session_id
        row = existing

    session.commit()
    session.refresh(row)

    # upsert each lamella
    for lamella in experiment.positions:
        sync_lamella(session, lamella, experiment_id=experiment._id)

    return row


def sync_lamella(session: Session, lamella: "Lamella", experiment_id: str) -> LamellaDB:
    """Upsert a single lamella to the DB (data blob + defect_state column).

    Does not touch task_history rows.
    """
    import json

    existing = session.get(LamellaDB, lamella._id)

    if existing is None:
        row = lamella_to_db(lamella, experiment_id=experiment_id)
        session.add(row)
    else:
        existing.defect_state = lamella.defect.state.name
        existing.current_task_id = lamella.task_state.task_id if lamella.task_state else None
        existing.data = json.dumps({
            "path": str(lamella.path),
            "number": lamella.number,
            "alignment_area": lamella.alignment_area.to_dict(),
            "poses": {k: v.to_dict() for k, v in lamella.poses.items()},
            "task_config": {k: v.to_dict() for k, v in lamella.task_config.items()},
            "defect": lamella.defect.to_dict(),
            "milling_angle": lamella.milling_angle,
            "objective_position": lamella.objective_position,
            "poi": lamella.poi.to_dict(),
        })
        existing.updated_at = _now()
        row = existing

    session.commit()
    session.refresh(row)
    return row
