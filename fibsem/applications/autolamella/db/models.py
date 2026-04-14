"""SQLModel ORM models for the AutoLamella database.

These are DB-layer models only. Application dataclasses (Experiment, Lamella, etc.)
are unchanged — see adapters.py for the mapping between the two.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


def _now() -> float:
    return datetime.timestamp(datetime.now())


def _uuid() -> str:
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# user
# -----------------------------------------------------------------------------

class UserDB(SQLModel, table=True):
    __tablename__ = "user"

    id: str = Field(default_factory=_uuid, primary_key=True)
    username: str = Field(unique=True)
    name: str = Field(default="")
    email: str = Field(default="")
    organization: str = Field(default="")
    role: str = Field(default="user")
    is_default: bool = Field(default=False)
    preferences: str = Field(default="{}")   # JSON blob
    created_at: float = Field(default_factory=_now)

    def get_preferences(self) -> dict:
        return json.loads(self.preferences)

    def set_preferences(self, prefs: dict) -> None:
        self.preferences = json.dumps(prefs)


# -----------------------------------------------------------------------------
# project
# -----------------------------------------------------------------------------

class ProjectDB(SQLModel, table=True):
    __tablename__ = "project"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str
    description: str = Field(default="")
    organisation: str = Field(default="")
    owner_user_id: str = Field(foreign_key="user.id")
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)


# -----------------------------------------------------------------------------
# protocol
# -----------------------------------------------------------------------------

class ProtocolDB(SQLModel, table=True):
    __tablename__ = "protocol"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str
    version: str = Field(default="1.0")
    method: str = Field(default="")
    path: str = Field(default="")
    data: str = Field(default="{}")          # JSON blob
    is_default: bool = Field(default=False)
    created_by: Optional[str] = Field(default=None, foreign_key="user.id")
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)

    def get_data(self) -> dict:
        return json.loads(self.data)


# -----------------------------------------------------------------------------
# session
# -----------------------------------------------------------------------------

class SessionDB(SQLModel, table=True):
    __tablename__ = "session"

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(foreign_key="user.id")
    microscope_name: str = Field(default="")
    connected_at: float = Field(default_factory=_now)
    disconnected_at: Optional[float] = Field(default=None)
    data: str = Field(default="{}")          # JSON: system/hardware metadata

    def get_data(self) -> dict:
        return json.loads(self.data)


# -----------------------------------------------------------------------------
# experiment
# -----------------------------------------------------------------------------

class ExperimentDB(SQLModel, table=True):
    __tablename__ = "experiment"

    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str
    path: str
    description: str = Field(default="")
    user_id: str = Field(foreign_key="user.id")
    project_id: Optional[str] = Field(default=None, foreign_key="project.id")
    session_id: Optional[str] = Field(default=None, foreign_key="session.id")
    protocol_id: Optional[str] = Field(default=None, foreign_key="protocol.id")
    protocol_json: str = Field(default="{}")    # full AutoLamellaTaskProtocol snapshot
    protocol_name: str = Field(default="")
    protocol_version: str = Field(default="")
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)

    def get_protocol(self) -> dict:
        return json.loads(self.protocol_json)


# -----------------------------------------------------------------------------
# lamella
# -----------------------------------------------------------------------------

class LamellaDB(SQLModel, table=True):
    __tablename__ = "lamella"

    id: str = Field(default_factory=_uuid, primary_key=True)
    experiment_id: str = Field(foreign_key="experiment.id")
    petname: str
    defect_state: str = Field(default="NONE")
    current_task_id: Optional[str] = Field(default=None)
    data: str = Field(default="{}")             # JSON blob: poses, alignment, poi, task_config, defect detail
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)

    def get_data(self) -> dict:
        return json.loads(self.data)


# -----------------------------------------------------------------------------
# task_history
# -----------------------------------------------------------------------------

class TaskHistoryDB(SQLModel, table=True):
    __tablename__ = "task_history"

    id: str = Field(default_factory=_uuid, primary_key=True)
    lamella_id: str = Field(foreign_key="lamella.id")
    experiment_id: str = Field(foreign_key="experiment.id")
    name: str = Field(default="")
    task_type: str = Field(default="")
    step: str = Field(default="")
    status: str = Field(default="NotStarted")
    status_message: str = Field(default="")
    start_timestamp: float = Field(default_factory=_now)
    end_timestamp: Optional[float] = Field(default=None)

    @property
    def duration(self) -> Optional[float]:
        if self.end_timestamp is None:
            return None
        return self.end_timestamp - self.start_timestamp
