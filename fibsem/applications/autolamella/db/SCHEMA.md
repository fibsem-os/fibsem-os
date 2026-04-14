# AutoLamella Database Schema

## Overview

Relational SQLite database for the AutoLamella application. Replaces ad-hoc JSON/YAML file storage to enable analytics, multi-user access, and project-level organisation.

**Stack:** SQLModel (Pydantic + SQLAlchemy) for the DB layer. Application dataclasses (`Experiment`, `Lamella`, etc.) are unchanged — thin adapter functions map them to DB rows.

---

## Tables

### `user`
Application-level user identity (`AutoLamellaUser`). Replaces `microscope.user = FibsemUser.from_environment()`.

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID |
| username | TEXT UNIQUE NOT NULL | OS / login handle |
| name | TEXT NOT NULL | display name |
| email | TEXT | |
| organization | TEXT | |
| role | TEXT | admin / user / guest |
| is_default | BOOLEAN | loaded automatically at startup |
| preferences | TEXT | JSON blob |
| created_at | REAL | Unix timestamp |

---

### `project`
Named collection of experiments (e.g. a sample campaign or grant).

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID |
| name | TEXT NOT NULL | |
| description | TEXT | |
| organisation | TEXT | |
| owner_user_id | TEXT FK → user | |
| created_at | REAL | |
| updated_at | REAL | |

---

### `protocol`
Catalogued / reusable protocol. Experiments always store a full `protocol_json` snapshot regardless of whether this row exists.

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID = `AutoLamellaTaskProtocol._id` |
| name | TEXT NOT NULL | |
| version | TEXT | |
| method | TEXT | e.g. "autolamella", "liftout" |
| path | TEXT | path to `.yaml` on disk |
| data | TEXT | JSON: full protocol serialisation |
| is_default | BOOLEAN | |
| created_by | TEXT FK → user | nullable |
| created_at | REAL | |
| updated_at | REAL | |

---

### `session`
One row per instrument connection. New row each time the user connects to the microscope.

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID |
| user_id | TEXT FK → user | who connected |
| microscope_name | TEXT | e.g. "TESCAN-LYRA3" |
| connected_at | REAL NOT NULL | Unix timestamp |
| disconnected_at | REAL | nullable, set on disconnect |
| data | TEXT | JSON: system/hardware metadata |

---

### `experiment`
A single AutoLamella run. Maps to the `Experiment` dataclass.

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID = `Experiment._id` |
| name | TEXT NOT NULL | |
| path | TEXT NOT NULL | filesystem path |
| description | TEXT | |
| user_id | TEXT FK → user | who ran it |
| project_id | TEXT FK → project | nullable |
| session_id | TEXT FK → session | nullable |
| protocol_id | TEXT FK → protocol | nullable |
| protocol_json | TEXT | full `AutoLamellaTaskProtocol.to_dict()` snapshot |
| protocol_name | TEXT | |
| protocol_version | TEXT | |
| created_at | REAL | |
| updated_at | REAL | |

---

### `lamella`
One physical lamella within an experiment. Hybrid approach: key queryable fields as columns, complex nested data as a JSON blob.

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID = `Lamella._id` |
| experiment_id | TEXT FK → experiment CASCADE | |
| petname | TEXT NOT NULL | e.g. "01-fancy-panda"; UNIQUE per experiment |
| defect_state | TEXT | NONE / FAILURE / REWORK — column for fast yield queries |
| current_task_id | TEXT | nullable FK → task_history (active task) |
| data | TEXT NOT NULL | JSON blob: poses, alignment_area, poi, milling_angle, task_config, defect detail |
| created_at | REAL | |
| updated_at | REAL | |

---

### `task_history`
Core analytics table. One row per task execution. Maps to `AutoLamellaTaskState`.

| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID = `AutoLamellaTaskState.task_id` |
| lamella_id | TEXT FK → lamella CASCADE | |
| experiment_id | TEXT FK → experiment CASCADE | denormalized for fast experiment-level queries |
| name | TEXT NOT NULL | display name e.g. "Mill Rough" |
| task_type | TEXT NOT NULL | free-form string (plugin tasks supported) |
| step | TEXT | current step within task |
| status | TEXT | NotStarted / InProgress / Completed / Failed / Skipped |
| status_message | TEXT | |
| start_timestamp | REAL NOT NULL | |
| end_timestamp | REAL | nullable until completed |

---

## Relationships

```
user ──► project ──► experiment ◄── session
                         │
              protocol ◄─┤ (optional FK + protocol_json snapshot)
                         │
                      lamella
                      (data JSON blob: poses, alignment, poi, task_config)
                         │
                    task_history
                    (one row per task execution)
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| UUID TEXT PKs | Matches `uuid.uuid4()` used throughout application dataclasses |
| `experiment_id` on `task_history` | Avoids join-through-lamella for the most common analytics query |
| `defect_state` column on `lamella` | Frequently queried for yield reporting — avoids JSON parsing |
| `lamella.data` JSON blob | Poses, alignment_area, poi, task_config are complex/nested — `.to_dict()` already exists |
| `task_history` as proper table | Primary analytics target: duration, status, ordering — can't aggregate inside a blob |
| `task_type` free-form TEXT | Plugin tasks are supported; no hardcoded CHECK constraint |
| `protocol_json` snapshot | Full round-trip even if the `protocol` row is later deleted |
| SQLModel for DB layer only | Application dataclasses use `EventedList`/`EventedDict` (psygnal) for UI — migrating them to Pydantic would be a large risky refactor |

---

## File Layout

```
fibsem/applications/autolamella/db/
├── __init__.py
├── SCHEMA.md          # this file
├── schema.sql         # CREATE TABLE statements
├── models.py          # SQLModel ORM models
├── crud.py            # insert / update / query helpers
└── adapters.py        # dataclass ↔ DB row converters (to_db + from_db)
```

## Implementation Status

- [x] Schema design (`SCHEMA.md`)
- [x] `schema.sql` — CREATE TABLE statements for all 6 tables
- [x] `models.py` — SQLModel ORM models (`UserDB`, `ProjectDB`, `ProtocolDB`, `SessionDB`, `ExperimentDB`, `LamellaDB`, `TaskHistoryDB`)
- [x] `adapters.py` — bidirectional converters: `user_to_db/from_db`, `experiment_to_db/from_db`, `lamella_to_db/from_db`, `task_state_to_db/from_db`, `session_to_db`
- [x] `crud.py` — full insert/update/query helpers + `sync_experiment` / `sync_lamella` upsert helpers
- [x] `fibsem/config.py` — `AUTOLAMELLA_DB_PATH` added (`~/.autolamella/autolamella.db`)
- [x] `AutoLamellaUI` wiring — DB engine + default user initialised on startup; `_save_and_sync()` replaces all `experiment.save()` calls in `AutoLamellaUI.py` and `AutoLamellaMainUI.py`

---

## Remaining Work

### 1. Task history — wire into `base.py`
Hook points are `pre_task()` and `post_task()` in `AutoLamellaTask` ([workflows/tasks/base.py](../workflows/tasks/base.py)).

- `pre_task()` (line ~163): call `create_task_history(session, task_state, experiment_id)` after setting `status = InProgress`
- `post_task()` (line ~177): call `update_task_history(session, task_state)` after setting `status = Completed`
- `run()` exception handler (line ~139): call `update_task_history` with `status = Failed` before re-raising

DB engine accessible via `self.parent_ui._db_engine` (guard for headless: `if self.parent_ui is None`).  
`experiment_id` accessible via `self.parent_ui.experiment._id`.

### 2. Sessions — wire into `AutoLamellaUI`
- `connect_to_microscope()` (line ~758 in `AutoLamellaUI.py`): call `create_session(session, user_id, microscope_name=...)`; store returned `SessionDB.id` as `self._db_session_id`
- `disconnect_from_microscope()` (line ~769): call `end_session(session, self._db_session_id)`
- Pass `session_id=self._db_session_id` into `sync_experiment` calls

### 3. Dependencies
Add `sqlmodel` to project dependencies (`pyproject.toml` or `setup.py`). Currently not listed.
