-- AutoLamella Database Schema
-- Target: SQLite (via SQLModel / SQLAlchemy)
-- UUIDs stored as TEXT; timestamps as REAL (Unix epoch seconds)

PRAGMA foreign_keys = ON;


-- -----------------------------------------------------------------------------
-- user
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user (
    id              TEXT PRIMARY KEY,
    username        TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL DEFAULT '',
    email           TEXT NOT NULL DEFAULT '',
    organization    TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'user'
                        CHECK (role IN ('admin', 'user', 'guest')),
    is_default      INTEGER NOT NULL DEFAULT 0
                        CHECK (is_default IN (0, 1)),
    preferences     TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);


-- -----------------------------------------------------------------------------
-- project
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS project (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    organisation    TEXT NOT NULL DEFAULT '',
    owner_user_id   TEXT NOT NULL REFERENCES user(id) ON DELETE RESTRICT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_project_owner ON project(owner_user_id);


-- -----------------------------------------------------------------------------
-- protocol
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS protocol (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL DEFAULT '1.0',
    method          TEXT NOT NULL DEFAULT '',
    path            TEXT NOT NULL DEFAULT '',
    data            TEXT NOT NULL DEFAULT '{}',
    is_default      INTEGER NOT NULL DEFAULT 0
                        CHECK (is_default IN (0, 1)),
    created_by      TEXT REFERENCES user(id) ON DELETE SET NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);


-- -----------------------------------------------------------------------------
-- session
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES user(id) ON DELETE RESTRICT,
    microscope_name TEXT NOT NULL DEFAULT '',
    connected_at    REAL NOT NULL,
    disconnected_at REAL,
    data            TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_session_user ON session(user_id);


-- -----------------------------------------------------------------------------
-- experiment
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    path            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    user_id         TEXT NOT NULL REFERENCES user(id) ON DELETE RESTRICT,
    project_id      TEXT REFERENCES project(id) ON DELETE SET NULL,
    session_id      TEXT REFERENCES session(id) ON DELETE SET NULL,
    protocol_id     TEXT REFERENCES protocol(id) ON DELETE SET NULL,
    protocol_json   TEXT NOT NULL DEFAULT '{}',
    protocol_name   TEXT NOT NULL DEFAULT '',
    protocol_version TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiment_user    ON experiment(user_id);
CREATE INDEX IF NOT EXISTS idx_experiment_project ON experiment(project_id);
CREATE INDEX IF NOT EXISTS idx_experiment_session ON experiment(session_id);


-- -----------------------------------------------------------------------------
-- lamella
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lamella (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    petname         TEXT NOT NULL,
    defect_state    TEXT NOT NULL DEFAULT 'NONE'
                        CHECK (defect_state IN ('NONE', 'FAILURE', 'REWORK')),
    current_task_id TEXT,
    data            TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,

    UNIQUE (experiment_id, petname)
);

CREATE INDEX IF NOT EXISTS idx_lamella_experiment ON lamella(experiment_id);
CREATE INDEX IF NOT EXISTS idx_lamella_defect     ON lamella(experiment_id, defect_state);


-- -----------------------------------------------------------------------------
-- task_history
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS task_history (
    id              TEXT PRIMARY KEY,
    lamella_id      TEXT NOT NULL REFERENCES lamella(id) ON DELETE CASCADE,
    experiment_id   TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    name            TEXT NOT NULL DEFAULT '',
    task_type       TEXT NOT NULL DEFAULT '',
    step            TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'NotStarted'
                        CHECK (status IN (
                            'NotStarted', 'InProgress', 'Completed', 'Failed', 'Skipped'
                        )),
    status_message  TEXT NOT NULL DEFAULT '',
    start_timestamp REAL NOT NULL,
    end_timestamp   REAL
);

CREATE INDEX IF NOT EXISTS idx_task_history_lamella    ON task_history(lamella_id);
CREATE INDEX IF NOT EXISTS idx_task_history_experiment ON task_history(experiment_id);
CREATE INDEX IF NOT EXISTS idx_task_history_status     ON task_history(experiment_id, status);
CREATE INDEX IF NOT EXISTS idx_task_history_timeline   ON task_history(experiment_id, start_timestamp);
