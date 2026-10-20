"""What every application's experiment is: a folder of items and their record.

An experiment has an identity (``name``, ``id``, ``path``, ``created_at``),
free-text ``metadata`` from the create dialog, the ``session`` that last worked
on it, ``items()``: what its tasks run on, and ``proposals``: what was asked
about the experiment rather than one of its items. :class:`BaseExperiment`
holds those, saves and stamps them, and finds an item by id. An application's
experiment adds its items, its protocol and its saved keys.

:class:`FibsemExperiment` is the plain one, with no items: a folder to record
what is done when no application's experiment is open.
"""

from __future__ import annotations

import importlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type

import yaml
from psygnal import Signal

from fibsem.acting import Actor
from fibsem.structures import FibsemUser, SessionInfo
from fibsem.util.timestamps import now, to_datetime, to_iso
from fibsem.utils import configure_logging as _configure_logging
from fibsem.workflows.proposals import (
    EXPERIMENT_WRITE_LOCK,
    Proposal,
    proposals_from_dict,
    proposals_to_dict,
    record,
)
from fibsem.workflows.saved_format import FORMAT_VERSION, check_app, saved_app

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

EXPERIMENT_FILENAME = "experiment.yaml"


class BaseExperiment:
    """An experiment's identity, metadata, session and items."""

    # The application this experiment belongs to: written at the top of its
    # experiment.yaml, and onto every image its runs acquire.
    app: ClassVar[str] = ""
    CONFIGURATION_FILENAME: ClassVar[str] = "configuration.yaml"

    # Fired after a decision is applied, with (item_id, task_name), on the thread
    # decide() ran on. The work queue listens so a stalled run wakes and rescans.
    decided = Signal(str, str)
    # Fired after a question is recorded mid-task, with (item_id, task_name),
    # on the main thread. The window listens, to put the question up on the
    # widget it is answered on.
    asked = Signal(str, str)
    # Fired after a suggestion on a pending proposal is made or declined, with
    # (item_id, task_name), where the experiment is edited.
    suggested = Signal(str, str)

    def __init__(
        self,
        path: Path,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """A new experiment, in the folder ``name`` under ``path``. Nothing is
        written until it is saved."""
        self.name: str = name
        self.id = str(uuid.uuid4())
        self.path: Path = os.path.join(path, name)
        self.created_at: Optional[datetime] = now()
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        # The instrument, operator and software that last worked on this. Filled in
        # by register_metadata, when a session adopts the experiment and there is a
        # microscope to ask; None until then, because creating an experiment happens
        # in a dialog that has no microscope. See FIB-451.
        self.session: Optional[SessionInfo] = None
        # Why the last save failed, or None when the file holds what memory
        # does. Not written: it describes the file, not the experiment.
        self.save_error: Optional[str] = None
        # What was asked about the experiment rather than one of its items,
        # oldest first, each saying who asked it.
        self.proposals: List[Proposal] = []

    # -- items -----------------------------------------------------------------

    def items(self) -> List[Any]:
        """What this experiment's tasks run on and its proposals sit on."""
        raise NotImplementedError

    def get_item_by_id(self, item_id: str) -> Optional[Any]:
        """The item with ``item_id``. By id -- names change."""
        return next((item for item in self.items() if item.id == item_id), None)

    def item_path(self, item: Any) -> Path:
        """The directory an item's recorded outputs are relative to."""
        return Path(item.path)

    @property
    def turns_beams_off(self) -> bool:
        """Whether a run over this experiment turns the beams off as it ends."""
        return False

    def record_proposal(self, proposal: Proposal) -> Proposal:
        """Put ``proposal``, about the experiment itself, on the record. It
        says who asked it (``asked_by``)."""
        with EXPERIMENT_WRITE_LOCK:
            return record(self.proposals, proposal)

    # -- saved form --------------------------------------------------------------

    def to_dict(self) -> dict:
        """The keys every experiment writes; an application adds its own."""
        return {
            "app": self.app,
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "_id": self.id,
            "path": self.path,
            "created_at": to_iso(self.created_at),
            "metadata": self.metadata,
            "session": self.session.to_dict() if self.session is not None else None,
            "proposals": proposals_to_dict(self.proposals),
        }

    def _read_identity(self, ddict: dict) -> None:
        """Take the keys :meth:`to_dict` writes, other than name and path, from
        a saved experiment."""
        version = ddict.get("format_version") or 0
        if version > FORMAT_VERSION:
            logging.warning(
                f"{self.name} was written in experiment format {version}, newer "
                f"than this version reads ({FORMAT_VERSION}); what it does not "
                "know is dropped, and saving would lose it."
            )
        self.created_at = to_datetime(ddict.get("created_at"))
        self.id = ddict.get("_id", "NULL")
        self.metadata = ddict.get("metadata", {})
        # Absent from every experiment written before FIB-451, and from any that no
        # session has adopted yet. Both mean the same thing -- nothing is known about
        # what produced this -- and None says that without guessing.
        session = ddict.get("session")
        self.session = SessionInfo.from_dict(session) if session else None
        self.proposals = proposals_from_dict(ddict.get("proposals"))

    @property
    def unsaved(self) -> bool:
        """Whether the last save failed, so the file is behind memory and the
        event log (FIB-1352). The next save that succeeds catches it up."""
        return self.save_error is not None

    def save(self, save_protocol: bool = False) -> None:
        """Write the experiment to its experiment.yaml.

        A save that fails raises, and leaves the experiment ``unsaved`` until
        one succeeds: a read-only or locked file, a full disk, a value that
        cannot be written. Nothing in memory is lost by it (FIB-1352).
        """
        try:
            with EXPERIMENT_WRITE_LOCK:
                data = self.to_dict()
            # Serialised in full before the file is opened: dumping straight into
            # it left a zero-byte experiment.yaml behind the moment the writer met
            # a value it could not represent, and nothing then loaded.
            text = yaml.safe_dump(data, indent=4)
            with open(os.path.join(self.path, EXPERIMENT_FILENAME), "w") as f:
                f.write(text)
        except Exception as e:
            self.save_error = f"{type(e).__name__}: {e}"
            raise
        self.save_error = None
        if save_protocol:
            self.save_protocol()

    def save_protocol(self) -> None:
        """Write the experiment's protocol beside it; nothing when it has none."""

    # -- metadata ----------------------------------------------------------------

    @property
    def description(self) -> str:
        """Get the experiment description from metadata."""
        return self.metadata.get("description", "")

    @description.setter
    def description(self, value: str):
        """Set the experiment description in metadata."""
        self.metadata["description"] = value

    @property
    def user(self) -> str:
        """Get the user name from metadata."""
        return self.metadata.get("user", "")

    @user.setter
    def user(self, value: str):
        """Set the user name in metadata."""
        self.metadata["user"] = value

    @property
    def project(self) -> str:
        """Get the project name from metadata."""
        return self.metadata.get("project", "")

    @project.setter
    def project(self, value: str):
        """Set the project name in metadata."""
        self.metadata["project"] = value

    @property
    def organisation(self) -> str:
        """Get the organisation name from metadata."""
        return self.metadata.get("organisation", "")

    @organisation.setter
    def organisation(self, value: str):
        """Set the organisation name in metadata."""
        self.metadata["organisation"] = value

    def actor(self) -> Actor:
        """Who is deciding, as a proposal decision records it, from the operator
        named on the experiment (or the OS account when nobody was)."""
        user = self._declared_user() or FibsemUser.from_environment()
        return Actor.operator(user.name)

    def _declared_user(self) -> Optional[FibsemUser]:
        """The operator named when the experiment was created, if anyone was.

        The create dialog collects a name and organisation as free text into the
        untyped ``metadata`` dict. Those beat ``FibsemUser.from_environment()``,
        which reads the OS account -- on a shared facility login that names the
        workstation rather than the person, and somebody who typed a name meant it.

        Only the two fields the dialog offers are overridden. ``hostname`` stays as
        the environment reports it, because it answers a different question: which
        machine, not which person.
        """
        metadata = self.metadata or {}
        name = metadata.get("user")
        organization = metadata.get("organisation")
        if not name and not organization:
            return None

        user = FibsemUser.from_environment()
        if name:
            user.name = name
        if organization:
            user.organization = organization
        return user

    # -- the session working on it -----------------------------------------------

    def configure_logging(self) -> str:
        """Send this process's log output to the experiment's logfile.

        Note this reconfigures the *root* logger: it is process-global, not
        scoped to this experiment, and it replaces whatever handlers were set up
        before. Two consequences worth knowing before calling it:

        * a process that logs elsewhere stops doing so
        * a second call points logging at the second experiment

        ``load`` and ``create`` deliberately do not call this. Reading an
        experiment must not reach into the caller's logging -- the load dialog
        reads one on every click to preview it -- so the callers that do want it
        say so. The app calls this when it adopts an experiment; a headless
        script that wants its output in the experiment folder calls it after
        loading or creating one. See FIB-421.

        Returns:
            The path of the logfile being written to.
        """
        return _configure_logging(path=self.path, log_filename="logfile")

    def register_metadata(self, microscope: "FibsemMicroscope") -> None:
        """Stamp this experiment's identity onto the images ``microscope`` acquires.

        Which experiment produced an image is a property of the run, not of the GUI.
        This previously happened only in AutoLamellaUI, so images acquired from a
        script, a headless run, or the standalone FibsemUI carried the microscope's
        default ``FibsemExperimentRef()`` -- id ``None`` -- and could not be associated
        with an experiment afterwards. See FIB-449.

        Mirrors ``configure_logging``: ``load`` and ``create`` deliberately do not
        call it, because reading an experiment must not reach into the caller's
        microscope. Callers that own the run say so. The app calls it when it adopts
        an experiment, the task manager calls it when it takes one to run, and a
        script driving the microscope directly calls it itself.

        Registration also runs the other way: this is the only moment the experiment
        meets a microscope, so it is where the experiment learns which session is
        working on it -- instrument, operator, software and plugins (FIB-451).
        """
        from fibsem.utils import _register_metadata

        _register_metadata(
            microscope=microscope,
            application_software=self.app,
            experiment_id=self.id,
            experiment_name=self.name,
        )

        # After _register_metadata, which sets `application` on the very SystemInfo
        # being snapshotted here.
        self.session = SessionInfo.collect(
            microscope, user=self._declared_user(), previous=self.session
        )

        # Written now rather than left to whatever saves next -- relying on someone
        # else's save is how FIB-490 lost every failed task. Only for an experiment
        # that already has a file, though: constructing one deliberately does not
        # touch the disk (FIB-420), and both `create` and `load` guarantee a file
        # exists, so in a real run this always writes. A caller driving an
        # in-memory experiment is not asking this method to give it a directory.
        if os.path.exists(os.path.join(self.path, EXPERIMENT_FILENAME)):
            self._snapshot_configuration(microscope)
            self.save()

    def _snapshot_configuration(self, microscope: "FibsemMicroscope") -> None:
        """Copy the microscope configuration this session runs with into the experiment.

        The experiment already keeps its own copy of the protocol and records which
        instrument and software ran it (`session`). What it did not keep was the
        configuration: the calibration (holder slots, objective limits, pre-tilt)
        and the defaults in effect. Those get edited -- a holder is recalibrated, a
        limit changed -- and afterwards nothing could say what an earlier run used.
        A copy, not a reference, for that reason.

        The live record rather than the file on disk: `microscope.system` is what the
        session actually ran with, including anything applied after loading. A later
        session overwrites it, so it is the configuration of the latest session --
        the same rule `session` follows. An unchanged configuration writes nothing.

        Never fails registration: a copy that could not be written is logged.
        """
        from fibsem.structures import CONFIGURATION_VERSION
        from fibsem.utils import _plain, _write_configuration_file, load_yaml

        path = os.path.join(self.path, self.CONFIGURATION_FILENAME)
        try:
            snapshot = _plain(
                {"version": CONFIGURATION_VERSION, **microscope.system.to_dict()}
            )
            if os.path.exists(path):
                try:
                    if load_yaml(path) == snapshot:
                        return
                except Exception:
                    pass  # unreadable: overwrite it
            _write_configuration_file(path, snapshot)
        except Exception as e:
            logging.warning(
                f"Could not copy the microscope configuration into {path}: {e}"
            )


from fibsem.workflows.deciding import ProposalDecisions  # noqa: E402


class FibsemExperiment(ProposalDecisions, BaseExperiment):
    """An experiment of no application: no items, no protocol. Where what is
    done is recorded -- its events, and what is asked about it -- when no
    application's experiment is open."""

    app: ClassVar[str] = "fibsem"

    def items(self) -> List[Any]:
        return []

    @classmethod
    def create(
        cls, path: Path, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "FibsemExperiment":
        """A new one in the folder ``name`` under ``path``, saved."""
        experiment = cls(path=path, name=name, metadata=metadata)
        os.makedirs(experiment.path, exist_ok=True)
        experiment.save()
        return experiment

    @classmethod
    def load(cls, path: Path) -> "FibsemExperiment":
        """The one whose experiment.yaml, or folder, is at ``path``."""
        path = Path(path)
        if path.is_dir():
            path = path / EXPERIMENT_FILENAME
        with open(path) as f:
            ddict = yaml.safe_load(f) or {}
        check_app(ddict, cls.app, path)
        experiment = cls(path=path.parent.parent, name=path.parent.name)
        experiment._read_identity(ddict)
        experiment.withdraw_what_was_asked("the app closed before it was answered")
        return experiment


# app -> the experiment class that reads it, for :func:`load_experiment`. The
# built-ins are named by import path so this module imports no application.
_BUILTIN_TYPES: Dict[str, str] = {
    "autolamella": "fibsem.applications.autolamella.structures:Experiment",
    "volume": "fibsem.applications.volume.experiment:VolumeExperiment",
}
EXPERIMENT_TYPES: Dict[str, Type[BaseExperiment]] = {
    FibsemExperiment.app: FibsemExperiment
}


def register_experiment_type(cls: Type[BaseExperiment]) -> Type[BaseExperiment]:
    """Make :func:`load_experiment` open ``cls.app``'s experiments as ``cls``."""
    EXPERIMENT_TYPES[cls.app] = cls
    return cls


def experiment_type(app: str) -> Type[BaseExperiment]:
    """The experiment class for ``app``; KeyError when nothing reads it."""
    if app not in EXPERIMENT_TYPES and app in _BUILTIN_TYPES:
        module, name = _BUILTIN_TYPES[app].split(":")
        EXPERIMENT_TYPES[app] = getattr(importlib.import_module(module), name)
    if app not in EXPERIMENT_TYPES:
        raise KeyError(f"No application here reads {app} experiments.")
    return EXPERIMENT_TYPES[app]


def load_experiment(path: Path) -> BaseExperiment:
    """Open the experiment.yaml at ``path`` as the application that wrote it."""
    path = Path(path)
    if path.is_dir():
        path = path / EXPERIMENT_FILENAME
    with open(path) as f:
        ddict = yaml.safe_load(f) or {}
    return experiment_type(saved_app(ddict)).load(path)
