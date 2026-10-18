"""The running application, without Qt.

An :class:`AppSession` holds what a session of an application is: the
microscope it is connected to, the experiment it has open, the event stream of
that connection, and whoever answers the run's questions. A window attaches to
it as one client and reads and writes these through it; a headless server can
be another, so nothing here may need a widget.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fibsem.acting import OPERATOR
from fibsem.workflows.event_recording import EventRecorder


class AppSession:
    """The microscope, the open experiment and the connection's event stream."""

    def __init__(self, responder: Optional[Any] = None) -> None:
        self.microscope: Optional[Any] = None
        self.experiment: Optional[Any] = None
        # Whoever answers the run's questions: the window's QtResponder, or a
        # server-side one when there is no window. The recorder records the
        # prompts it puts up.
        self.responder = responder
        # The connection's event stream (FIB-1031): made when a microscope
        # connects, whatever else runs, and recorded to events.jsonl in the
        # experiment's directory. None only while disconnected.
        self.recorder: Optional[EventRecorder] = None

    def adopt(self, experiment: Any) -> None:
        """Open ``experiment``: the session's, and the one its events are
        recorded to from now on."""
        self.experiment = experiment
        if self.recorder is not None:
            self.recorder.set_experiment(experiment.path, experiment)

    def start_recording(self) -> None:
        """Record events for the connected microscope.

        Keeps the stream it has for the same microscope: a client may ask again
        whenever it refreshes. A different microscope gets a new stream, and
        no microscope none. Never raises: the recording is a record of the
        connection, and failing to keep one must not stop the connection being
        used.
        """
        recorder = self.recorder
        if recorder is not None and recorder.microscope is self.microscope:
            return
        self.stop_recording()
        if self.microscope is None:
            return
        try:
            self.recorder = EventRecorder(
                self.microscope,
                responder=self.responder,
                experiment_path=self.experiment.path if self.experiment else None,
                # tasks and the agent mark their own calls; the rest are the
                # operator's
                default_actor=OPERATOR,
                experiment=self.experiment,
            )
        except Exception:
            logging.exception("event stream failed to start; continuing without it")
            self.recorder = None

    def stop_recording(self) -> None:
        """Close the event stream, whose writer thread would otherwise run
        until the process ends."""
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
