"""Imaging as a service: the imaging routines that step a beam over time, starting
with focus.

The beam's own operations (``acquire``, ``autocontrast``, the instrument's own
``auto_focus``, live view) are commands on the beam. What is here coordinates them
over time, the same way on every backend: ``auto_focus`` is the one place that
chooses between the instrument's autofocus and the image-based sweep, and
``focus_sweep`` is the sweep (`fibsem.autofunctions.autofocus.run_auto_focus`),
which a ``stop`` from another caller ends.

No driver needs its own imaging code yet, so every backend builds this class.
Being a service is what lets an agent find and run these routines over the App
Server, under ``/microscope/services/imaging``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Type, TypeVar, Union

# At import, not under TYPE_CHECKING: the App Server decodes a command's arguments
# from its type hints (an AutoFocusSettings sent as a dict).
from fibsem.autofunctions.autofocus import AutoFocusResult, AutoFocusSettings
from fibsem.cancellation import AnyStopEvent
from fibsem.devices.beam import Beam
from fibsem.devices.core import Role, RoleUnfilled, command
from fibsem.services.core import Service
from fibsem.structures import BeamType, FibsemRectangle

# The role each beam fills.
_BEAM_ROLES: Dict[BeamType, str] = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}

_I = TypeVar("_I", bound="Imaging")


class Imaging(Service):
    """Imaging routines over the beams: ``auto_focus`` (the instrument's own where
    the beam has one, else the sweep), ``focus_sweep`` and ``stop``."""

    electron = Role(Beam, required=False, doc="The electron beam.")
    ion = Role(Beam, required=False, doc="The ion beam.")

    def __init__(self, name: str = "imaging", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._stop_requested = threading.Event()

    def beam(self, beam_type: BeamType) -> Beam:
        """The beam that images as ``beam_type``; a ValueError when this microscope
        has none (the column is disabled)."""
        try:
            return getattr(self, _BEAM_ROLES[beam_type])
        except RoleUnfilled:
            raise ValueError(f"The {beam_type.name} beam is not enabled.") from None

    # -- commands ----------------------------------------------------------------

    @command
    def auto_focus(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        reduced_area: Optional[FibsemRectangle] = None,
    ) -> Optional[AutoFocusResult]:
        """Focus ``beam_type``: with the instrument's own autofocus where the beam
        has one, else with the image-based sweep at the beam's field of view.

        Returns the sweep's result, or None after the instrument's autofocus (it
        reports nothing) or a sweep this beam can't run (its working distance can't
        be set).
        """
        beam = self.beam(beam_type)
        if beam.commands["auto_focus"].available:
            beam.auto_focus(reduced_area)
            return None
        logging.debug(
            f"{beam_type.name} has no autofocus of its own; running the focus sweep."
        )
        return self.focus_sweep(
            beam_type=beam_type, settings=AutoFocusSettings(reduced_area=reduced_area)
        )

    @command
    def focus_sweep(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        hfw: Optional[float] = None,
        settings: Optional[AutoFocusSettings] = None,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
        path: Optional[str] = None,
        name: str = "autofocus",
    ) -> Optional[AutoFocusResult]:
        """Focus ``beam_type`` by sweeping its working distance and scoring an image
        at each step (`fibsem.autofunctions.autofocus.run_auto_focus`).

        ``hfw`` is the probe images' field of view, the beam's own when None. A
        ``stop_event`` set, or ``stop`` called, while it runs puts the working
        distance back and raises `OperationCancelledError`. Given a ``path``, the
        result is saved to ``<path>/<name>``. None when the beam's working distance
        can't be set, so there was nothing to sweep.
        """
        from fibsem.autofunctions.autofocus import run_auto_focus

        beam = self.beam(beam_type)
        if hfw is None:
            hfw = beam.hfw.get_value()
        self._stop_requested.clear()
        events = [e for e in (stop_event, self._stop_requested) if e is not None]
        return run_auto_focus(
            self.parent,
            beam_type=beam_type,
            hfw=hfw,
            settings=settings,
            stop_event=AnyStopEvent(*events),
            path=path,
            name=name,
        )

    @command
    def stop(self) -> None:
        """End the sweep in progress, as cancelled."""
        self._stop_requested.set()


def bind_imaging(service: Type[_I], microscope: Any) -> Optional[_I]:
    """Build a microscope's imaging service of class *service* over its beams, or
    None when it has no beam (every column disabled), and ``auto_focus`` then
    raises."""
    beams = microscope.beams
    if not beams:
        return None
    imaging = service(parent=microscope)
    imaging.fill_roles(
        **{_BEAM_ROLES[beam_type]: beam for beam_type, beam in beams.items()}
    )
    imaging.connect()
    return imaging
