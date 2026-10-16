"""Alignment as a service: bringing the view back onto a saved reference image.

The alignment itself is `fibsem.alignment`'s, the one the workflows use: acquire
an image with the reference's settings, measure its shift from the reference,
and correct it with the beam shift or the stage, a few times over
(`multi_step_alignment_v2`). The service runs it from a saved reference *path*,
since a route can't carry an image, and adds ``measure_shift``, the measurement
alone between two saved images, which moves nothing.

Being a service is what lets an agent align over the App Server, under
``/microscope/services/alignment``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Type, TypeVar

from fibsem.devices.core import command, query
from fibsem.services.core import Service
from fibsem.structures import FibsemImage

_A = TypeVar("_A", bound="Alignment")


class Alignment(Service):
    """Alignment to saved reference images: ``align_to_reference`` corrects the view onto
    one, ``measure_shift`` measures the shift between two, ``compare_methods``
    measures it by every method, ``stop`` ends an
    ``align_to_reference`` between steps."""

    def __init__(self, name: str = "alignment", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._stop_requested = threading.Event()

    @command(long=True)
    def align_to_reference(
        self,
        reference_path: str,
        steps: int = 3,
        subsystem: str = "beam-shift",
        use_autocontrast: bool = False,
        path: Optional[str] = None,
        use_autofocus: bool = False,
        method: str = "cross-correlation",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Align the view onto the saved reference image at *reference_path*:
        acquire with its settings, measure the shift and correct it, *steps*
        times. *subsystem* corrects with the ``beam-shift`` (fine, small range),
        the ``stage`` (a stable move) or ``stage-vertical`` (a vertical move).
        *use_autofocus* focuses before each acquisition; *method* measures the
        shift (as ``measure_shift``); *validate* takes a final image and checks
        that the methods agree on what is left.

        Saves the run (each image, the measurements and a plot) under *path*,
        by default beside the reference, in ``Alignment``. Answers each step's
        shift in metres and pixels with its score, whether the methods agree
        on what is left, and the folder; the run is on the record as an
        ``alignment`` event. ``stop`` ends it between steps, ``cancelled``.
        """
        from fibsem.alignment import (
            AlignmentMethod,
            AlignmentSubsystem,
            multi_step_alignment_v2,
        )

        reference = FibsemImage.load(reference_path)
        if path is None:
            path = os.path.join(os.path.dirname(reference_path), "Alignment")
        self._stop_requested.clear()
        run = multi_step_alignment_v2(
            self.parent,
            reference,
            steps=steps,
            use_autocontrast=use_autocontrast,
            use_autofocus=use_autofocus,
            subsystem=AlignmentSubsystem(subsystem),
            method=AlignmentMethod(method),
            validate=validate,
            stop_event=self._stop_requested,
            run_name=os.path.splitext(os.path.basename(reference_path))[0],
            path=path,
        )
        cancelled = self._stop_requested.is_set()
        return {
            **run.to_dict(),
            "status": "cancelled" if cancelled else "completed",
            "path": os.path.join(path, run.name) if run.results else None,
        }

    @command
    def stop(self) -> None:
        """Stop an ``align_to_reference`` after the step it is on."""
        self._stop_requested.set()

    @query
    def measure_shift(
        self,
        reference_path: str,
        image_path: str,
        method: str = "cross-correlation",
    ) -> Dict[str, Any]:
        """The shift of the saved image at *image_path* from the one at
        *reference_path*, as ``align_to_reference`` measures it: in metres and pixels, with
        the score (higher is better) and whether it cleared the minimum.
        Reads the files and moves nothing. *method* is ``cross-correlation``,
        ``phase-correlation`` or ``skimage-phase-correlation``."""
        from fibsem.alignment import AlignmentMethod, _calculate_shift

        result = _calculate_shift(
            FibsemImage.load(reference_path),
            FibsemImage.load(image_path),
            AlignmentMethod(method),
        )
        return result.to_dict()

    @query
    def compare_methods(
        self,
        reference_path: str,
        image_path: str,
        agreement_px: float = 2.0,
    ) -> Dict[str, Any]:
        """The shift of the saved image at *image_path* from the one at
        *reference_path* by every method, how far apart they are in pixels, and
        whether they agree (within *agreement_px*): the check
        ``align_to_reference`` validates with. Reads the files and moves
        nothing."""
        from fibsem.alignment import compare_alignment_methods

        return compare_alignment_methods(
            FibsemImage.load(reference_path),
            FibsemImage.load(image_path),
            agreement_threshold_px=agreement_px,
        ).to_dict()


def bind_alignment(service: Type[_A], microscope: Any) -> Optional[_A]:
    """Build a microscope's alignment service of class *service*, or None when
    it has no imaging to align with."""
    if getattr(microscope, "imaging", None) is None:
        return None
    alignment = service(parent=microscope)
    alignment.connect()
    return alignment
