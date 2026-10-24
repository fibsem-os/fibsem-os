"""Development menu: SEM/FIB coincidence alignment at the current position.

Each entry runs the coincidence loop off the GUI thread and says how it went by
toast; the measured image pairs show in the views as they land.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QAction, QMenu

from fibsem.structures.settings import BeamType


class CoincidenceMenu(QObject):
    # CoincidenceProgress from the loop's worker, delivered on the GUI thread.
    progress = pyqtSignal(object)

    def __init__(
        self,
        menu: QMenu,
        *,
        microscope: Callable[[], object],
        experiment_path: Callable[[], Optional[str]],
        show_toast: Callable[..., None],
        parent: Optional[QObject] = None,
    ) -> None:
        """Adds a "Coincidence Alignment" submenu to *menu*. *experiment_path* is
        the open experiment's folder, where each run's diagnostics go."""
        super().__init__(parent)
        self._microscope = microscope
        self._experiment_path = experiment_path
        self._show_toast = show_toast
        self.progress.connect(self._on_progress)

        submenu = menu.addMenu("Coincidence Alignment")
        self.actions = []
        for label, tooltip, run in (
            (
                "Run SEM/FIB Coincidence Alignment",
                "Measure the SEM/FIB coincidence at the current position and "
                "correct it with vertical moves until within tolerance (FIB-868)",
                lambda: self.ensure_coincidence(BeamType.ELECTRON),
            ),
            (
                "Run SEM/FIB Coincidence Alignment (keep FIB view)",
                "As above, but what is centred in the FIB view stays put and the "
                "SEM view is the one corrected - the case after choosing a site "
                "in the FIB view",
                lambda: self.ensure_coincidence(BeamType.ION),
            ),
            (
                "Tilt to Milling Angle (Coincident)",
                "Tilt straight to the configured milling angle and restore "
                "SEM/FIB coincidence there, keeping the FIB view (FIB-899)",
                lambda: self.tilt_coincident("milling"),
            ),
            (
                "Tilt to SEM Orientation (Coincident)",
                "Tilt straight to the SEM orientation and restore SEM/FIB "
                "coincidence there, keeping the FIB view (FIB-899)",
                lambda: self.tilt_coincident("sem"),
            ),
        ):
            action = QAction(label, submenu)
            action.setToolTip(tooltip)
            action.triggered.connect(run)
            submenu.addAction(action)
            self.actions.append(action)

    def _diagnostics_path(self) -> str:
        """Where a run's diagnostic figures go: the open experiment, else the log."""
        from fibsem import config as cfg
        from fibsem.alignment import ALIGNMENT_SUBDIR

        return os.path.join(self._experiment_path() or cfg.LOG_PATH, ALIGNMENT_SUBDIR)

    def _start(self, run, on_done, failed: str) -> None:
        """Run *run* on a worker with the entries off until it ends."""
        from fibsem.ui.qt.threading import thread_worker

        for action in self.actions:
            action.setEnabled(False)
        worker = thread_worker(run)()
        worker.returned.connect(on_done)
        worker.errored.connect(
            lambda exc: self._show_toast(f"{failed}: {exc}", "error")
        )
        worker.finished.connect(lambda: [a.setEnabled(True) for a in self.actions])
        worker.start()

    def ensure_coincidence(self, reference: BeamType) -> None:
        """The align loop at the current position. It acquires its own image
        pairs and may move the stage several times; *reference* is the view
        whose centre is kept."""
        microscope = self._microscope()
        if microscope is None:
            self._show_toast("Not connected to a microscope.", "warning")
            return
        diagnostics_path = self._diagnostics_path()
        # progress crosses back to the GUI thread as a signal, never a call
        on_progress = self.progress.emit

        def run():
            from fibsem.alignment.coincidence import ensure_coincident
            from fibsem.alignment.plotting import save_coincidence_diagnostics

            result = ensure_coincident(
                microscope, on_progress=on_progress, reference=reference
            )
            save_coincidence_diagnostics(result, diagnostics_path)
            return result

        self._start(run, self._on_aligned, "Coincidence alignment failed")

    def tilt_coincident(self, where: str) -> None:
        """Tilt straight to the milling angle or the SEM orientation and run the
        loop at the target, stepping only if the measurement refuses; keeps the
        FIB view."""
        microscope = self._microscope()
        if microscope is None:
            self._show_toast("Not connected to a microscope.", "warning")
            return
        on_progress = self.progress.emit

        def run():
            from fibsem.alignment.coincidence import tilt_coincident
            from fibsem.transformations import get_stage_tilt_from_milling_angle

            if where == "milling":
                target = get_stage_tilt_from_milling_angle(
                    microscope, np.radians(microscope.system.stage.milling_angle)
                )
            else:
                target = microscope.get_orientation("SEM").t
            return tilt_coincident(
                microscope, target, reference=BeamType.ION, on_progress=on_progress
            )

        self._start(run, self._on_tilted, "Coincident tilt failed")

    def _on_progress(self, progress) -> None:
        self._show_toast(progress.describe(), "info", temporary=True)
        # each measured pair in the views as it lands: `imaging.show` is what the
        # image widgets listen to
        m = progress.measurement
        if m is None or m.sem_image is None or m.fib_image is None:
            return
        microscope = self._microscope()
        imaging = microscope.imaging if microscope is not None else None
        if imaging is not None:
            imaging.show(m.sem_image)
            imaging.show(m.fib_image)

    def _on_aligned(self, result) -> None:
        final_dz = result.final.dz * 1e6
        where = f" Diagnostics in {self._diagnostics_path()}"
        if result.converged:
            self._show_toast(
                f"Coincident: residual {final_dz:+.2f} um after "
                f"{result.moves_applied} corrective move(s)."
                f"{' Coarse pass used.' if result.coarse_used else ''}{where}",
                "success",
                duration=10000,
            )
        else:
            self._show_toast(
                f"Not coincident: {result.reason} "
                f"(last measured dz {final_dz:+.2f} um, "
                f"{result.moves_applied} move(s) applied).{where}",
                "warning",
                duration=10000,
            )

    def _on_tilted(self, result) -> None:
        tilts = ", ".join(f"{np.degrees(t):.1f}" for t in result.tilts)
        if result.converged:
            offset = (
                f" Surface {result.tilt_axis_offset * 1e6:+.0f} um off the tilt axis;"
                f" walk of {abs(result.walk) * 1e6:.0f} um"
                f"{' undone' if result.walk_undone else ' left'}."
                if result.tilt_axis_offset is not None
                else ""
            )
            self._show_toast(
                f"Coincident at {np.degrees(result.tilts[-1]):.1f} deg after "
                f"{result.moves_applied} corrective move(s) (tilts visited: {tilts})."
                f"{offset}",
                "success",
                duration=10000,
            )
        else:
            self._show_toast(
                f"Not coincident at the target tilt: {result.reason} "
                f"(tilts visited: {tilts}, {result.moves_applied} move(s) applied).",
                "warning",
                duration=10000,
            )
