"""Guided SEM to FM calibration for an offset FM (METEOR, iFLM).

Walks the operator through `fibsem.fm.sem_to_fm.SemToFmCalibration`: the
rotation centre, then slot 1 (and more sites, as a check or when the FM is
tilted from FIB) at the beams and under the objective, each with a round trip
back, then the check and save. The steps are
the calibration's; this dialog shows the images and passes on the clicks.

Images are in the dialog: the reference on the left, the current one on the
right. A click on the right image marks the point a step asks for, drawn
there, and moves nothing. A double-click is a stable move (an FM stable move on an FM image), and
Alt + double-click a vertical move, through the same service calls as the
Microscope tab. Non-modal, like the holder calibration.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from fibsem import conversions
from fibsem.fm.sem_to_fm import SITES, CalibrationRefused, SemToFmCalibration
from fibsem.structures.settings import BeamType, Point
from fibsem.theme.tokens import (
    BORDER_COLOR,
    ERROR_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui import stylesheets
from fibsem.ui.control import control_of
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays.point_overlay import PointOverlay
from fibsem.ui.widgets.holder_calibration_dialog import _label, _RailRow

STEP_TITLES = [
    "Rotation centre",
    "Slot 1 at the beams",
    "Slot 1 at the FM",
    "Round trip",
    "More sites",
    "Check & save",
]
STEP_ROTATION, STEP_BEAMS, STEP_FM, STEP_RETURN, STEP_SITES, STEP_SAVE = range(6)

# What each kind of image is, for a click on it: the beam it was taken with, or
# None for the FM.
FM_VIEW = None


class SemToFmDialog(QtWidgets.QDialog):
    """``calibration_saved`` carries the saved `DeviceCalibration`."""

    calibration_saved = pyqtSignal(object)

    def __init__(
        self, microscope, parent=None, configuration_path: Optional[str] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate SEM to FM")
        self.setModal(False)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"background: {SURFACE_COLOR}; color: {TEXT_COLOR};")
        self.microscope = microscope
        self.calibration = SemToFmCalibration(microscope, configuration_path)
        self._step = STEP_ROTATION
        self._worker: Optional[FunctionWorker] = None
        self._closed = False
        self._saved = False
        # The image on the right, and what took it, for a click to be read in.
        self._current_image = None
        self._current_view: Optional[BeamType] = BeamType.ELECTRON
        self._build_ui()
        self._show_step(STEP_ROTATION)

    # -- layout ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        rail = QtWidgets.QWidget()
        rail.setFixedWidth(180)
        rail.setStyleSheet(f"background: {PANEL_COLOR};")
        rail_layout = QtWidgets.QVBoxLayout(rail)
        rail_layout.setContentsMargins(10, 16, 10, 12)
        rail_layout.setSpacing(2)
        self._rail_rows = [_RailRow(title) for title in STEP_TITLES]
        for row in self._rail_rows:
            rail_layout.addWidget(row)
        rail_layout.addStretch()
        rail_layout.addWidget(
            _label(
                "Measured against the centre of slot 1. Travel to the FM stays off "
                "until this is saved.",
                11,
                TEXT_MUTED_COLOR,
                wrap=True,
            )
        )
        outer.addWidget(rail)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setStyleSheet(f"color: {BORDER_COLOR};")
        outer.addWidget(divider)

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(18, 16, 18, 14)
        right.setSpacing(10)
        self._title = _label("", 16, TEXT_STRONG_COLOR, bold=True)
        self._subtitle = _label("", 12, TEXT_COLOR, wrap=True)
        right.addWidget(self._title)
        right.addWidget(self._subtitle)

        images = QtWidgets.QHBoxLayout()
        self.reference_label = _label("Reference", 11, TEXT_MUTED_COLOR)
        self.current_label = _label("Current", 11, TEXT_MUTED_COLOR)
        self.reference_canvas = FibsemImageCanvas()
        self.current_canvas = FibsemImageCanvas()
        for label, canvas in (
            (self.reference_label, self.reference_canvas),
            (self.current_label, self.current_canvas),
        ):
            column = QtWidgets.QVBoxLayout()
            column.addWidget(label)
            column.addWidget(canvas, 1)
            images.addLayout(column, 1)
        right.addLayout(images, 1)
        self.current_canvas.canvas_clicked.connect(self._on_click)
        self.current_canvas.canvas_double_clicked.connect(self._on_double_click)
        # Where a click marked the point; modal, so it never takes the clicks.
        self.click_marker = PointOverlay(
            color="magenta",
            selected_color="magenta",
            marker="+",
            size=14,
            edge_width=1.2,
            add_on_right_click=False,
            removable=False,
            modal=True,
        )
        self.current_canvas.add_overlay(self.click_marker)

        self.measure_label = _label("", 12, TEXT_COLOR, wrap=True)
        right.addWidget(self.measure_label)
        self.status_label = _label("", 12, TEXT_MUTED_COLOR, wrap=True)
        right.addWidget(self.status_label)

        self._actions = QtWidgets.QHBoxLayout()
        self._actions.setSpacing(8)
        right.addLayout(self._actions)

        footer = QtWidgets.QHBoxLayout()
        self.button_cancel = QtWidgets.QPushButton("Cancel")
        self.button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_cancel.clicked.connect(self.reject)
        footer.addWidget(self.button_cancel)
        footer.addStretch()
        self.button_next = QtWidgets.QPushButton("Next")
        self.button_next.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_next.clicked.connect(self._on_next)
        footer.addWidget(self.button_next)
        right.addLayout(footer)
        outer.addLayout(right, 1)

        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)

    def _set_actions(self, actions: List[Tuple[str, Callable[[], None]]]) -> None:
        """The step's buttons, in order; the last is the step's main one."""
        while self._actions.count():
            item = self._actions.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.action_buttons = {}
        for i, (text, handler) in enumerate(actions):
            button = QtWidgets.QPushButton(text)
            button.setAutoDefault(False)
            button.setStyleSheet(
                stylesheets.PRIMARY_BUTTON_STYLESHEET
                if i == len(actions) - 1
                else stylesheets.SECONDARY_BUTTON_STYLESHEET
            )
            button.clicked.connect(handler)
            self._actions.addWidget(button)
            self.action_buttons[text] = button
        self._actions.addStretch()

    # -- the steps ---------------------------------------------------------------

    def _show_step(self, step: int) -> None:
        self._step = step
        site = len(self.calibration.sites)
        for i, row in enumerate(self._rail_rows):
            row.set_state("done" if i < step else "current" if i == step else "pending")
        self._title.setText(STEP_TITLES[step])
        self.measure_label.setText("")
        self._set_status("")
        if step == STEP_ROTATION:
            self._subtitle.setText(
                "At the beams in SEM, double-click a mark near slot 1 to centre it, "
                "then Capture SEM. Turn to FIB keeps the point; click the mark in the "
                "new SEM image, then Apply. Skip keeps the rotation centre as it is."
            )
            self._set_actions(
                [
                    ("Take SEM image", lambda: self._acquire(BeamType.ELECTRON)),
                    ("Capture SEM", self._capture_rotation_reference),
                    ("Turn to FIB", self._turn),
                    ("Apply", self._apply_rotation_centre),
                ]
            )
            self.button_next.setText("Skip")
        elif step in (STEP_BEAMS, STEP_SITES):
            if step == STEP_BEAMS:
                where = "slot 1"
            elif self._sites_needed() > 1:
                where = f"site {site + 1} of {SITES}, at least 3 mm from slot 1 in y"
            else:
                where = f"site {site + 1}, a check on slot 1's travel"
            self._subtitle.setText(
                f"At the beams in FIB, centre {where} in both images (Alt + "
                "double-click for a vertical move), then Capture beams. The "
                "coincidence is measured as a guide; judge it from the images."
            )
            self._set_actions(
                [
                    ("Take images", self._acquire_both),
                    ("Capture beams", self._capture_beams),
                ]
            )
            self.measure_label.setText(self._fm_pose_text())
            self.button_next.setText("Next")
            self._acquire_both()
        elif step == STEP_FM:
            self._subtitle.setText(
                "Travel retracts the objective, travels to the FM and inserts it. "
                "Double-click the mark to centre it, focus, then Capture FM."
            )
            self._set_actions(
                [
                    ("Travel to FM", self._travel_to_fm),
                    ("Take FM image", self._acquire_fm),
                    ("Capture FM", self._capture_fm),
                ]
            )
            self.button_next.setText("Next")
        elif step == STEP_RETURN:
            self._subtitle.setText(
                "Back to the beams by what this site measured. Click the mark in the "
                "SEM image to record how far it is from where it was centred."
            )
            self._set_actions([("Return to beams", self._return_to_beams)])
            self.button_next.setText("Next")
        elif step == STEP_SAVE:
            self._subtitle.setText(
                "The sites, their mean travel and spread. Save writes the "
                "calibration to the configuration; Check then runs SEM to FM to SEM."
            )
            self._set_actions(
                [
                    ("Add a site", lambda: self._show_step(STEP_SITES)),
                    ("Check", self._check),
                    ("Save", self._save),
                ]
            )
            self.action_buttons["Check"].setEnabled(False)
            self.action_buttons["Add a site"].setEnabled(self._can_add_site())
            self._show_summary()
            self.button_next.setText("Close")
        self._update_next()

    def _update_next(self) -> None:
        sites = self.calibration.sites
        last = sites[-1] if sites else None
        ready = {
            STEP_ROTATION: True,
            STEP_BEAMS: last is not None and last.fm is None,
            STEP_SITES: last is not None and last.fm is None,
            STEP_FM: last is not None and last.fm is not None,
            STEP_RETURN: last is not None and last.residual is not None,
            STEP_SAVE: True,
        }[self._step]
        self.button_next.setEnabled(ready and not self._busy())

    def _on_next(self) -> None:
        step = self._step
        if step == STEP_ROTATION:
            self._show_step(STEP_BEAMS)
        elif step in (STEP_BEAMS, STEP_SITES):
            self._show_step(STEP_FM)
        elif step == STEP_FM:
            self._show_step(STEP_RETURN)
        elif step == STEP_RETURN:
            done = len(self.calibration.sites) >= self._sites_needed()
            self._show_step(STEP_SAVE if done else STEP_SITES)
        else:
            self.accept()

    # -- what the buttons do -----------------------------------------------------

    def _capture_rotation_reference(self) -> None:
        self._run(
            self.calibration.capture_rotation_reference,
            lambda _: self._after_reference(),
        )

    def _after_reference(self) -> None:
        self._show_on(self.reference_canvas, self._current_image, "SEM, centred")
        self._set_status("Captured. Turn to FIB.", OK_COLOR)

    def _turn(self) -> None:
        def turn_and_image():
            self.calibration.turn("FIB")
            return self.microscope.acquire_image(beam_type=BeamType.ELECTRON)

        self._run(
            turn_and_image, self._after_image(BeamType.ELECTRON, "Click the mark.")
        )

    def _apply_rotation_centre(self) -> None:
        try:
            correction = self.calibration.apply_rotation_centre()
        except ValueError as e:
            self._set_status(str(e), ERROR_COLOR)
            return
        self._set_status(
            f"Applied: correction {correction[0] * 1e6:.1f}, {correction[1] * 1e6:.1f} µm. "
            "Turn again to confirm, or Next.",
            OK_COLOR,
        )
        self.button_next.setText("Next")

    def _capture_beams(self) -> None:
        self._set_status("Measuring the coincidence…")
        self._run(self.calibration.capture_beams, lambda site: self._after_beams(site))

    def _after_beams(self, site) -> None:
        dz = site.coincidence.get("dz")
        text = (
            f"Coincidence: {dz * 1e6:.1f} µm height error (a guide, not a gate)."
            if dz is not None
            else f"Coincidence not measured: {site.coincidence.get('error', 'no result')}."
        )
        self.measure_label.setText(f"{self._fm_pose_text()}\n{text}")
        self._show_on(self.reference_canvas, self._current_image, f"{site.name}, beams")
        self._set_status(f"Captured {site.name} at the beams.", OK_COLOR)

    def _travel_to_fm(self) -> None:
        def travel_and_image():
            self.calibration.travel_to_fm()
            return self.microscope.fm.acquire_image()

        self._run(travel_and_image, self._after_image(FM_VIEW, "Centre the mark."))

    def _capture_fm(self) -> None:
        self._run(
            self.calibration.capture_fm,
            lambda site: self._set_status(f"Captured {site.name} at the FM.", OK_COLOR),
        )

    def _return_to_beams(self) -> None:
        def back_and_image():
            self.calibration.return_to_beams()
            return self.microscope.acquire_image(beam_type=BeamType.ELECTRON)

        self._run(
            back_and_image, self._after_image(BeamType.ELECTRON, "Click the mark.")
        )

    def _save(self) -> None:
        try:
            calibration = self.calibration.save()
        except ValueError as e:
            self._set_status(str(e), ERROR_COLOR)
            return
        self._saved = True
        self.action_buttons["Save"].setEnabled(False)
        self.action_buttons["Check"].setEnabled(True)
        self._set_status("Saved. Travel to the FM is on.", OK_COLOR)
        self.calibration_saved.emit(calibration)

    def _check(self) -> None:
        def round_trip():
            self.calibration.check()
            return self.microscope.acquire_image(beam_type=BeamType.ELECTRON)

        self._run(
            round_trip,
            self._after_image(
                BeamType.ELECTRON, "Back at the beams in SEM: is the mark centred?"
            ),
        )

    # -- images and clicks -------------------------------------------------------

    def _acquire(self, beam_type: BeamType) -> None:
        self._run(
            lambda: self.microscope.acquire_image(beam_type=beam_type),
            self._after_image(beam_type, ""),
        )

    def _acquire_both(self) -> None:
        def both():
            sem = self.microscope.acquire_image(beam_type=BeamType.ELECTRON)
            fib = self.microscope.acquire_image(beam_type=BeamType.ION)
            return sem, fib

        def show(images) -> None:
            sem, fib = images
            self._show_on(self.reference_canvas, sem, "SEM")
            self._show_current(fib, BeamType.ION, "FIB")

        self._run(both, show)

    def _acquire_fm(self) -> None:
        self._run(self.microscope.fm.acquire_image, self._after_image(FM_VIEW, ""))

    def _after_image(self, view: Optional[BeamType], status: str):
        def show(image) -> None:
            name = "FM" if view is FM_VIEW else view.name.title()
            self._show_current(image, view, name)
            if status:
                self._set_status(status)

        return show

    def _show_current(self, image, view: Optional[BeamType], name: str) -> None:
        self._current_image, self._current_view = image, view
        self.click_marker.clear_points()
        self._show_on(self.current_canvas, image, name, current=True)

    def _show_on(self, canvas, image, name: str, current: bool = False) -> None:
        if image is None:
            return
        try:
            canvas.set_array(image.data, pixel_size=self._pixel_size(image))
        except Exception as e:  # noqa: BLE001 - an image that won't draw
            logging.warning(f"Could not show the {name} image: {e}")
            return
        (self.current_label if current else self.reference_label).setText(name)

    @staticmethod
    def _pixel_size(image) -> Optional[float]:
        pixel_size = getattr(image.metadata, "pixel_size", None)
        if pixel_size is None:
            return None
        return getattr(pixel_size, "x", pixel_size)

    def _clicked_offset(self, x: float, y: float) -> Optional[Point]:
        """A click on the current image, in metres from its centre, +y up."""
        image = self._current_image
        pixel_size = None if image is None else self._pixel_size(image)
        if pixel_size is None:
            return None
        h, w = image.data.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        return conversions.image_to_microscope_image_coordinates(
            coord=Point(x=x, y=y), image=image.data, pixelsize=pixel_size
        )

    def _on_click(self, x: float, y: float, modifiers) -> None:
        """A click marks the point the step asks for; nothing moves."""
        point = self._clicked_offset(x, y)
        if point is None or self._busy():
            return
        try:
            marks = self._step in (STEP_ROTATION, STEP_RETURN, STEP_SAVE)
            if marks and self._current_view is not None:
                self.click_marker.set_points([(x, y)])
            if self._step == STEP_ROTATION and self._current_view is BeamType.ELECTRON:
                if self.calibration.rotation_reference is None:
                    return
                centre = self.calibration.measure_rotation_centre(point.x, point.y)
                correction = self.calibration.rotation_centre_correction()
                self.measure_label.setText(
                    f"Rotation centre {centre[0] * 1e3:.4f}, {centre[1] * 1e3:.4f} mm; "
                    f"correction {correction[0] * 1e6:.1f}, {correction[1] * 1e6:.1f} µm."
                )
            elif self._step in (STEP_RETURN, STEP_SAVE) and self._current_view:
                residual = self.calibration.record_residual(
                    point.x, point.y, self._current_view
                )
                self.measure_label.setText(
                    f"Residual {residual[0] * 1e6:.1f}, {residual[1] * 1e6:.1f} µm."
                )
        except ValueError as e:
            self._set_status(str(e), ERROR_COLOR)
        self._update_next()

    def _on_double_click(self, x: float, y: float, modifiers) -> None:
        """Double-click is a stable move, Alt + double-click a vertical move, as on
        the Microscope tab; on an FM image, an FM stable move."""
        point = self._clicked_offset(x, y)
        if point is None or self._busy():
            return
        movement = self.microscope.stage_movement
        view = self._current_view
        if view is FM_VIEW:
            move = lambda: movement.fm_stable_move(dx=point.x, dy=point.y)  # noqa: E731
            again = self.microscope.fm.acquire_image
        elif "Alt" in modifiers:
            dx = point.x if view is BeamType.ION else 0.0
            move = lambda: movement.vertical_move(dy=point.y, dx=dx, beam_type=view)  # noqa: E731
            again = lambda: self.microscope.acquire_image(beam_type=view)  # noqa: E731
        else:
            move = lambda: movement.stable_move(dx=point.x, dy=point.y, beam_type=view)  # noqa: E731
            again = lambda: self.microscope.acquire_image(beam_type=view)  # noqa: E731

        def move_and_image():
            move()
            return again()

        name = "FM" if view is FM_VIEW else view.name.title()
        self._run(move_and_image, self._after_image(view, f"Moved; {name} retaken."))

    # -- plumbing ----------------------------------------------------------------

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _run(self, func, on_done) -> None:
        """Run one step off the GUI thread, with the buttons off until it ends."""
        if self._busy():
            return
        self._set_buttons(False)
        self._worker = FunctionWorker(func)
        self._worker.returned.connect(lambda result: self._done(on_done, result))
        self._worker.errored.connect(self._failed)
        self._worker.finished.connect(lambda: self._set_buttons(True))
        self._worker.start(
            control=control_of(self.microscope), action="calibrate SEM to FM"
        )

    def _done(self, on_done, result) -> None:
        if self._closed:
            return
        on_done(result)
        self._update_next()

    def _failed(self, error) -> None:
        if not self._closed:
            self._set_status(str(error), ERROR_COLOR)

    def _set_buttons(self, enabled: bool) -> None:
        if self._closed:
            return
        for button in getattr(self, "action_buttons", {}).values():
            button.setEnabled(enabled)
        if enabled and self._step == STEP_SAVE:
            self.action_buttons["Save"].setEnabled(
                not self._saved and self.calibration.summary()["can_save"]
            )
            self.action_buttons["Check"].setEnabled(self._saved)
            self.action_buttons["Add a site"].setEnabled(self._can_add_site())
        self._update_next()

    def _set_status(self, text: str, colour: str = TEXT_MUTED_COLOR) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"background: transparent; color: {colour}; font-size: 12px;"
        )

    def _sites_needed(self) -> int:
        try:
            return self.calibration.sites_needed()
        except CalibrationRefused:
            return SITES

    def _can_add_site(self) -> bool:
        return not self._saved and len(self.calibration.sites) < SITES

    def _fm_pose_text(self) -> str:
        try:
            dr, dt = self.calibration.fm_pose_difference()
        except ValueError as e:
            return str(e)
        name = self.calibration.fm_orientation()
        where = "Set in Configuration > Geometry > FM place."
        if abs(dr) < 0.05 and abs(dt) < 0.05:
            return f"The FM images in the {name} pose, the same as FIB's. {where}"
        return (
            f"The FM's pose is {dr:+.1f}° from FIB's in rotation and {dt:+.1f}° in "
            f"tilt: re-pose by that much at the beams before capturing. {where}"
        )

    def _show_summary(self) -> None:
        summary = self.calibration.summary()
        lines = []
        for site in summary["sites"]:
            travel = site["travel"]
            residual = site["residual"]
            lines.append(
                f"{site['name']}: travel "
                + (
                    "–"
                    if travel is None
                    else f"{travel['x'] * 1e3:.4f}, {travel['y'] * 1e3:.4f} mm"
                )
                + (
                    ""
                    if residual is None
                    else f"; residual {residual[0] * 1e6:.1f}, {residual[1] * 1e6:.1f} µm"
                )
            )
        if summary["travel"] is not None:
            travel = (
                f"{summary['travel']['x'] * 1e3:.4f}, "
                f"{summary['travel']['y'] * 1e3:.4f} mm"
            )
            lines.append(
                f"Mean travel {travel}; spread {summary['spread'] * 1e6:.1f} µm."
                if summary["checked"]
                else f"Travel {travel}, from one site. Add a site to check it."
            )
        lines += summary["problems"]
        self.measure_label.setText("\n".join(lines))
        self.action_buttons["Save"].setEnabled(summary["can_save"])

    def accept(self) -> None:
        if not self._saved:
            self.calibration.restore()
        super().accept()

    def reject(self) -> None:
        if not self._saved:
            self.calibration.restore()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._closed = True
        if not self._saved:
            self.calibration.restore()
        super().closeEvent(event)


__all__ = ["SemToFmDialog"]
