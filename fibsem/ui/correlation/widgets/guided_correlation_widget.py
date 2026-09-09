"""Guided correlation: the correlation widget as a walk, not four tabs (FIB-956).

The existing :class:`CorrelationTabWidget` lays its parts out as peer tabs
(Images, Coordinates, Results, Refractive Index) that do not say which one you
are in or what is left. This widget is the *same* widget -- every canvas,
list, panel, handler and file it uses is inherited unchanged -- arranged as
four steps on a rail that reads the widget's own state:

1. **Images** -- load the FIB reference and the FM stack; a readiness strip
   says what this pair supports (rotation from geometry, a spot-burn pattern,
   the z-step) and why not when it does not.
2. **Fiducials** -- the spot-burn pattern is drawn on the FIB image, fitted
   there, and projected into the FM as hollow *predicted* markers. The user
   drags three well-spread ones onto their burns; each drop is fitted and the
   remaining predictions re-project through the refined transform.
3. **Target & surface** -- place the point of interest and the FM surface
   point above it; the refractive-index panel lives here.
4. **Correlate & review** -- run the seeded fit; the result, per-fiducial
   errors and provenance appear in place.

Steps auto-advance on state and stay revisitable; nothing here traps. The
rail must read state rather than declare a second sequence -- the lesson of
the first-session walkthrough (FIB-784) -- so every step's status is derived
from the images, the coordinates and the result.

Built as a subclass rather than a copy: the parent's ``_setup_ui`` is split
into builders (menubar, image panes, side widgets, run bar) that this class
composes differently. Its handlers reach their parts by attribute name, so
they keep working. The core here -- steps 2 to 4 over a loaded pair -- is the
shape of the ``correlation`` review renderer (FIB-957); this file is the
standalone host that adds step 1.

Run standalone::

    python -m fibsem.ui.correlation.widgets.guided_correlation_widget <project dir>

The project directory is a correlation run folder or a lamella folder; the
spot-burn pattern is read from the experiment file found above it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationResult,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)
from fibsem.structures import CameraImageTransform, Point
from fibsem.ui import stylesheets
from fibsem.ui.correlation.widgets.correlation_tab_widget import (
    CorrelationTabWidget,
    load_project,
)
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    ERROR_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)

__all__ = [
    "GuidedCorrelationWidget",
    "find_spot_burns",
    "main",
    "open_project",
]

STEP_IMAGES, STEP_FIDUCIALS, STEP_TARGET, STEP_REVIEW = range(4)
_STEP_TITLES = ("Images", "Fiducials", "Target & surface", "Correlate & review")

# How many confirmed pairs a run needs. The solver fits six degrees of freedom
# even when seeded, so three pairs determine it exactly and leave no residual --
# one bad pair rotates the whole transform and nothing says so. Four is the
# first count with a residual; three will do once a constrained solver fits
# only what the seed leaves open (FIB-774). Re-projection refits the rotation
# only from five pairs and moves the translation alone below that.
_MIN_SEEDED_PAIRS = 4
_MIN_UNSEEDED_PAIRS = 4
_MIN_PAIRS_FOR_ROTATION_REFIT = 5

# A fit applied on a drop or on Fit remaining is a refinement of a position the
# user or the projection already put near the burn; a jump beyond these is the
# fitter finding something else, and the position is kept instead.
_MAX_CREDIBLE_FIT_PX = {"fib": 30.0, "fm": 8.0}

_CHIP_STYLE = (
    "font-size: 11px; padding: 1px 7px; border-radius: 9px; "
    "border: 1px solid {border}; color: {color}; background: {bg};"
)


def _chip(text: str, tone: str = "ok") -> str:
    color = {
        "ok": OK_COLOR,
        "warn": WARN_COLOR,
        "err": ERROR_COLOR,
        "info": ACCENT_COLOR,
    }[tone]
    return (
        f'<span style="{_CHIP_STYLE.format(border=color, color=color, bg=PANEL_COLOR)}">'
        f"{text}</span>"
    )


# ---------------------------------------------------------------------------
# The rail
# ---------------------------------------------------------------------------


class _StepRail(QWidget):
    """Four steps down the left, each with a state dot and a one-line subtitle.

    Purely a view: the widget tells it each step's state; clicking a step asks
    the widget to show it, and a blocked step does not respond.
    """

    step_selected = pyqtSignal(int)
    all_steps_toggled = pyqtSignal(bool)

    _DOT = {
        "todo": (TEXT_MUTED_COLOR, "none"),
        "active": (ACCENT_COLOR, "none"),
        "done": (OK_COLOR, OK_COLOR),
        "stale": (WARN_COLOR, WARN_COLOR),
        "blocked": (BORDER_COLOR, "none"),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(210)
        self.setStyleSheet(
            f"background: {SURFACE_COLOR}; border-right: 1px solid {BORDER_COLOR};"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(4)
        title = QLabel("Correlation")
        title.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-weight: 600; font-size: 13px; border: none;"
        )
        layout.addWidget(title)
        sub = QLabel("guided (preview)")
        sub.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px; border: none;")
        layout.addWidget(sub)
        layout.addSpacing(8)

        self._buttons: List[QPushButton] = []
        self._dots: List[QLabel] = []
        self._subtitles: List[QLabel] = []
        self._states: List[str] = ["todo"] * len(_STEP_TITLES)
        for i, name in enumerate(_STEP_TITLES):
            row = QWidget()
            row.setStyleSheet("border: none;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            dot = QLabel()
            dot.setFixedSize(14, 14)
            row_layout.addWidget(dot, alignment=Qt.AlignmentFlag.AlignTop)
            btn = QPushButton(f"{i + 1} · {name}")
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ text-align: left; color: {TEXT_STRONG_COLOR}; "
                f"font-weight: 600; border: none; padding: 2px 0; }}"
                f"QPushButton:disabled {{ color: {TEXT_MUTED_COLOR}; }}"
            )
            btn.clicked.connect(lambda _=False, k=i: self.step_selected.emit(k))
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)
            col.addWidget(btn)
            subtitle = QLabel("")
            subtitle.setWordWrap(True)
            subtitle.setStyleSheet(
                f"color: {TEXT_MUTED_COLOR}; font-size: 11px; border: none; padding-left: 2px;"
            )
            col.addWidget(subtitle)
            row_layout.addLayout(col, stretch=1)
            layout.addWidget(row)
            self._buttons.append(btn)
            self._dots.append(dot)
            self._subtitles.append(subtitle)
        layout.addStretch(1)
        self._all = QCheckBox("Show all steps")
        self._all.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 12px; border: none;"
        )
        self._all.toggled.connect(self.all_steps_toggled)
        layout.addWidget(self._all)
        self._active = 0
        self._paint()

    def set_step(self, index: int, state: str, subtitle: str) -> None:
        self._states[index] = state
        self._subtitles[index].setText(subtitle)
        self._buttons[index].setEnabled(state != "blocked")
        self._paint()

    def set_active(self, index: int) -> None:
        self._active = index
        self._paint()

    @property
    def all_steps(self) -> bool:
        return self._all.isChecked()

    def _paint(self) -> None:
        for i, dot in enumerate(self._dots):
            state = self._states[i]
            border, fill = self._DOT.get(state, self._DOT["todo"])
            if i == self._active and state not in ("done", "stale"):
                border = ACCENT_COLOR
            dot.setStyleSheet(
                f"border: 2px solid {border}; border-radius: 7px; background: {fill};"
            )
            self._buttons[i].setStyleSheet(
                self._buttons[i].styleSheet()
                + (
                    f"QPushButton {{ background: {PANEL_COLOR}; }}"
                    if i == self._active
                    else ""
                )
            )


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------


class GuidedCorrelationWidget(CorrelationTabWidget):
    """The correlation widget arranged as four guided steps. See the module doc."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # state the parent does not have; set before the parent's __init__ runs
        # _setup_ui, which reads some of it
        self._step = STEP_IMAGES
        self._result_live = False
        self._spot_burns: List[Point] = []
        self._spot_burn_fov: Optional[float] = None
        self._nominal_note = ""
        self._pages: List[QWidget] = []
        # The camera transform to assume for an FM stack that did not record one
        # (every odemis-written stack, and every Arctis stack before 2026-08-04).
        # The library refuses to guess; the host makes the assumption explicit,
        # shows it, and lets the user change it when predictions land mirrored.
        self._fm_transform_assumed = CameraImageTransform.NONE
        self._translation_centred = False
        self._last_projection = (np.zeros((2, 3)), np.zeros(2), 0.0)
        super().__init__(*args, **kwargs)
        self.data_changed.connect(self._refresh_rail)
        self.result_changed.connect(self._refresh_rail)
        self._refresh_rail()
        self._show_step(STEP_IMAGES)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:  # noqa: PLR0915 - one layout, read top to bottom
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._rail = _StepRail()
        self._rail.step_selected.connect(self._on_rail_clicked)
        self._rail.all_steps_toggled.connect(self._set_all_steps)
        root.addWidget(self._rail)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._build_menubar())
        root.addWidget(main, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)
        fib_pane, fm_pane = self._build_image_panes()
        splitter.addWidget(fib_pane)
        splitter.addWidget(fm_pane)

        self._build_side_widgets()
        # The coordinates tab stays alive but off screen: its five list panels are
        # re-homed into the step pages below, while its fit-method and channel
        # combos keep feeding the parent's fit path. The guided flow fits holes.
        self._coords_tab.setVisible(False)
        self._coords_tab._fib_method_combo.set_value("Hole")
        self._coords_tab._fm_fid_method_combo.set_value("Hole")
        self._coords_tab._auto_accept_check.setChecked(True)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)

        self._instruction = QLabel("")
        self._instruction.setWordWrap(True)
        self._instruction.setStyleSheet(
            f"background: {SURFACE_COLOR}; color: {TEXT_STRONG_COLOR}; padding: 8px 12px; "
            f"border-left: 3px solid {ACCENT_COLOR}; font-size: 12px;"
        )
        side_layout.addWidget(self._instruction)

        self._stack = QStackedWidget()
        side_layout.addWidget(self._stack, stretch=1)
        self._all_scroll = QScrollArea()
        self._all_scroll.setWidgetResizable(True)
        self._all_scroll.setVisible(False)
        self._all_host = QWidget()
        self._all_layout = QVBoxLayout(self._all_host)
        self._all_layout.setContentsMargins(0, 0, 0, 0)
        self._all_layout.setSpacing(12)
        self._all_layout.addStretch(1)
        self._all_scroll.setWidget(self._all_host)
        side_layout.addWidget(self._all_scroll, stretch=1)

        self._pages = [
            self._build_images_page(),
            self._build_fiducials_page(),
            self._build_target_page(),
            self._build_review_page(),
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        run_bar = self._build_run_bar()
        # step navigation beside Run / Continue
        self._btn_back = QPushButton("Back")
        self._btn_back.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self._btn_back.clicked.connect(lambda: self._show_step(self._step - 1))
        self._btn_next = QPushButton("Next")
        self._btn_next.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self._btn_next.clicked.connect(lambda: self._show_step(self._step + 1))
        self._run_button_row.insertWidget(0, self._btn_back)
        self._run_button_row.insertWidget(1, self._btn_next)
        self._btn_export = QToolButton()
        self._btn_export.setText("Export")
        self._btn_export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self._btn_export)
        export_menu.addAction(self._action_save_plot)
        export_menu.addAction(self._action_export_csv)
        export_menu.addAction(self._action_save)
        self._btn_export.setMenu(export_menu)
        self._run_button_row.insertWidget(3, self._btn_export)
        side_layout.addWidget(run_bar)

        splitter.addWidget(side)
        splitter.setSizes([480, 480, 400])

        self._build_point_registry()

    # -- pages -----------------------------------------------------------

    @staticmethod
    def _page(title: str, blurb: str) -> Tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        head = QLabel(title)
        head.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 14px; font-weight: 600;"
        )
        layout.addWidget(head)
        text = QLabel(blurb)
        text.setWordWrap(True)
        text.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 12px;")
        layout.addWidget(text)
        return page, layout

    def _build_images_page(self) -> QWidget:
        page, layout = self._page(
            "1 · Images",
            "Load the FIB reference and the FM stack. The widget reads their "
            "geometry and says what this pair supports. An anisotropic stack is "
            "rescaled for the fit automatically; resampling it is optional and "
            "only for viewing and picking.",
        )
        self._readiness = QLabel("")
        self._readiness.setWordWrap(True)
        self._readiness.setTextFormat(Qt.TextFormat.RichText)
        self._readiness.setStyleSheet("font-size: 12px; line-height: 1.8;")
        layout.addWidget(self._readiness)
        assume = QWidget()
        assume_layout = QHBoxLayout(assume)
        assume_layout.setContentsMargins(0, 0, 0, 0)
        assume_layout.setSpacing(8)
        self._lbl_transform = QLabel("FM camera transform (not recorded in the stack):")
        self._lbl_transform.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 12px;"
        )
        assume_layout.addWidget(self._lbl_transform)
        self._cmb_transform = QComboBox()
        for tr in CameraImageTransform:
            self._cmb_transform.addItem(tr.name.replace("_", " ").lower(), tr)
        self._cmb_transform.setToolTip(
            "The flip the camera driver applied when this stack was acquired. Read "
            "from the stack when it recorded one; otherwise assumed. If the "
            "predicted markers land mirrored, this is the setting to change."
        )
        self._cmb_transform.currentIndexChanged.connect(self._on_transform_assumed)
        assume_layout.addWidget(self._cmb_transform)
        assume_layout.addStretch(1)
        self._assume_row = assume
        self._assume_row.setVisible(False)
        layout.addWidget(assume)
        layout.addWidget(self._images_tab, stretch=1)
        return page

    def _build_fiducials_page(self) -> QWidget:
        page, layout = self._page(
            "2 · Fiducials",
            "The spot-burn pattern is drawn on the FIB image and projected into the "
            "FM. Drag the three highlighted markers onto their burns; the rest "
            "re-project after each one.",
        )
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        self._btn_predict = QPushButton("Predict from spot burns")
        self._btn_predict.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._btn_predict.setToolTip(
            "Draw the spot-burn pattern on the FIB image, fit each burn there, and "
            "project them into the FM through the geometry's transform."
        )
        self._btn_predict.clicked.connect(self.predict_fiducials)
        row_layout.addWidget(self._btn_predict)
        self._btn_fit_remaining = QPushButton("Fit remaining")
        self._btn_fit_remaining.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self._btn_fit_remaining.setToolTip(
            "Run the local fit on every predicted FM position from where it is now."
        )
        self._btn_fit_remaining.clicked.connect(self.fit_remaining)
        row_layout.addWidget(self._btn_fit_remaining)
        row_layout.addStretch(1)
        layout.addWidget(row)
        self._pairs_summary = QLabel("")
        self._pairs_summary.setWordWrap(True)
        self._pairs_summary.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12px;")
        layout.addWidget(self._pairs_summary)
        layout.addWidget(self._coords_tab._fib_panel)
        layout.addWidget(self._coords_tab._fm_panel, stretch=1)
        return page

    def _build_target_page(self) -> QWidget:
        page, layout = self._page(
            "3 · Target & surface",
            "Place the point of interest in the FM stack, then the surface point "
            "directly above it. The refractive-index correction acts along the "
            "depth axis the geometry fixed.",
        )
        layout.addWidget(self._coords_tab._poi_panel)
        layout.addWidget(self._coords_tab._fm_surface_panel)
        layout.addWidget(self._coords_tab._surface_panel)
        layout.addWidget(self._ri_tab, stretch=1)
        return page

    def _build_review_page(self) -> QWidget:
        page, layout = self._page(
            "4 · Correlate & review",
            "Run the seeded fit. The result appears here with the corrected target "
            "on the FIB image and the uncorrected position as a ghost. Editing a "
            "point marks this stale.",
        )
        self._provenance = QLabel("")
        self._provenance.setWordWrap(True)
        self._provenance.setTextFormat(Qt.TextFormat.RichText)
        self._provenance.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12px;")
        layout.addWidget(self._provenance)
        layout.addWidget(self._results_tab, stretch=1)
        return page

    # ------------------------------------------------------------------
    # Steps and the rail
    # ------------------------------------------------------------------

    def _show_step(self, index: int) -> None:
        index = max(STEP_IMAGES, min(STEP_REVIEW, index))
        self._step = index
        if not self._rail.all_steps:
            self._stack.setCurrentIndex(index)
        self._rail.set_active(index)
        self._btn_back.setEnabled(index > STEP_IMAGES)
        self._btn_next.setEnabled(index < STEP_REVIEW)
        self._refresh_instruction()

    def _on_rail_clicked(self, index: int) -> None:
        self._show_step(index)

    def _set_all_steps(self, on: bool) -> None:
        """Move the pages between the one-at-a-time stack and a scrolling column."""
        if on:
            for page in self._pages:
                self._stack.removeWidget(page)
                frame = QFrame()
                frame.setFrameShape(QFrame.Shape.StyledPanel)
                frame_layout = QVBoxLayout(frame)
                frame_layout.setContentsMargins(0, 0, 0, 0)
                frame_layout.addWidget(page)
                self._all_layout.insertWidget(self._all_layout.count() - 1, frame)
                page.setVisible(True)
            self._stack.setVisible(False)
            self._all_scroll.setVisible(True)
        else:
            while self._all_layout.count() > 1:
                item = self._all_layout.takeAt(0)
                frame = item.widget()
                if frame is None:
                    continue
                page = frame.layout().itemAt(0).widget()
                page.setParent(None)
                self._stack.addWidget(page)
                frame.deleteLater()
            self._all_scroll.setVisible(False)
            self._stack.setVisible(True)
            self._stack.setCurrentIndex(self._step)

    def _refresh_rail(self, *_: Any) -> None:
        loaded = self._fib_image is not None and self._fm_image is not None
        n_pairs = len(self.data.fib_coordinates)
        n_tentative = sum(
            1
            for c in self._coords_tab.fm_list.coordinates
            if c.status in PointStatus.TENTATIVE
        )
        needed = self._pairs_needed()
        has_poi = bool(self._coords_tab.poi_list.coordinates)
        has_surface = bool(self._coords_tab.fm_surface_list.coordinates) or bool(
            self._coords_tab.surface_list.coordinates
        )
        seed_ok = bool(self._seed_available())

        self._rail.set_step(
            STEP_IMAGES,
            "done" if loaded else "todo",
            (
                "FIB + FM loaded · "
                + ("predictions available" if seed_ok else "fit by hand")
                if loaded
                else "Load FIB and FM"
            ),
        )
        if not loaded:
            fid_state = "blocked"
        elif n_pairs >= needed:
            fid_state = "done"
        else:
            fid_state = "todo"
        fid_sub = (
            f"{n_pairs} of {needed} pairs confirmed"
            + (f" · {n_tentative} predicted" if n_tentative else "")
            if loaded
            else "Confirm three pairs"
        )
        self._rail.set_step(STEP_FIDUCIALS, fid_state, fid_sub)
        if n_pairs < needed:
            target_state = "blocked"
        elif has_poi:
            target_state = "done"
        else:
            target_state = "todo"
        self._rail.set_step(
            STEP_TARGET,
            target_state,
            (
                (
                    "POI placed"
                    + (" + surface" if has_surface else " · surface optional")
                )
                if has_poi
                else "Place the POI"
            ),
        )
        if not has_poi:
            review_state = "blocked"
        elif self._result is None:
            review_state = "todo"
        elif self._result_live:
            review_state = "done"
        else:
            review_state = "stale"
        review_sub = "Seeded fit" if seed_ok else "Unseeded fit"
        if self._result is not None:
            px = self._fib_pixel_size_m()
            rms = self._result.rms_error * (px * 1e6 if px else 1.0)
            unit = "µm" if px else "px"
            review_sub = (
                f"RMS {rms:.2f} {unit}"
                if self._result_live
                else "Stale — points changed, re-run"
            )
        self._rail.set_step(STEP_REVIEW, review_state, review_sub)
        self._refresh_instruction()
        self._refresh_pairs_summary()
        self._btn_predict.setEnabled(loaded and bool(self._spot_burns) and seed_ok)
        self._btn_fit_remaining.setEnabled(n_tentative > 0)

    def _refresh_instruction(self) -> None:
        step = self._step
        if step == STEP_IMAGES:
            self._instruction.setText(
                "Load a FIB reference image and the FM stack."
                if self._fib_image is None or self._fm_image is None
                else "Both images are loaded. Continue to fiducials."
            )
        elif step == STEP_FIDUCIALS:
            fm = self._coords_tab.fm_list.coordinates
            suggested = [
                f"FM{i + 1}"
                for i, c in enumerate(fm)
                if c.status == PointStatus.SUGGESTED
            ]
            n_pairs = len(self.data.fib_coordinates)
            needed = self._pairs_needed()
            if not fm and self._spot_burns and self._seed_available():
                self._instruction.setText(
                    "Press Predict from spot burns to draw the pattern and project it into the FM."
                )
            elif suggested and n_pairs < needed:
                self._instruction.setText(
                    f"Drag {', '.join(suggested)} onto their burns "
                    f"({n_pairs} of {needed} confirmed). Each drop is fitted and the "
                    "remaining predictions re-project."
                )
            elif n_pairs >= needed:
                self._instruction.setText(
                    f"{n_pairs} pairs confirmed. Drag any marker to correct it, "
                    "press Fit remaining to fit the rest, or continue to the target."
                )
            else:
                self._instruction.setText(
                    f"Pick matching fiducials on both images ({n_pairs} of {needed})."
                )
        elif step == STEP_TARGET:
            has_poi = bool(self._coords_tab.poi_list.coordinates)
            self._instruction.setText(
                "Right-click the FM image to add the point of interest."
                if not has_poi
                else "Add the FM surface point directly above the target (right-click, "
                "at the slice where the ice surface is sharp), then Apply on the "
                "refractive-index panel."
            )
        else:
            if self._result is None:
                self._instruction.setText("Press Run Correlation.")
            elif self._result_live:
                self._instruction.setText(
                    "Live result. Continue commits the target; Export saves the plot or CSV."
                )
            else:
                self._instruction.setText(
                    "Points changed since this run. Re-run to update."
                )

    def _refresh_pairs_summary(self) -> None:
        fm = self._coords_tab.fm_list.coordinates
        if not fm:
            self._pairs_summary.setText("")
            return
        counts: Dict[str, int] = {}
        for c in fm:
            key = c.status or ("fitted" if c.fitted else "placed")
            counts[key] = counts.get(key, 0) + 1
        self._pairs_summary.setText(
            " · ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        )

    # ------------------------------------------------------------------
    # Readiness and the seed
    # ------------------------------------------------------------------

    def _seed_available(self) -> bool:
        nominal, note = self._nominal_transform()
        self._nominal_note = note
        return nominal is not None

    def _fm_geometry_recorded(self) -> bool:
        md = getattr(self._fm_image, "metadata", None)
        return getattr(md, "geometry", None) is not None

    def _on_transform_assumed(self, _index: int) -> None:
        self._fm_transform_assumed = self._cmb_transform.currentData()
        self._discard_result()
        self._refresh_readiness()
        self._project_predictions()
        self.data_changed.emit(self.data)

    def _nominal_transform(self):  # type: ignore[override]
        """The parent's, plus the assumed camera transform for a stack without one.

        The assumption is the host's, not the library's: ``nominal_transform``
        still raises for a stack that records no geometry, and this passes the
        geometry built from the FIB image's record and the transform shown in
        step 1 -- so what was assumed is on screen, never silent.
        """
        from fibsem.correlation.geometry import (
            NominalTransformError,
            fm_geometry_for,
            nominal_transform,
        )

        if self._fib_image is None or self._fm_image is None:
            return None, ""
        if self._fm_geometry_recorded():
            return super()._nominal_transform()
        fib_geometry = getattr(self._fib_image.metadata, "hardware_geometry", None)
        if fib_geometry is None:
            return super()._nominal_transform()
        try:
            geometry = fm_geometry_for(fib_geometry, self._fm_transform_assumed)
            return nominal_transform(
                self._fib_image, self._fm_image, fm_geometry=geometry
            ), ""
        except NominalTransformError as exc:
            return None, str(exc)
        except Exception as exc:
            logging.warning(f"Could not build the nominal correlation transform: {exc}")
            return None, "the nominal transform could not be built"

    def _pairs_needed(self) -> int:
        return _MIN_SEEDED_PAIRS if self._seed_available() else _MIN_UNSEEDED_PAIRS

    def _refresh_readiness(self) -> None:
        chips: List[str] = []
        if self._fib_image is None:
            chips.append(_chip("FIB: not loaded", "warn"))
        else:
            md = self._fib_image.metadata
            geometry = getattr(md, "hardware_geometry", None) if md else None
            state = getattr(md, "microscope_state", None) if md else None
            beam = getattr(state, "ion_beam", None)
            rot = getattr(beam, "scan_rotation", None)
            if geometry is None or rot is None:
                chips.append(_chip("FIB: no hardware geometry recorded", "warn"))
            else:
                chips.append(
                    _chip(
                        f"FIB geometry: {'compustage' if geometry.is_compustage else 'offset mount'}, "
                        f"scan rotation {np.degrees(rot):.0f}°"
                    )
                )
        if self._fm_image is None:
            chips.append(_chip("FM: not loaded", "warn"))
        else:
            md = self._fm_image.metadata
            geometry = getattr(md, "geometry", None)
            if geometry is not None:
                chips.append(_chip("FM geometry recorded"))
            else:
                chips.append(
                    _chip(
                        "FM: camera transform not recorded — assuming "
                        f"{self._fm_transform_assumed.name.replace('_', ' ').lower()}",
                        "warn",
                    )
                )
            if hasattr(self, "_assume_row"):
                self._assume_row.setVisible(geometry is None)
            pz = getattr(md, "pixel_size_z", None)
            px = getattr(md, "pixel_size_x", None)
            if px and pz:
                ratio = pz / px
                chips.append(
                    _chip(
                        f"z: {pz * 1e6:.2f} µm slices ({ratio:.1f}× xy) — rescaled for the fit",
                        "warn",
                    )
                    if abs(ratio - 1) > 0.05
                    else _chip("z: isotropic")
                )
        if self._spot_burns:
            chips.append(_chip(f"Pattern: {len(self._spot_burns)} spot burns"))
        else:
            chips.append(_chip("Pattern: no spot burns for this lamella", "warn"))
        if self._fib_image is not None and self._fm_image is not None:
            nominal, note = self._nominal_transform()
            if nominal is not None:
                chips.append(
                    _chip(
                        f"Predictions available · rotation from geometry "
                        f"(tilt {90 - np.degrees(np.arcsin(min(1.0, nominal.foreshortening))):.0f}°)",
                        "info",
                    )
                )
            else:
                chips.append(_chip(f"Fit by hand: {note}", "err"))
        self._readiness.setText(" ".join(chips))

    def set_fib_image(self, image) -> None:  # type: ignore[override]
        super().set_fib_image(image)
        self._refresh_readiness()

    def set_fm_image(self, fm_image) -> None:  # type: ignore[override]
        super().set_fm_image(fm_image)
        self._refresh_readiness()

    def set_spot_burns(
        self, coordinates: Sequence[Point], field_of_view: Optional[float] = None
    ) -> None:
        """The spot-burn pattern, normalised 0-1 to the burn reference image.

        ``field_of_view`` is that reference's hfw; when the loaded FIB image has a
        different hfw the pattern is rescaled about the centre (the stage did not
        move between the two).
        """
        self._spot_burns = list(coordinates)
        self._spot_burn_fov = field_of_view
        self._refresh_readiness()
        self._refresh_rail()

    # ------------------------------------------------------------------
    # Predicted fiducials
    # ------------------------------------------------------------------

    def _pattern_in_fib_px(self) -> List[Tuple[float, float]]:
        if self._fib_image is None:
            return []
        h, w = self._fib_image.data.shape[:2]
        scale = 1.0
        hfw = getattr(
            getattr(self._fib_image.metadata, "image_settings", None), "hfw", None
        )
        if self._spot_burn_fov and hfw:
            scale = self._spot_burn_fov / hfw
        return [
            ((p.x - 0.5) * w * scale + w / 2, (p.y - 0.5) * h * scale + h / 2)
            for p in self._spot_burns
        ]

    def predict_fiducials(self) -> None:
        """Draw the pattern on the FIB image, fit each burn there, project into the FM."""
        if self._fib_image is None or self._fm_image is None or not self._spot_burns:
            return
        nominal, note = self._nominal_transform()
        if nominal is None:
            self._lbl_status.setText(f"Cannot predict: {note}")
            return
        fib_coords = [
            Coordinate(
                point=PointXYZ(x=x, y=y, z=0.0),
                point_type=PointType.FIB,
                provenance=PointProvenance.PATTERN,
            )
            for x, y in self._pattern_in_fib_px()
        ]
        # fit each burn in the FIB image from its pattern position; a failed or
        # surprising fit keeps the pattern position, marked so
        self._coords_tab.fib_list.coordinates = fib_coords
        self._coords_tab.fm_list.coordinates = []
        for coord in self._coords_tab.fib_list.coordinates:
            if not self._fit_quietly(coord):
                coord.status = ""  # the pattern position stands
        fib_coords = self._coords_tab.fib_list.coordinates
        self._refresh_canvas(self._fib_adapter)

        z_slice = float(self._fm_display.current_z)
        fm_coords = [
            Coordinate(
                point=PointXYZ(x=0.0, y=0.0, z=z_slice),
                point_type=PointType.FM,
                status=PointStatus.PREDICTED,
                provenance=PointProvenance.PROJECTED,
            )
            for _ in fib_coords
        ]
        for i in self._suggest_indices([(c.point.x, c.point.y) for c in fib_coords]):
            fm_coords[i].status = PointStatus.SUGGESTED
        self._coords_tab.fm_list.coordinates = fm_coords
        self._project_predictions()
        self._discard_result()
        self.data_changed.emit(self.data)
        placed = (
            "centred on the FM image (the stage metadata gave no usable offset)"
            if self._translation_centred
            else "placed from the stage metadata"
        )
        self._lbl_status.setText(
            f"{len(fib_coords)} burns drawn on the FIB image and projected into the FM, "
            f"{placed}. Drag the highlighted markers onto their burns."
        )

    @staticmethod
    def _suggest_indices(
        points: Sequence[Tuple[float, float]], k: int = 3
    ) -> List[int]:
        """The ``k`` points that span the pattern best (farthest-point sampling)."""
        if len(points) <= k:
            return list(range(len(points)))
        pts = np.asarray(points, dtype=float)
        centre = pts.mean(axis=0)
        chosen = [int(np.argmax(np.linalg.norm(pts - centre, axis=1)))]
        while len(chosen) < k:
            d = np.min(
                np.stack([np.linalg.norm(pts - pts[c], axis=1) for c in chosen]), axis=0
            )
            chosen.append(int(np.argmax(d)))
        return chosen

    def _current_projection(self):
        """``(P, t, z_iso)`` for the FM->FIB map as best known now, or None.

        Rotation and scale from the geometry; translation from the confirmed
        pairs when there are any, else from the stage metadata; the whole thing
        refitted (seeded) once three pairs are confirmed.
        """
        nominal, _ = self._nominal_transform()
        if nominal is None:
            return None
        fib = self._coords_tab.fib_list.coordinates
        fm = self._coords_tab.fm_list.coordinates
        pairs = [
            (a, b)
            for a, b in zip(fib, fm)
            if a.usable and b.usable and a.status != PointStatus.REJECTED
        ]
        zan = nominal.z_anisotropy
        z_iso = (
            float(np.mean([b.point.z for _, b in pairs])) * zan
            if pairs
            else float(self._fm_display.current_z) * zan
        )
        P = nominal.projection
        if len(pairs) >= _MIN_PAIRS_FOR_ROTATION_REFIT:
            from fibsem.correlation.correlation_v2 import _fit_from_seed

            fm_iso = np.array(
                [[b.point.x, b.point.y, b.point.z * zan] for _, b in pairs]
            )
            fib_xy = np.array([[a.point.x, a.point.y, 0.0] for a, _ in pairs])
            try:
                R, s, _rms = _fit_from_seed(
                    fm_iso, fib_xy, nominal.eulers_deg(), nominal.scale
                )
                P = s * R[:2, :]
            except Exception as exc:  # keep the nominal rather than fail the drag
                logging.debug(f"seeded refit for re-projection failed: {exc}")
        if pairs:
            t = np.mean(
                [
                    np.array([a.point.x, a.point.y])
                    - P @ np.array([b.point.x, b.point.y, b.point.z * zan])
                    for a, b in pairs
                ],
                axis=0,
            )
        else:
            t = nominal.translation
            # A translation from stage metadata can be nonsense -- an odemis-written
            # stack records its position in odemis's own frame -- and predictions
            # off the image are useless. Then centre the pattern on the FM image;
            # the user's first drop supplies the real translation.
            if fib and self._fm_image is not None:
                h, w = self._fm_image.data.shape[-2:]
                inv = np.linalg.inv(P[:, :2])
                centre_fib = np.mean([[c.point.x, c.point.y] for c in fib], axis=0)
                centre_fm = inv @ (centre_fib - t - P[:, 2] * z_iso)
                if not (0 <= centre_fm[0] < w and 0 <= centre_fm[1] < h):
                    t = centre_fib - P @ np.array([w / 2, h / 2, z_iso])
                    self._translation_centred = True
                else:
                    self._translation_centred = False
        return P, t, z_iso

    def _project_predictions(self) -> None:
        """Re-place every tentative FM point from its FIB partner."""
        proj = self._current_projection()
        if proj is None:
            return
        P, t, z_iso = proj
        self._last_projection = (P, t, z_iso)
        nominal, _ = self._nominal_transform()
        zan = nominal.z_anisotropy if nominal else 1.0
        fib = self._coords_tab.fib_list.coordinates
        fm = self._coords_tab.fm_list.coordinates
        try:
            inv = np.linalg.inv(P[:, :2])
        except np.linalg.LinAlgError:
            return
        for a, b in zip(fib, fm):
            if b.status not in PointStatus.TENTATIVE:
                continue
            xy = inv @ (np.array([a.point.x, a.point.y]) - t - P[:, 2] * z_iso)
            b.point.x, b.point.y, b.point.z = float(xy[0]), float(xy[1]), z_iso / zan
            self._coords_tab.fm_list.refresh_coordinate(b)
        self._refresh_canvas(self._fm_adapter)

    def _fit_quietly(self, coord: Coordinate) -> bool:
        """Run the local fit at ``coord`` and apply it only when it is credible.

        No dialog, ever: this runs on a drop and on Fit remaining, where a modal
        would stop the flow. A failed or surprising fit leaves the position where
        it is and marks the point ``fit_failed``, which the list shows; the user
        can still ask for the diagnostic through the list's refit button.
        """
        try:
            result = self._run_point_fit(coord)
        except Exception as exc:
            logging.debug(f"local fit raised: {exc}")
            result = None
        side = "fib" if coord.point_type is PointType.FIB else "fm"
        if result is not None and result.status.name == "UNCHANGED":
            # the fitter handed the input back (no feature there to refine on):
            # not a failure to report, and not a fit either
            return False
        if (
            result is None
            or result.fitted is None
            or result.status.name != "OK"
            or result.delta_px > _MAX_CREDIBLE_FIT_PX[side]
        ):
            coord.status = PointStatus.FIT_FAILED
            for spec in self._point_specs.values():
                if spec.point_type is coord.point_type:
                    spec.list_widget.refresh_coordinate(coord)
                    spec.adapter.refresh_coordinate(coord)
            return False
        self._apply_fit_result(result)
        return True

    def _last_translation(self) -> np.ndarray:
        """The FIB-px translation the last projection used (for tests and the status line)."""
        return self._last_projection[1]

    def fit_remaining(self) -> None:
        """Run the local fit on every predicted FM position from where it is."""
        for coord in self._coords_tab.fm_list.coordinates:
            if coord.status not in PointStatus.TENTATIVE:
                continue
            if self._fit_quietly(coord):
                coord.provenance = PointProvenance.DETECTED
        self._project_predictions()
        self._refresh_canvas(self._fm_adapter)
        self.data_changed.emit(self.data)

    # -- hooks into the parent's handlers ----------------------------------

    def _on_canvas_moved(self, coord: Coordinate) -> None:  # type: ignore[override]
        """A drop on a predicted FM marker is the user's answer: fit it there and
        re-project the rest through what the confirmed pairs now say."""
        was_tentative = (
            coord.point_type is PointType.FM and coord.status in PointStatus.TENTATIVE
        )
        if was_tentative:
            coord.status = PointStatus.ADJUSTED
            coord.provenance = PointProvenance.USER
        super()._on_canvas_moved(coord)
        if was_tentative:
            if not self._fit_quietly(coord):
                # the drop itself is the answer; say the fit did not improve it
                coord.status = PointStatus.ADJUSTED
                self._coords_tab.fm_list.refresh_coordinate(coord)
                self._lbl_status.setText(
                    "Kept your position: the local fit did not find the burn there."
                )
            self._project_predictions()
            self.data_changed.emit(self.data)

    def _set_result_live(self, live: bool) -> None:  # type: ignore[override]
        self._result_live = live
        super()._set_result_live(live)
        if hasattr(self, "_rail"):
            self._refresh_rail()

    def _on_result_ready(self, result: CorrelationResult, live: bool = True) -> None:  # type: ignore[override]
        super()._on_result_ready(result, live=live)
        self._refresh_provenance(result)
        if live:
            self._show_step(STEP_REVIEW)

    def _refresh_provenance(self, result: CorrelationResult) -> None:
        fm = self._coords_tab.fm_list.coordinates
        auto = sum(1 for c in fm if c.provenance == PointProvenance.DETECTED)
        hand = sum(
            1
            for c in fm
            if c.provenance in ("", PointProvenance.USER, PointProvenance.IMPORTED)
        )
        rejected = sum(1 for c in fm if c.status == PointStatus.REJECTED)
        parts = [f"<b>Pairs</b> {len(result.reprojected_3d)} used"]
        if auto or hand:
            parts[-1] += f": {auto} auto-found, {hand} hand-placed"
        if rejected:
            parts[-1] += f"; {rejected} rejected"
        check = result.branch_check or {}
        if result.seed is not None:
            parts.append(
                f"<b>Transform</b> seeded from geometry, {check.get('angle_to_nominal_deg', 0.0):.1f}° off nominal"
                + (
                    f", mirror branch RMS {check['rms_mirror']:.1f} px"
                    if check.get("rms_mirror") is not None
                    else ""
                )
            )
        else:
            parts.append("<b>Transform</b> unseeded fit")
        dz = result.dimage_dz_px_per_slice
        if dz is not None:
            parts.append(f"<b>Depth</b> {dz[1]:+.2f} px per slice along image y")
        if result.refractive_index_correction_factor is not None:
            parts.append(
                f"<b>RI</b> ×{result.refractive_index_correction_factor:.3f} "
                f"({result.refractive_index_correction_mode})"
            )
        if check.get("warning"):
            parts.append(f'<span style="color:{WARN_COLOR}">{check["warning"]}</span>')
        self._provenance.setText("<br>".join(parts))

    # ------------------------------------------------------------------
    # What feeds the fit
    # ------------------------------------------------------------------

    @property
    def data(self) -> CorrelationInputData:  # type: ignore[override]
        """The parent's inputs with tentative and rejected pairs left out.

        A predicted position is a guess and must never feed the fit; a rejected
        one stays on screen and in the file but not in the transform. Pairs are
        by index, so a dropped point drops its partner too.
        """
        base = super().data
        fib, fm = base.fib_coordinates, base.fm_coordinates
        kept = [
            (a, b)
            for a, b in zip(fib, fm)
            if a.usable and b.usable and a.status != PointStatus.REJECTED
        ]
        base.fib_coordinates = [a for a, _ in kept]
        base.fm_coordinates = [b for _, b in kept]
        return base

    def _update_run_button(self, data: Optional[CorrelationInputData] = None) -> None:  # type: ignore[override]
        data = data if data is not None else self.data
        can = self._can_run(data)
        self._btn_run.setEnabled(can)
        if can:
            self._lbl_status.setText("Ready.")
            return
        if self._fib_image is None or self._fm_image is None:
            self._lbl_status.setText("Load FIB and FM images to continue.")
            return
        n = len(data.fib_coordinates)
        need = self._pairs_needed()
        missing = []
        if n < need:
            missing.append(
                f"{need - n} more fiducial pair{'s' if need - n != 1 else ''}"
            )
        if not data.poi_coordinates:
            missing.append("a point of interest")
        self._lbl_status.setText("Need " + " and ".join(missing) + ".")

    def _can_run(self, data: Optional[CorrelationInputData] = None) -> bool:  # type: ignore[override]
        data = data if data is not None else self.data
        if self._fib_image is None or self._fm_image is None:
            return False
        if self._worker is not None and self._worker.isRunning():
            return False
        n = len(data.fib_coordinates)
        return (
            n == len(data.fm_coordinates)
            and n >= self._pairs_needed()
            and bool(data.poi_coordinates)
        )


# ---------------------------------------------------------------------------
# Standalone host
# ---------------------------------------------------------------------------


def find_spot_burns(project_dir: str) -> Tuple[List[Point], Optional[float], str]:
    """The spot-burn pattern for the lamella a project folder belongs to.

    Walks up from ``project_dir`` to the experiment file, matches the lamella by
    its folder name, and reads the spot-burn task's coordinates and the field of
    view they are normalised to. Returns ``([], None, reason)`` when any of that
    is missing, so the caller can say why there is no pattern.
    """
    import yaml

    path = os.path.abspath(project_dir)
    experiment_file = None
    lamella_dir = None
    probe = path
    for _ in range(6):
        candidate = os.path.join(probe, "experiment.yaml")
        if os.path.isfile(candidate):
            experiment_file = candidate
            break
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        # the lamella folder is the one directly under the experiment folder
        lamella_dir = probe
        probe = parent
    if experiment_file is None:
        return [], None, "no experiment.yaml above the project folder"
    lamella_name = os.path.basename(lamella_dir) if lamella_dir else None
    try:
        with open(experiment_file) as fh:
            experiment = yaml.safe_load(fh)
    except Exception as exc:
        return [], None, f"could not read {experiment_file}: {exc}"
    positions = experiment.get("positions", []) if isinstance(experiment, dict) else []
    for position in positions:
        name = position.get("petname") or position.get("name")
        folder = (
            str(position.get("path", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        )
        if lamella_name not in (name, folder):
            continue
        for cfg in (position.get("task_config") or {}).values():
            if (
                not isinstance(cfg, dict)
                or cfg.get("task_type") != "SPOT_BURN_FIDUCIAL"
            ):
                continue
            coords = [Point(x=c["x"], y=c["y"]) for c in cfg.get("coordinates", [])]
            fov = (cfg.get("reference_imaging") or {}).get("field_of_view1")
            if coords:
                return coords, fov, ""
            return [], None, f"{lamella_name}: spot-burn task has no coordinates"
        return [], None, f"{lamella_name}: no spot-burn task in the experiment"
    return [], None, f"{lamella_name!r} is not a lamella in {experiment_file}"


def open_project(widget: GuidedCorrelationWidget, path: str) -> str:
    """Open a run folder or a lamella folder, finding the images the run names.

    ``load_project`` expects the images beside the run file. In an experiment
    they are not: the FIB references live in the lamella folder and an FM stack
    can sit at the experiment root, while the run file records their names. So
    this resolves the run first, then looks for each named image in the run
    folder, the lamella folder and the experiment folder before falling back to
    ``load_project``'s own discovery. Returns the run folder used.
    """
    from fibsem.correlation.history import LamellaCorrelation
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.structures import FibsemImage

    path = os.path.abspath(path)
    run_dir = path
    if os.path.isdir(os.path.join(path, "Correlation")):
        latest = LamellaCorrelation.discover(os.path.join(path, "Correlation")).latest()
        if latest is not None:
            run_dir = latest.path
    run_file = next(
        (
            os.path.join(run_dir, name)
            for name in (
                "correlation.json",
                "correlation_result.json",
                "correlation_data.json",
            )
            if os.path.isfile(os.path.join(run_dir, name))
        ),
        None,
    )
    if run_file is None:
        load_project(widget, path)
        return run_dir

    import json

    # Every run file present contributes what it knows: a container written
    # with no image loaded carries no names, while the legacy result beside it
    # still does. First non-empty name per kind wins, container first.
    names: Dict[str, Optional[str]] = {"fib": None, "fm": None}
    for name in (
        "correlation.json",
        "correlation_result.json",
        "correlation_data.json",
    ):
        full = os.path.join(run_dir, name)
        if not os.path.isfile(full):
            continue
        try:
            with open(full) as fh:
                raw = json.load(fh)
        except Exception as exc:
            logging.warning(f"Could not read {full}: {exc}")
            continue
        if not isinstance(raw, dict):
            continue
        inputs = raw.get("input_data") or raw.get("computed_from") or raw
        names["fib"] = names["fib"] or inputs.get("fib_image_filename")
        names["fm"] = names["fm"] or inputs.get("fm_image_filename")
    roots = [run_dir]
    probe = run_dir
    for _ in range(4):
        probe = os.path.dirname(probe)
        roots.append(probe)

    def locate(name: Optional[str], kind: str) -> Optional[str]:
        """The named image, or the one image of that kind beside the run.

        Older run files recorded the FIB reference without its ``_ib`` suffix and
        the FM stack by its *channel* name, so a plain name match is tried first
        and a kind-aware search second: the FIB stem with the beam suffix, and
        for the FM the stack in the lamella folder or the one at the experiment
        root carrying the lamella's number (``L13`` for ``13-bright-gecko``).
        """
        stem = os.path.splitext(os.path.basename(name))[0] if name else ""
        if stem.endswith(".ome"):
            stem = stem[:-4]
        candidates = (
            [name, f"{stem}.tif", f"{stem}.tiff", f"{stem}_ib.tif", f"{stem}.ome.tiff"]
            if name
            else []
        )
        for root in roots:
            for candidate in candidates:
                full = os.path.join(root, os.path.basename(candidate))
                if os.path.isfile(full):
                    return full
        if kind == "fm":
            import glob

            lamella = os.path.basename(roots[2]) if len(roots) > 2 else ""
            number = lamella.split("-")[0] if lamella[:1].isdigit() else ""
            for root in roots[:4]:
                stacks = sorted(glob.glob(os.path.join(root, "*.ome.tif*")))
                if len(stacks) == 1 and root in roots[:3]:
                    return stacks[0]
                if number:
                    tagged = [f for f in stacks if f"L{number}" in os.path.basename(f)]
                    if len(tagged) == 1:
                        return tagged[0]
        return None

    widget.set_project_dir(run_dir)
    fib_path, fm_path = locate(names["fib"], "fib"), locate(names["fm"], "fm")
    if fib_path:
        widget.set_fib_image(FibsemImage.load(fib_path))
    if fm_path:
        widget.set_fm_image(FluorescenceImage.load(fm_path))
    if not (fib_path and fm_path):
        load_project(widget, run_dir)  # whatever it can find beside the run
    widget.load_correlation(run_file)
    return run_dir


def main() -> None:
    import argparse
    import sys

    from PyQt5.QtWidgets import QApplication

    parser = argparse.ArgumentParser(description="Guided FIB-FM correlation (preview)")
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Correlation run folder or lamella folder to open (FIB/FM images, "
        "correlation.json, and the spot-burn pattern from the experiment above it).",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheets.NAPARI_STYLE)

    widget = GuidedCorrelationWidget()
    if args.project:
        run_dir = open_project(widget, args.project)
        burns, fov, reason = find_spot_burns(run_dir)
        if burns:
            widget.set_spot_burns(burns, fov)
        else:
            logging.info(f"No spot-burn pattern: {reason}")
    widget.resize(1500, 900)
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
