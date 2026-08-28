"""One Overview tab, with the imaging modality chosen above the canvas.

The FIB/SEM overview and the fluorescence overview were two top-level tabs sitting next
to each other, driving the same stage over the same experiment through two widgets that
share fifty-five method names. Every fix in the area had to be made twice, and twice it
was not: the beam tab went silent on click-to-move (FIB-765) purely by drifting from the
fluorescence one. This is the tab that holds both (FIB-780).

# What this does and does not merge

It merges the **tab**, not the widgets. Both host tabs are kept whole and put in a stack;
picking a modality raises one. `AutoLamellaOverviewTab` and
`AutoLamellaFluorescenceOverviewTab` are untouched, and everything they and their widgets
do -- poses, projections, settings columns, view chips, acquisition -- goes on working
exactly as it did as a separate tab.

That is deliberate rather than a first step deferred. The two widgets composite
differently, take pixels from different instruments and mark different stage poses; a
single widget with a modality flag would be mostly branches, which is the finding
FIB-693 already recorded. What was worth removing was the *second tab*, not the second
widget.

# Neither side is torn down on a switch

Both tabs stay built and stay subscribed while the other is showing. Three things follow,
and all three are wanted:

* **Each modality remembers its own view.** The beam tab's chip strip is still on the
  view it was left on, because nothing destroyed it. FIB-780 asks for this explicitly;
  here it costs nothing.
* **FIB-706 still applies unchanged.** One overview must not drive the stage while the
  other is acquiring. Both can still be acquiring, so the window's `_apply_overview_locks`
  is as necessary as it was, and reads both tabs the same way.
* **A hidden tab is still doing work.** Which is why unavailability is answered by
  dropping the widget (`set_enabled`, `_can_build`) rather than by hiding a page -- the
  same build-or-drop answer both tabs already give.

# The chips

Two levels, one row, read left to right as it narrows:

    [ FIB/SEM | Fluorescence ] | [ SEM @ SEM · FIB @ Milling ]
       what this tab *is*          what the canvas is *showing*

The view chips are **not built here.** They are the page's own -- the beam widget
discovers its views as they are acquired in and marks the one the next run would land in
-- so this only lends them a place to sit, beside the modality chips. Building a second
copy would be a second thing to keep in step, which is the failure this tab exists to
stop. `_mount_view_strip` moves the active page's strip into the bar and `_unmount_view_strip`
hands it back before that page is rebuilt; the fluorescence page has no such strip, so
its half of the bar is empty and the divider hides.

This does move the view chips off the canvas's own width, which `overview_widget`'s
`canvas_pane` comment had deliberately chosen -- "it keeps saying *this selects what the
canvas is showing*". The bar is now led by a control that governs the settings column
too, so spanning the tab is the honest width for it, and splitting the row to preserve
the old rule would cost two rows of chrome to make a distinction the divider already
makes.

An unavailable modality keeps its chip, disabled, with a tooltip saying why -- the same
choice the tab bar already made for the FM tab (FIB-614), and for the same reason: a
control that appears and vanishes with the hardware is harder to live with than one that
explains itself. That only became true when the chip styles grew a `:disabled` rule --
before it, a greyed chip was pixel-identical to a live one, so "disabled" was a claim the
UI did not actually make.

# No flag on either modality

This tab ships to everyone, and so do both of its chips. `features.overview_canvas_tab`
gated the beam side while the canvas overview sat beside the napari one; it is retired
here rather than carried, because there is nothing left for it to gate -- the canvas
overview *is* the Overview tab now. What survives is `features.napari_overview_tab`,
pointing at the old tab instead: off by default, it brings the Minimap tab back for
anyone who needs it, and goes with that tab before the full release.

A modality is therefore unavailable only when there is no hardware behind it, which is
the FM tab's existing capability check and nothing new.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.ui.autolamella_fluorescence_overview_tab import (
    AutoLamellaFluorescenceOverviewTab,
)
from fibsem.applications.autolamella.ui.autolamella_overview_tab import (
    AutoLamellaOverviewTab,
)
from fibsem.applications.autolamella.ui.overview_tab_base import (
    AutoLamellaOverviewTabBase,
)
from fibsem.ui.tokens import BORDER_COLOR
from fibsem.ui.widgets.overview_widget import (
    MODALITY_CHIP_STYLE,
    VIEW_CHIP_SPACING,
    VIEW_STRIP_STYLE,
)

logger = logging.getLogger(__name__)

# The two imaging systems, as the strip labels them. "FIB/SEM" rather than "Beam"
# because it is what the instrument is called in the room, and it matches the view chips
# below it, which spell out SEM and FIB rather than electron and ion.
MODALITY_FIBSEM = "FIBSEM"
MODALITY_FLUORESCENCE = "FLUORESCENCE"

MODALITY_LABELS = {
    MODALITY_FIBSEM: "FIB/SEM",
    MODALITY_FLUORESCENCE: "Fluorescence",
}

# Left to right, and this order rather than the other: the beam overview is the one every
# system has, and the one a session starts in. It is also the order the two tabs sat in.
MODALITY_ORDER = (MODALITY_FIBSEM, MODALITY_FLUORESCENCE)


class AutoLamellaOverviewContainerTab(QWidget):
    """The Overview tab: a modality strip, and one host tab per modality beneath it."""

    # Whether *either* modality has a live widget, i.e. whether the tab does anything at
    # all. The window listens so it can enable or disable the tab; this object does not
    # touch the tab bar, exactly as neither host tab does.
    availability_changed = pyqtSignal(bool)
    # Forwarded from whichever host tab raised it. The window syncs the other lists; the
    # host tabs go on emitting their own, and the window may still connect to them
    # directly -- this is the tab-shaped signal, not a replacement for theirs.
    lamella_selected = pyqtSignal(object)
    # True while *either* modality is acquiring. The window uses it to lock the other
    # overview (FIB-706); both host tabs stay alive under the stack, so the situation it
    # guards against is unchanged by the merge.
    acquiring_changed = pyqtSignal(bool)
    # The modality now showing, for a host that wants to say so somewhere.
    modality_changed = pyqtSignal(str)

    def __init__(self, autolamella_ui, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.autolamella_ui = autolamella_ui

        self.beam_tab = AutoLamellaOverviewTab(autolamella_ui)
        self.fm_tab = AutoLamellaFluorescenceOverviewTab(autolamella_ui)
        self._tabs: Dict[str, AutoLamellaOverviewTabBase] = {
            MODALITY_FIBSEM: self.beam_tab,
            MODALITY_FLUORESCENCE: self.fm_tab,
        }

        self._modality = MODALITY_FIBSEM
        self._chips: Dict[str, QPushButton] = {}
        # The page whose view strip is currently in the bar, and the strip itself.
        self._mounted: Optional[Tuple[QWidget, QWidget]] = None

        self.stack = QStackedWidget()
        for modality in MODALITY_ORDER:
            self.stack.addWidget(self._tabs[modality])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_modality_strip())
        layout.addWidget(self.stack, stretch=1)

        for modality, tab in self._tabs.items():
            # Availability is per modality and drives that chip; the tab-level answer is
            # the or of the two, recomputed rather than tracked, so a missed signal
            # cannot leave the tab enabled with nothing behind either chip.
            tab.availability_changed.connect(
                partial(self._on_tab_availability, modality)
            )
            tab.lamella_selected.connect(self.lamella_selected)
            # Re-derived from both rather than forwarded: the bool arriving here says
            # what *one* tab is doing, and a host locking on it would unlock while the
            # other was still running.
            tab.acquiring_changed.connect(self._on_tab_acquiring)

        self._refresh_chips()

    # ── the modality strip ───────────────────────────────────────────────

    def _build_modality_strip(self) -> QWidget:
        """One bar: the modality chips, a divider, then the active page's view chips.

        Side by side rather than stacked, so the whole of "what am I looking at" is one
        row of chrome instead of two. The order reads left to right as it narrows --
        imaging system, then the view within it.

        The view chips are not built here. They belong to the page's own widget, which
        discovers its views as they are acquired in and knows which one the next run
        would land in; this only lends them a place to sit. `_mount_view_strip` is what
        moves the active page's strip in, and moves it back out before that page is
        rebuilt.

        The modality chips are in a scroll area for the same reason the view chips are:
        a row of controls in a plain layout sets the *window's* minimum width. Two chips
        is a small floor, but the whole bar has to be allowed to get narrow, and one
        fixed half would hold the window open on its own.
        """
        chips_row = QWidget()
        self._strip_layout = QHBoxLayout(chips_row)
        self._strip_layout.setContentsMargins(0, 4, 0, 4)
        self._strip_layout.setSpacing(VIEW_CHIP_SPACING)

        for modality in MODALITY_ORDER:
            chip = QPushButton(MODALITY_LABELS[modality])
            chip.setCheckable(True)
            chip.setChecked(modality == self._modality)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(MODALITY_CHIP_STYLE)
            chip.clicked.connect(partial(self._on_chip_clicked, modality))
            self._strip_layout.addWidget(chip)
            self._chips[modality] = chip

        self.modality_strip = QScrollArea()
        self.modality_strip.setWidget(chips_row)
        self.modality_strip.setWidgetResizable(True)
        self.modality_strip.setFrameShape(QFrame.NoFrame)
        self.modality_strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.modality_strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.modality_strip.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        # The strip's own background, the same way the page's view strip carries it, and
        # not `background: transparent` -- a scroll area's *viewport* is a separate
        # widget that the sheet does not reach, so transparent left it painting the
        # window's base colour instead: measured (30,32,39) against the bar's (38,41,48),
        # a visible rectangle around the modality chips.
        self.modality_strip.setObjectName("viewStrip")
        self.modality_strip.setStyleSheet(VIEW_STRIP_STYLE)
        chips_row.adjustSize()
        self.modality_strip.setFixedHeight(chips_row.sizeHint().height())

        # The divider the two levels are read across. A rule rather than a gap: with a
        # gap alone the modality chips read as the first two of one long row, which is
        # the reading this whole arrangement exists to prevent.
        self._divider = QFrame()
        self._divider.setFrameShape(QFrame.VLine)
        self._divider.setFrameShadow(QFrame.Plain)
        self._divider.setStyleSheet(f"color: {BORDER_COLOR};")
        self._divider.setFixedHeight(chips_row.sizeHint().height())
        self._divider.setVisible(False)

        # Where the active page's view strip is mounted. Empty for a page that has no
        # views to choose between -- the fluorescence one, which has a single camera.
        self._view_slot = QWidget()
        self._view_slot_layout = QHBoxLayout(self._view_slot)
        self._view_slot_layout.setContentsMargins(0, 0, 0, 0)
        self._view_slot_layout.setSpacing(0)

        # A `QFrame`, not a plain `QWidget`: a bare `QWidget` does not paint a background
        # set from a stylesheet, so the bar's own margins and spacing stayed the window's
        # darker base colour -- measured (30,32,39) against the strip's (38,41,48) --
        # three thin vertical gaps through what is meant to be one continuous bar. The
        # two scroll areas inside carry the same sheet and always painted correctly,
        # because `QScrollArea` is itself a `QFrame`.
        bar = QFrame()
        bar.setFrameShape(QFrame.NoFrame)
        bar.setObjectName("viewStrip")
        bar.setStyleSheet(VIEW_STRIP_STYLE)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 0, 8, 0)
        bar_layout.setSpacing(8)
        bar_layout.addWidget(self.modality_strip)
        bar_layout.addWidget(self._divider)
        bar_layout.addWidget(self._view_slot, stretch=1)
        bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return bar

    # ── lending the active page's view chips a place in the bar ──────────

    def _mount_view_strip(self) -> None:
        """Move the active page's view strip into the bar, beside the modality chips.

        Moved rather than copied. The strip is the page's -- it rebuilds itself as views
        are acquired in, and marks the one the next run would land in -- so building a
        second copy here would be a second thing to keep in step, which is the failure
        this whole tab exists to stop.
        """
        self._unmount_view_strip()
        overview = self._tabs[self._modality].overview
        strip = getattr(overview, "view_strip", None)
        if strip is None:
            self._divider.setVisible(False)
            return
        owner = strip.parentWidget()
        if owner is not None and owner.layout() is not None:
            owner.layout().removeWidget(strip)
        self._view_slot_layout.addWidget(strip)
        strip.show()
        # Remembered as a pair: the strip alone is not enough to put it back, and the
        # widget alone is not enough to know whether anything was taken.
        self._mounted = (overview, strip)
        self._divider.setVisible(True)

    def _unmount_view_strip(self) -> None:
        """Hand the strip back before its page is rebuilt or destroyed.

        Without this the bar would hold a child of a widget Qt has deleted: the strip is
        parented *here* while mounted, so it does not go with its owner, and the next
        thing to touch it would be reading a dead C++ object. The same shape of problem
        as the lamella list in `AutoLamellaOverviewTabBase._drop_overview`, and handled
        the same way -- given back rather than left to be collected.

        `RuntimeError` is caught rather than prevented because the page may already be
        gone by the time this runs -- a test dropping a widget directly does exactly
        that -- and in that case there is nothing to hand it back to.
        """
        if self._mounted is None:
            return
        owner, strip = self._mounted
        self._mounted = None
        self._divider.setVisible(False)
        try:
            self._view_slot_layout.removeWidget(strip)
            strip.setParent(owner)
            strip.hide()
        except RuntimeError:
            # The page's widget was destroyed and took the strip with it.
            pass

    def modality_chip(self, modality: str) -> QPushButton:
        """The chip for *modality*, for a host or a test that wants to read its state.

        Public because "why can I not reach fluorescence" is now answered on the chip
        rather than on the tab, and the answer has to be readable from outside.
        """
        return self._chips[modality]

    def _on_chip_clicked(self, modality: str) -> None:
        self.set_modality(modality)

    def _refresh_chips(self) -> None:
        """Put each chip in the state its modality is actually in.

        Called after anything that can change availability, and it is what makes the
        strip honest: a checked chip means "this is what you are looking at", a disabled
        one means "this system cannot be reached, and here is why".
        """
        for modality, chip in self._chips.items():
            available = self._tabs[modality].is_available
            chip.setEnabled(available)
            chip.setChecked(available and modality == self._modality)
            chip.setToolTip(
                MODALITY_LABELS[modality]
                if available
                else self.unavailable_reason(modality)
            )

    # ── modality ─────────────────────────────────────────────────────────

    @property
    def modality(self) -> str:
        """Which imaging system this tab is showing."""
        return self._modality

    def available_modalities(self) -> List[str]:
        return [m for m in MODALITY_ORDER if self._tabs[m].is_available]

    def set_modality(self, modality: str) -> bool:
        """Show *modality*, if there is anything behind it.

        Refuses rather than showing an empty page: an unavailable modality has no widget
        at all -- `refresh_microscope` dropped it -- so raising its page would put a bare
        container on screen with no way to tell that from a canvas that failed to draw.
        The chip for it is disabled, so this is a guard rather than a path a click takes.
        """
        if modality not in self._tabs:
            raise ValueError(f"unknown overview modality: {modality}")
        if not self._tabs[modality].is_available:
            self._refresh_chips()
            return False
        changed = modality != self._modality
        self._modality = modality
        self.stack.setCurrentWidget(self._tabs[modality])
        self._mount_view_strip()
        self._refresh_chips()
        if changed:
            self.modality_changed.emit(modality)
        return True

    def unavailable_reason(self, modality: str) -> str:
        """Why *modality* cannot be reached, in the user's terms.

        Three absences that look identical on a greyed chip and are not: no microscope at
        all is about to change, no fluorescence detector is a fact about this system, and
        a flag that is off is the user's own setting. `availability_changed` is a bool and
        cannot carry the difference; this object holds the microscope, so it is worked out
        here.
        """
        microscope = (
            self.autolamella_ui.microscope if self.autolamella_ui is not None else None
        )
        if microscope is None:
            return "Connect a microscope to use the Overview"
        if modality == MODALITY_FLUORESCENCE:
            return "No Fluorescence Microscope Available"
        # Every system has beams, so with a microscope connected there is no second way
        # for this one to be unavailable -- only a widget that failed to build, which
        # `refresh_microscope` has already logged.
        return "The FIB/SEM overview is unavailable"

    # ── what the window asks of a tab ────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return any(tab.is_available for tab in self._tabs.values())

    @property
    def is_acquiring(self) -> bool:
        return any(tab.is_acquiring for tab in self._tabs.values())

    def refresh_microscope(self) -> None:
        """Rebuild or drop both host tabs to match the instrument.

        Both, on every connection, regardless of which one is showing: the hidden tab
        holds its microscope for life, so one left unrefreshed would go on reading
        geometry from an instrument nobody is driving the moment it was raised.
        """
        # Before the rebuilds, not after: `refresh_microscope` drops the old widget,
        # and a strip still mounted here would be left pointing at it.
        self._unmount_view_strip()
        for tab in self._tabs.values():
            tab.refresh_microscope()
        self._settle_modality()
        self._mount_view_strip()

    def refresh_experiment(self) -> None:
        for tab in self._tabs.values():
            tab.refresh_experiment()

    def refresh_positions(self) -> None:
        for tab in self._tabs.values():
            tab.refresh_positions()

    def set_selected(self, lamella) -> None:
        for tab in self._tabs.values():
            tab.set_selected(lamella)

    def set_interactive(self, enabled: bool, reason: str = "") -> None:
        for tab in self._tabs.values():
            tab.set_interactive(enabled, reason)

    # ── availability ─────────────────────────────────────────────────────

    def _on_tab_availability(self, modality: str, available: bool) -> None:
        self._settle_modality()

    def _on_tab_acquiring(self, _: bool) -> None:
        self.acquiring_changed.emit(self.is_acquiring)

    def _settle_modality(self) -> None:
        """Keep the shown modality one that exists, and tell the host what is left.

        The case this covers is a system with no FM, whose fluorescence chip never
        becomes available -- and the window between a connection and the beam widget
        being built, where neither is. Falling back rather than showing an empty page,
        and in `MODALITY_ORDER` so the fallback is not whichever tab happened to answer
        last.
        """
        available = self.available_modalities()
        if available and self._modality not in available:
            self.set_modality(available[0])
        else:
            self._refresh_chips()
        self.availability_changed.emit(bool(available))

    def unavailable_summary(self) -> Tuple[bool, str]:
        """Whether the tab is worth opening, and what to say on it when it is not.

        Only reached when neither modality is available, so the reason is the one for the
        modality a user would otherwise be in.
        """
        available = self.available_modalities()
        if available:
            return True, ""
        return False, self.unavailable_reason(self._modality)
