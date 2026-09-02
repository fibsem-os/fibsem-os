"""Microscope tab · Sample: the hardware view of the grids.

The holder on every machine (its slots, their calibration, which grid is in each,
Move, and the calibration wizard) and, when the system has an autoloader, the
magazine beneath it. The Grids tab is the experiment's view of the same grids;
this one is the instrument's, and it is where the twelve-row slot list lives.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget
from fibsem.ui.widgets.sample_loader_widget import SampleLoaderWidget


class FibsemSampleWidget(QWidget):
    # A request to drive to a calibrated holder slot, routed by the host through
    # the Movement widget so the readout and post-move images follow.
    move_to_requested = pyqtSignal(object)  # FibsemStagePosition

    def __init__(self, microscope, parent=None):
        super().__init__(parent)
        self.microscope = microscope
        stage = microscope._stage

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # Never wider than the tab: a long facts line wraps rather than pushing
        # the rows' right-hand column out of view.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(6, 6, 6, 6)
        inner_layout.setSpacing(8)

        self.holder_widget = SampleHolderWidget(microscope=microscope)
        self.holder_widget.set_holder(stage.holder)
        self.holder_widget.move_to_requested.connect(self.move_to_requested)
        inner_layout.addWidget(self.holder_widget)

        self.loader_widget: Optional[SampleLoaderWidget] = None
        if stage.loader is not None:
            self.loader_widget = SampleLoaderWidget(microscope=microscope)
            # An exchange changes what is in the holder's working slot.
            self.loader_widget.loader_changed.connect(self.holder_widget.refresh)
            inner_layout.addWidget(self.loader_widget)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def refresh(self) -> None:
        self.holder_widget.refresh()
        if self.loader_widget is not None:
            self.loader_widget.refresh()

    def set_controls_enabled(self, enabled: bool) -> None:
        """Lock the exchange controls while a workflow owns the loader."""
        if self.loader_widget is not None:
            self.loader_widget.set_controls_enabled(enabled)
