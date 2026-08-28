"""The Movement tab: the panels an operator moves the stage from.

A container. It lays out four things -- the position readout, the move actions, the
saved-position list and, when the flag is on, the sample holder -- and forwards the
handful of calls other windows make on the tab.

The two halves it composes are separately owned (FIB-783):

* ``StagePositionWidget`` says where the stage is and reads back what has been typed.
  It has no host contract at all.
* ``StageControlWidget`` moves the stage, and carries every coupling that requires:
  the image widget, the milling widget and the quad-view controller.

**The forwarding surface is not decoration.** Six call sites in four files reach a
Movement tab, and they reach it as ``movement_widget`` -- ``move_to_position`` from both
minimaps, the lamella list and AutoLamella's pose move; ``update_ui`` from
AutoLamella's stage-position signal; ``_teardown_connections`` from both windows'
disconnect paths; and ``_toggle_interactions`` from ``FibsemImageSettingsWidget``, which
reaches back through the same attribute. Those names stay here.
"""

from typing import Optional

from PyQt5 import QtWidgets

from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.widgets.custom_widgets import TitledPanel

# ACQUIRING_IMAGES and INSTRUCTIONS_TEXT are re-exported: they read as the Movement
# tab's vocabulary rather than one widget's internals, and callers already import them
# from here.
from fibsem.ui.widgets.stage_control_widget import (  # noqa: F401  (re-export)
    ACQUIRING_IMAGES,
    INSTRUCTIONS_TEXT,
    StageControlWidget,
)
from fibsem.ui.widgets.stage_position_widget import StagePositionWidget


class FibsemMovementWidget(QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope,
        parent: QtWidgets.QWidget,
    ):
        super().__init__(parent=parent)
        self.parent = parent

        if not hasattr(parent, "image_widget") or not isinstance(
            parent.image_widget, FibsemImageSettingsWidget
        ):
            raise ValueError(
                "Parent must have an 'image_widget' attribute of type FibsemImageSettingsWidget"
            )

        self.microscope = microscope
        self.image_widget: FibsemImageSettingsWidget = parent.image_widget
        self._setup_ui()
        self.setup_connections()

    def _setup_ui(self):
        # Outer layout
        self.gridLayout = QtWidgets.QGridLayout(self)

        # Scroll area
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 2)

        # --- Panel: Stage Movement ---
        # The readout above the actions, in one panel, as they have always been shown.
        stage_content = QtWidgets.QWidget()
        self.gridLayout_3 = QtWidgets.QGridLayout(stage_content)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)

        self.position_widget = StagePositionWidget(microscope=self.microscope)
        self.control_widget = StageControlWidget(
            microscope=self.microscope,
            position_widget=self.position_widget,
            host=self.parent,
        )
        self.gridLayout_3.addWidget(self.position_widget, 0, 0)
        self.gridLayout_3.addWidget(self.control_widget, 1, 0)

        # The form's button, in this panel's header rather than in the form's own grid
        # -- the panel chrome is the container's.
        self.stage_panel = TitledPanel(
            "Stage Movement", content=stage_content, collapsible=False
        )
        self.stage_panel.add_header_widget(self.position_widget.btn_refresh)
        self.gridLayout_2.addWidget(self.stage_panel, 0, 0)

        # Options panel removed — movement acquisition prefs are now in Edit > Preferences

        # --- Panel: Saved Positions ---
        from fibsem.ui.widgets.saved_position_widget import SavedPositionListWidget

        self.saved_positions_widget = SavedPositionListWidget(microscope=None)
        self.saved_positions_panel = TitledPanel(
            "Saved Positions", content=self.saved_positions_widget, collapsible=True
        )
        self.gridLayout_2.addWidget(self.saved_positions_panel, 2, 0)

        # Bottom spacer (row 4 — row 3 reserved for optional sample holder widget)
        self.gridLayout_2.addItem(
            QtWidgets.QSpacerItem(
                20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            ),
            4,
            0,
        )

    def setup_connections(self):
        # The form asks for a refresh; the control half is the one allowed to read the
        # device for it.
        self.position_widget.refresh_requested.connect(
            lambda: self.control_widget.update_ui(None)
        )

        # saved positions
        self.saved_positions_widget.microscope = self.microscope
        self.saved_positions_widget._header.btn_add.setEnabled(True)
        self.saved_positions_widget._load_default_positions()
        self.saved_positions_widget.move_to_requested.connect(self.move_to_position)

        if cfg.FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED:
            from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget

            self.sample_holder_widget = SampleHolderWidget(microscope=self.microscope)
            self.sample_holder_widget.set_holder(self.microscope._stage.holder)
            self.gridLayout_2.addWidget(self.sample_holder_widget, 3, 0)

        self.control_widget.update_ui()

    # --- the surface other windows reach for ---------------------------------

    def move_to_position(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> None:
        """Move the stage. ``None`` means the position typed into the form."""
        self.control_widget.move_to_position(stage_position)

    def update_ui(self, stage_position: Optional[FibsemStagePosition] = None) -> None:
        """Show *stage_position*, or read the stage when it is None."""
        self.control_widget.update_ui(stage_position)

    def get_position_from_ui(self) -> FibsemStagePosition:
        """What is currently typed into the form."""
        return self.position_widget.get_position()

    def _toggle_interactions(self, enable: bool, caller: Optional[str] = None) -> None:
        """Enable or disable the move actions. ``FibsemImageSettingsWidget`` calls this
        with ``caller="ui"`` to stop the handshake bouncing back."""
        self.control_widget._toggle_interactions(enable, caller=caller)

    def _teardown_connections(self) -> None:
        """Drop the canvas double-click registrations before this tab is destroyed.

        The canvases outlive it, and a stale double-click on a deleted widget makes
        PyQt abort the process (FIB-329). Idempotent.
        """
        self.control_widget._teardown_connections()
