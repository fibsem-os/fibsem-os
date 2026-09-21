"""The pose rows' actions, for the two windows that host a `SelectedLamellaWidget`.

The main window and the coincidence viewer both show a lamella's pose rows and both
answer their *Derive* button. It is one action -- ask, overwrite, say what happened --
so it lives here once rather than in each host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtWidgets import QMessageBox, QWidget

from fibsem.applications.autolamella.poses import (
    POSE_NOUNS,
    derivation_question,
    derive_pose,
    other_pose,
    pose_disagreement,
)
from fibsem.ui import notification_service

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.applications.autolamella.ui.selected_lamella_widget import (
        SelectedLamellaWidget,
    )
    from fibsem.microscope import FibsemMicroscope


def fluorescence_orientations(microscope: Optional["FibsemMicroscope"]) -> List[str]:
    """The orientations the FM declares it images from, for the derive menu."""
    fm = getattr(microscope, "fm", None)
    return list(fm.acquisition_orientations) if fm is not None else []


def show_lamella_poses(
    widget: "SelectedLamellaWidget",
    microscope: Optional["FibsemMicroscope"],
    lamella: Optional["Lamella"],
) -> None:
    """Show *lamella* in the panel, with what only the host can work out for it:
    the orientations a fluorescence pose may be derived into, and how far apart the
    two poses are. Arithmetic on stored positions; the instrument is not asked."""
    widget.set_fluorescence_orientations(fluorescence_orientations(microscope))
    widget.set_lamella(lamella)
    widget.set_pose_disagreement(
        pose_disagreement(microscope, lamella) if lamella is not None else None
    )


def derive_lamella_pose(
    parent: QWidget,
    microscope: Optional["FibsemMicroscope"],
    experiment: Optional["Experiment"],
    lamella: Optional["Lamella"],
    pose_name: str,
    orientation: Optional[str] = None,
) -> bool:
    """The *Derive* button: overwrite one pose with one worked out from the other.

    Every derivation overwrites, so every one confirms. Saves and announces the
    change; the caller redraws its own panel. Returns True if a pose was written.
    """
    if microscope is None:
        notification_service.show_toast("No microscope connected.", "warning")
        return False
    if lamella is None:
        notification_service.show_toast("No lamella selected.", "warning")
        return False
    if pose_name not in POSE_NOUNS:
        return False
    noun, source = POSE_NOUNS[pose_name], POSE_NOUNS[other_pose(pose_name)]
    if lamella.poses.get(other_pose(pose_name)) is None:
        notification_service.show_toast(
            f"{lamella.name} has no {source} to derive from.", "warning"
        )
        return False

    ret = QMessageBox.question(
        parent,
        "Derive Pose",
        derivation_question(lamella, pose_name, orientation),
        QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
    )
    if ret != QMessageBox.Yes:  # type: ignore[attr-defined]
        return False

    if not derive_pose(microscope, lamella, pose_name, orientation):
        notification_service.show_toast(
            f"Could not derive the {noun} of {lamella.name}; it is left as it was.",
            "error",
        )
        return False

    if experiment is not None:
        experiment.save()
        # A pose written in place emits nothing on its own; the overview canvases
        # re-mark from this.
        experiment.positions.events.changed.emit()
    notification_service.show_toast(f"Derived the {noun} of {lamella.name}.", "info")
    return True
