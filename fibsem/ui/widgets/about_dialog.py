"""
About dialog: what version, what commit, what microscope, what environment.

Its job is to answer "what am I actually running" well enough to paste into a
bug report, so every value is selectable and the copy button in the header puts
the whole thing on the clipboard as plain text.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.versioning import get_branch, get_revision
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import SECONDARY_BUTTON_STYLESHEET

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

# Palette (matching fibsem.ui.stylesheets napari theme)
_BG = "#262930"
_PANEL = "#1e2027"
_BORDER = "#3d4251"
_TEXT = "#d6d6d6"
_TEXT_STRONG = "#f0f1f2"
_TEXT_MUTED = "#868e93"
_MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"

# Characters of the serial number left visible; the rest is masked so a
# screenshot or clipboard paste does not disclose the full number.
_SERIAL_VISIBLE_CHARS = 4

_UNKNOWN = "Unknown"


def mask_serial_number(serial: Optional[str]) -> str:
    """Mask all but the leading characters of a serial number.

    Enough of the prefix survives to tell two machines apart in a support
    thread, without publishing the whole number. Separators are preserved so the
    result keeps the shape of the original (``8372-1194`` -> ``8372-****``).
    """
    if not serial:
        return _UNKNOWN
    serial = str(serial).strip()
    if not serial:
        return _UNKNOWN
    if len(serial) <= _SERIAL_VISIBLE_CHARS:
        return "*" * len(serial)
    masked = [
        char if char.isalnum() is False else "*"
        for char in serial[_SERIAL_VISIBLE_CHARS:]
    ]
    return serial[:_SERIAL_VISIBLE_CHARS] + "".join(masked)


def _environment_rows() -> List[Tuple[str, str]]:
    """Python / Qt / napari / platform, reusing the bug report's collector.

    Imported lazily and defensively: this dialog must still open if the
    autolamella application package is unavailable for any reason.
    """
    try:
        from fibsem.applications.autolamella.tools.bug_report import (
            collect_system_context,
        )

        ctx = collect_system_context()
    except Exception:
        import platform

        ctx = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

    rows = [
        ("Python", ctx.get("python_version", _UNKNOWN)),
        ("Qt", ctx.get("PyQt5", _UNKNOWN)),
        ("napari", ctx.get("napari", _UNKNOWN)),
        ("Platform", ctx.get("platform", _UNKNOWN)),
    ]
    return [(k, v) for k, v in rows if v != _UNKNOWN] or rows


class AboutDialog(QDialog):
    """Modal 'about' dialog for fibsemOS and the application hosting it."""

    def __init__(
        self,
        microscope: Optional["FibsemMicroscope"] = None,
        application: str = "fibsemOS",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("About fibsemOS")
        self.setStyleSheet(f"QDialog {{ background-color: {_BG}; }}")
        self.setMinimumWidth(470)

        self._sections: List[Tuple[str, List[Tuple[str, str]]]] = []

        import fibsem

        version = getattr(fibsem, "__version__", None) or _UNKNOWN
        revision = get_revision()
        branch = get_branch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        layout.addLayout(self._build_header())

        meta = QLabel(f"{version}   ·   {revision}" if revision else version)
        meta.setStyleSheet(
            f"color: {_TEXT_MUTED}; font-size: 12px; font-family: {_MONO};"
        )
        meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(meta)

        software = [("Application", application)]
        # Only meaningful for a source install; a wheel install has neither.
        if revision:
            software.append(("Revision", revision))
        if branch:
            software.append(("Branch", branch))
        layout.addWidget(
            self._build_section(
                "Software", software, mono_keys={"Revision", "Branch"}
            )
        )

        microscope_rows = self._microscope_rows(microscope)
        if microscope_rows:
            layout.addWidget(
                self._build_section(
                    "Microscope",
                    microscope_rows,
                    mono_keys={"Serial number", "Firmware", "Software"},
                )
            )

        layout.addWidget(
            self._build_section(
                "Environment",
                _environment_rows(),
                mono_keys={"Python", "Qt", "napari", "Platform"},
            )
        )

        layout.addSpacing(2)
        footer = QHBoxLayout()
        footer.addStretch()
        close_button = QPushButton("Close")
        close_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

    # -- construction helpers ------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        title = QLabel("fibsemOS")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {_TEXT_STRONG};"
        )
        header.addWidget(title)
        header.addStretch()

        self.copy_button = QToolButton()
        self.copy_button.setIcon(fibsem_icon("mdi:content-copy", color=_TEXT_MUTED))
        self.copy_button.setToolTip("Copy details to clipboard")
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.setAutoRaise(True)
        self.copy_button.setStyleSheet(
            "QToolButton { border: none; padding: 4px; border-radius: 3px; }"
            f"QToolButton:hover {{ background-color: {_PANEL}; }}"
        )
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        header.addWidget(self.copy_button, 0, Qt.AlignTop)
        return header

    @staticmethod
    def _microscope_rows(
        microscope: Optional["FibsemMicroscope"],
    ) -> List[Tuple[str, str]]:
        """Rows for the connected microscope, or empty when there is none.

        Never raises: an About dialog that cannot open because the microscope is
        in a bad state is worse than one missing a section.
        """
        if microscope is None:
            return []
        try:
            info = microscope.system.info
        except Exception:
            return []
        return [
            ("Name", str(getattr(info, "name", _UNKNOWN))),
            ("Manufacturer", str(getattr(info, "manufacturer", _UNKNOWN))),
            ("Model", str(getattr(info, "model", _UNKNOWN))),
            ("Serial number", mask_serial_number(getattr(info, "serial_number", None))),
            ("Firmware", str(getattr(info, "hardware_version", _UNKNOWN))),
            ("Software", str(getattr(info, "software_version", _UNKNOWN))),
        ]

    def _build_section(
        self, title: str, rows: List[Tuple[str, str]], mono_keys=frozenset()
    ) -> QWidget:
        self._sections.append((title, rows))

        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        heading = QLabel(title.upper())
        heading.setStyleSheet(
            f"color: {_TEXT_MUTED}; font-size: 10px; font-weight: 600;"
            " letter-spacing: 1px;"
        )
        column.addWidget(heading)

        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setStyleSheet(
            f"color: {_BORDER}; background-color: {_BORDER}; max-height: 1px;"
        )
        column.addWidget(rule)

        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        for row, (key, value) in enumerate(rows):
            key_label = QLabel(key)
            key_label.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
            value_label = QLabel(value)
            font = (
                f"font-family: {_MONO}; font-size: 11px;"
                if key in mono_keys
                else "font-size: 12px;"
            )
            value_label.setStyleSheet(f"color: {_TEXT}; {font}")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(key_label, row, 0, Qt.AlignTop)
            grid.addWidget(value_label, row, 1, Qt.AlignTop)
        column.addLayout(grid)
        return box

    # -- copy ----------------------------------------------------------------

    def as_text(self) -> str:
        """Everything on screen, as plain text.

        Exactly what is displayed, so the masked serial number stays masked.
        """
        lines: List[str] = []
        for title, rows in self._sections:
            if lines:
                lines.append("")
            lines.append(f"{title}:")
            width = max((len(key) for key, _ in rows), default=0)
            for key, value in rows:
                lines.append(f"  {key.ljust(width)}  {value}")
        return "\n".join(lines)

    def _copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is None:  # pragma: no cover - no display
            return
        clipboard.setText(self.as_text())
        self.copy_button.setToolTip("Copied")


def open_about_dialog(
    microscope: Optional["FibsemMicroscope"] = None,
    application: str = "fibsemOS",
    parent: Optional[QWidget] = None,
) -> None:
    """Show the about dialog modally."""
    AboutDialog(microscope=microscope, application=application, parent=parent).exec_()
