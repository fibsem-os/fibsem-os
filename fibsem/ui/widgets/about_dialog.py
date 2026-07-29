"""
About dialog: what version, what commit, what microscope, what environment.

Its job is to answer "what am I actually running" well enough to paste into a
bug report, so every value is selectable and the copy button in the header puts
the whole thing on the clipboard as plain text.
"""
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt5.QtCore import QSize, Qt
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
from fibsem.ui.widgets.custom_widgets import IconToolButton, _SpinnerLabel

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

# Palette (matching fibsem.ui.stylesheets napari theme)
_BG = "#262930"
_PANEL = "#1e2027"
_BORDER = "#3d4251"
_TEXT = "#d6d6d6"
_TEXT_STRONG = "#f0f1f2"
_TEXT_MUTED = "#868e93"
_ACCENT = "#50a6ff"
_OK = "#4caf50"
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
        # Set once the dialog is dismissed, so a still-running update check can
        # tell that there is nothing left to render into.
        self._closed = False

        import fibsem

        version = getattr(fibsem, "__version__", None) or _UNKNOWN
        revision = get_revision()
        branch = get_branch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        layout.addLayout(self._build_header())

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(7)
        meta = QLabel(f"{version}   ·   {revision}" if revision else version)
        meta.setStyleSheet(
            f"color: {_TEXT_MUTED}; font-size: 12px; font-family: {_MONO};"
        )
        meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meta_row.addWidget(meta)
        meta_row.addWidget(self._build_update_slot())
        meta_row.addStretch()
        layout.addLayout(meta_row)

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

    # -- the update slot -----------------------------------------------------
    #
    # One slot beside the version does three jobs in sequence: offer a manual
    # check, report it in flight, then show the outcome. What it starts as
    # depends on whether the opt-in background check already ran.

    def _build_update_slot(self) -> QWidget:
        self._update_slot = QWidget()
        row = QHBoxLayout(self._update_slot)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        try:
            self._render_initial_update_state()
        except Exception:
            # A broken update check must never stop the dialog from opening.
            _logger.debug("Could not build the update slot", exc_info=True)
        return self._update_slot

    def _set_update_slot(self, *widgets: QWidget) -> None:
        row = self._update_slot.layout()
        while row.count():
            item = row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for widget in widgets:
            row.addWidget(widget)

    def _render_initial_update_state(self) -> None:
        from fibsem import update_check

        if not update_check.is_supported():
            return  # source install: a PyPI comparison would be meaningless

        latest = update_check.get_available_update()
        if latest:
            # The background check already found something actionable.
            self._render_update_available(latest)
        else:
            # Deliberately not "up to date" — the background check reports only
            # the actionable case. "Up to date" is an answer to a question, and
            # nobody has asked one yet.
            self._render_check_button()

    def _render_check_button(self) -> None:
        button = IconToolButton(
            icon="mdi:refresh",
            color=_TEXT_MUTED,
            tooltip="Check for updates",
        )
        button.setIconSize(QSize(14, 14))
        button.clicked.connect(self._start_update_check)
        self._set_update_slot(button)

    def _render_update_available(self, latest: str) -> None:
        icon = QLabel()
        icon.setPixmap(
            fibsem_icon("mdi:arrow-up-circle", color=_ACCENT).pixmap(13, 13)
        )
        note = QLabel(f"{latest} available")
        note.setStyleSheet(f"color: {_ACCENT}; font-size: 12px;")
        note.setToolTip(f"fibsemOS {latest} is available on PyPI")
        self._set_update_slot(icon, note)

    def _render_up_to_date(self) -> None:
        icon = QLabel()
        icon.setPixmap(fibsem_icon("mdi:check-circle", color=_OK).pixmap(13, 13))
        # Muted text: the tick carries the signal, and this is a non-event.
        note = QLabel("up to date")
        note.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        self._set_update_slot(icon, note)

    def _render_check_failed(self) -> None:
        icon = QLabel()
        icon.setPixmap(
            fibsem_icon("mdi:cloud-off-outline", color=_TEXT_MUTED).pixmap(13, 13)
        )
        note = QLabel("could not check")
        note.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        note.setToolTip("Could not reach PyPI")
        # Keep the button: a failed check is exactly when you want to retry, and
        # a transient network blip should not remove the affordance.
        retry = IconToolButton(
            icon="mdi:refresh", color=_TEXT_MUTED, tooltip="Try again"
        )
        retry.setIconSize(QSize(14, 14))
        retry.clicked.connect(self._start_update_check)
        self._set_update_slot(icon, note, retry)

    def _start_update_check(self) -> None:
        """Run a user-requested check off the GUI thread."""
        from fibsem import update_check
        from fibsem.ui.qt.threading import thread_worker

        spinner = _SpinnerLabel(size=14, step_deg=30, color=_TEXT_MUTED)
        note = QLabel("checking…")
        note.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        self._set_update_slot(spinner, note)
        spinner.start()

        @thread_worker
        def _run():
            # force: the user clicked, so the opt-in preference does not apply.
            return update_check.refresh(force=True)

        worker = _run()
        worker.returned.connect(self._on_update_check_returned)
        worker.errored.connect(lambda _exc: self._on_update_check_returned(None))
        worker.start()

    def _on_update_check_returned(self, status) -> None:
        """Render the outcome, if the dialog is still around to render it into.

        The dialog is modal but closable, so the worker can outlive it. PyQt5
        calls qFatal on any exception escaping a slot (FIB-329), which makes
        touching a deleted C++ object here an process abort rather than a
        traceback — hence both guards.
        """
        if self._closed:
            return
        try:
            if status is None:
                self._render_check_failed()
            elif status.update_available:
                self._render_update_available(status.latest)
            else:
                self._render_up_to_date()
        except RuntimeError:
            _logger.debug("About dialog closed during the update check")

    def done(self, result: int) -> None:
        # accept(), reject() and the window's close button all funnel here.
        self._closed = True
        super().done(result)

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
