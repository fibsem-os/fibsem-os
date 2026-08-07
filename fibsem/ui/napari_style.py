"""The application-level napari-dark stylesheet.

Set once on the QApplication, unlike everything in ``fibsem.ui.stylesheets``,
which is a per-component sheet applied to a single widget. It was ~340 of that
module's ~825 lines and does a different job, so it lives on its own.

Imported and re-exported by ``fibsem.ui.stylesheets``, so both
``from fibsem.ui.stylesheets import NAPARI_STYLE`` and
``stylesheets.NAPARI_STYLE`` keep working.
"""
import os as _os

from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    DISABLED_BG_COLOR,
    DISABLED_TEXT_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
)

# Same directory as stylesheets.py, so this resolves identically; kept local
# rather than imported so this module does not depend on that one.
_ICONS_DIR = _os.path.join(_os.path.dirname(__file__), "icons").replace("\\", "/")

# Napari-style dark theme stylesheet
#
# TODO: no token -- #4a5168 (the pressed/selected lift on controls), and the two
# :hover rules still on #2d313b. That value now has a name -- DISABLED_BG_COLOR
# -- but hover is not disabled, and ROW_ALT_COLOR is the token that means hover;
# they differ by 4 units of RGB, so reconciling them is a real colour change and
# wants a decision rather than a silent swap.
NAPARI_STYLE = f"""
QWidget {{
    background-color: {SURFACE_COLOR};
    color: {TEXT_COLOR};
}}

QMainWindow {{
    background-color: {SURFACE_COLOR};
}}

QMenuBar {{
    background-color: {SURFACE_COLOR};
    color: {TEXT_COLOR};
    border-bottom: 1px solid {BORDER_COLOR};
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 4px 10px;
}}

QMenuBar::item:selected {{
    background-color: {BORDER_COLOR};
}}

QMenu {{
    background-color: {SURFACE_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
}}

QMenu::item:selected {{
    background-color: {BORDER_COLOR};
}}

/* Without this, `QMenu {{ color }}` above also paints disabled items, so a greyed
   action is indistinguishable from an available one. Matches the disabled
   colour used by the button and field rules below. */
QMenu::item:disabled {{
    color: {DISABLED_TEXT_COLOR};
}}

QTabWidget::pane {{
    border: none;
    background-color: {SURFACE_COLOR};
}}

QTabBar::tab {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}}

QTabBar::tab:selected {{
    background-color: {SURFACE_COLOR};
    border-bottom: 2px solid {ACCENT_COLOR};
}}

QTabBar::tab:hover:!selected {{
    background-color: #2d313b;
}}

QTabBar::tab:disabled {{
    color: {DISABLED_TEXT_COLOR};
}}

QPushButton {{
    background-color: {BORDER_COLOR};
    color: {TEXT_COLOR};
    border: none;
    padding: 5px 12px;
    border-radius: 3px;
}}

QPushButton:hover {{
    background-color: #4a5168;
}}

QPushButton:pressed {{
    background-color: {ACCENT_COLOR};
}}

QPushButton:disabled {{
    background-color: {DISABLED_BG_COLOR};
    color: {DISABLED_TEXT_COLOR};
}}

QStatusBar {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border-top: 1px solid {BORDER_COLOR};
}}

QLabel {{
    color: {TEXT_COLOR};
}}

QGroupBox {{
    background-color: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    margin-top: 8px;
    padding-top: 16px;
    color: {TEXT_COLOR};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {TEXT_COLOR};
}}

QLineEdit {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 4px 8px;
}}

QLineEdit:focus {{
    border: 1px solid {ACCENT_COLOR};
}}

QLineEdit:disabled {{
    color: {DISABLED_TEXT_COLOR};
    background-color: {DISABLED_BG_COLOR};
}}

QComboBox {{
    background-color: {BORDER_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 4px 8px;
}}

QComboBox:hover {{
    border: 1px solid {ACCENT_COLOR};
}}

QComboBox:focus {{
    border: 1px solid {ACCENT_COLOR};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: url("__ICONS_DIR__/chevron_down.svg");
    width: 10px;
    height: 10px;
    margin-right: 6px;
}}

QComboBox:disabled {{
    color: {DISABLED_TEXT_COLOR};
    background-color: {DISABLED_BG_COLOR};
}}

QComboBox QAbstractItemView {{
    background-color: {BORDER_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: #4a5168;
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 4px 8px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT_COLOR};
}}

QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {DISABLED_TEXT_COLOR};
    background-color: {DISABLED_BG_COLOR};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: center right;
    background-color: {BORDER_COLOR};
    border: none;
    border-left: 1px solid {BORDER_COLOR};
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    width: 20px;
    height: 100%;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background-color: #4a5168;
}}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
    background-color: {ACCENT_COLOR};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("__ICONS_DIR__/plus.svg");
    width: 10px;
    height: 10px;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: center left;
    background-color: {BORDER_COLOR};
    border: none;
    border-right: 1px solid {BORDER_COLOR};
    border-top-left-radius: 3px;
    border-bottom-left-radius: 3px;
    width: 20px;
    height: 100%;
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #4a5168;
}}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background-color: {ACCENT_COLOR};
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("__ICONS_DIR__/minus.svg");
    width: 10px;
    height: 10px;
}}

QSlider::groove:horizontal {{
    background-color: {BORDER_COLOR};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {BORDER_COLOR};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background-color: #4a5168;
}}

QSlider::sub-page:horizontal {{
    background-color: {BORDER_COLOR};
    border-radius: 2px;
}}

QCheckBox {{
    color: {TEXT_COLOR};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    background-color: {PANEL_COLOR};
}}

QCheckBox::indicator:hover {{
    border: 1px solid {ACCENT_COLOR};
}}

QCheckBox::indicator:checked {{
    background-color: {BORDER_COLOR};
    image: url("__ICONS_DIR__/check.svg");
}}

QListWidget {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
}}

QListWidget::item {{
    padding: 4px;
    border-bottom: 1px solid {BORDER_COLOR};
}}

QListWidget::item:selected {{
    background-color: {BORDER_COLOR};
}}

QListWidget::item:hover {{
    background-color: #2d313b;
}}

QToolButton {{
    background-color: {BORDER_COLOR};
    color: {TEXT_COLOR};
    border: none;
    border-radius: 3px;
    padding: 4px;
}}

QToolButton:hover {{
    background-color: #4a5168;
}}

QToolButton:pressed {{
    background-color: {ACCENT_COLOR};
}}

QDialog {{
    background-color: {SURFACE_COLOR};
    color: {TEXT_COLOR};
}}

QDialogButtonBox QPushButton {{
    min-width: 70px;
}}

QToolTip {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    padding: 4px 8px;
    border-radius: 3px;
}}
""".replace("__ICONS_DIR__", _ICONS_DIR)
