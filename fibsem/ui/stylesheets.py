# style sheets
# TODO: TEMPLATE THIS CSS
import os as _os
_ICONS_DIR = _os.path.join(_os.path.dirname(__file__), "icons").replace("\\", "/")

GREEN_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: green;
    }
QPushButton:hover {
    background-color: rgba(0, 255, 0, 125);
    }"""

RED_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: red;
    }
QPushButton:hover {
    background-color: rgba(255, 0, 0, 125);
    }"""

BLUE_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: blue;
    }
QPushButton:hover {
    background-color: rgba(0, 0, 255, 125);
    }"""

YELLOW_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: yellow;
    }
QPushButton:hover {
    background-color: rgba(255, 255, 0, 125);
    }"""

WHITE_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: white;
    color: black;
    }
QPushButton:hover {
    background-color: rgba(255, 255, 255, 125);
    }"""

GRAY_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: gray;
    }
QPushButton:hover {
    background-color: rgba(125, 125, 125, 125);
    }"""

ORANGE_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: orange;
    color: black;
    }
QPushButton:hover {
    background-color: rgba(255, 125, 0, 125);
}"""

DISABLED_PUSHBUTTON_STYLE = """
QPushButton {
    background-color: none;
    }
"""    

PROGRESS_BAR_GREEN_STYLE = "QProgressBar::chunk {background-color: green;}"
PROGRESS_BAR_BLUE_STYLE = "QProgressBar::chunk {background-color: blue;}"


CHECKBOX_STYLE = """
QCheckBox::indicator {
        width: 16px;"
        height: 16px;
        border: 1px solid rgba(220, 220, 220, 0.7);
        border-radius: 3px;
        background-color: rgba(255, 255, 255, 0.05);
        }
QCheckBox::indicator:hover {
        border: 1px solid rgba(120, 180, 255, 0.9);
}
QCheckBox::indicator:checked {
        background-color: #3a6ea5;
        border: 1px solid #68a0dd;
}"""

LABEL_INSTRUCTIONS_STYLE = """font-style: italic; color: gray; font-size: 12px;"""


# Napari-style dark theme stylesheet
NAPARI_STYLE = """
QWidget {
    background-color: #262930;
    color: #d6d6d6;
}

QMainWindow {
    background-color: #262930;
}

QMenuBar {
    background-color: #262930;
    color: #d6d6d6;
    border-bottom: 1px solid #3d4251;
}

QMenuBar::item {
    background-color: transparent;
    padding: 4px 10px;
}

QMenuBar::item:selected {
    background-color: #3d4251;
}

QMenu {
    background-color: #262930;
    color: #d6d6d6;
    border: 1px solid #3d4251;
}

QMenu::item:selected {
    background-color: #3d4251;
}

QTabWidget::pane {
    border: none;
    background-color: #262930;
}

QTabBar::tab {
    background-color: #1e2027;
    color: #d6d6d6;
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    background-color: #262930;
    border-bottom: 2px solid #50a6ff;
}

QTabBar::tab:hover:!selected {
    background-color: #2d313b;
}

QTabBar::tab:disabled {
    color: #6b6b6b;
}

QPushButton {
    background-color: #3d4251;
    color: #d6d6d6;
    border: none;
    padding: 5px 12px;
    border-radius: 3px;
}

QPushButton:hover {
    background-color: #4a5168;
}

QPushButton:pressed {
    background-color: #50a6ff;
}

QPushButton:disabled {
    background-color: #2d313b;
    color: #6b6b6b;
}

QStatusBar {
    background-color: #1e2027;
    color: #d6d6d6;
    border-top: 1px solid #3d4251;
}

QLabel {
    color: #d6d6d6;
}

QGroupBox {
    background-color: #262930;
    border: 1px solid #3d4251;
    border-radius: 3px;
    margin-top: 8px;
    padding-top: 16px;
    color: #d6d6d6;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: #d6d6d6;
}

QLineEdit {
    background-color: #1e2027;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    border-radius: 3px;
    padding: 4px 8px;
}

QLineEdit:focus {
    border: 1px solid #50a6ff;
}

QLineEdit:disabled {
    color: #6b6b6b;
    background-color: #2d313b;
}

QComboBox {
    background-color: #3d4251;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    border-radius: 3px;
    padding: 4px 8px;
}

QComboBox:hover {
    border: 1px solid #50a6ff;
}

QComboBox:focus {
    border: 1px solid #50a6ff;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: url("__ICONS_DIR__/chevron_down.svg");
    width: 10px;
    height: 10px;
    margin-right: 6px;
}

QComboBox:disabled {
    color: #6b6b6b;
    background-color: #2d313b;
}

QComboBox QAbstractItemView {
    background-color: #3d4251;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    selection-background-color: #4a5168;
}

QSpinBox, QDoubleSpinBox {
    background-color: #1e2027;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    border-radius: 3px;
    padding: 4px 8px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #50a6ff;
}

QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #6b6b6b;
    background-color: #2d313b;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: center right;
    background-color: #3d4251;
    border: none;
    border-left: 1px solid #3d4251;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    width: 20px;
    height: 100%;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #4a5168;
}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background-color: #50a6ff;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("__ICONS_DIR__/plus.svg");
    width: 10px;
    height: 10px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: center left;
    background-color: #3d4251;
    border: none;
    border-right: 1px solid #3d4251;
    border-top-left-radius: 3px;
    border-bottom-left-radius: 3px;
    width: 20px;
    height: 100%;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #4a5168;
}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #50a6ff;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("__ICONS_DIR__/minus.svg");
    width: 10px;
    height: 10px;
}

QSlider::groove:horizontal {
    background-color: #3d4251;
    height: 4px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background-color: #3d4251;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #4a5168;
}

QSlider::sub-page:horizontal {
    background-color: #3d4251;
    border-radius: 2px;
}

QCheckBox {
    color: #d6d6d6;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3d4251;
    border-radius: 3px;
    background-color: #1e2027;
}

QCheckBox::indicator:hover {
    border: 1px solid #50a6ff;
}

QCheckBox::indicator:checked {
    background-color: #3d4251;
    image: url("__ICONS_DIR__/check.svg");
}

QListWidget {
    background-color: #1e2027;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    border-radius: 3px;
}

QListWidget::item {
    padding: 4px;
    border-bottom: 1px solid #3d4251;
}

QListWidget::item:selected {
    background-color: #3d4251;
}

QListWidget::item:hover {
    background-color: #2d313b;
}

QToolButton {
    background-color: #3d4251;
    color: #d6d6d6;
    border: none;
    border-radius: 3px;
    padding: 4px;
}

QToolButton:hover {
    background-color: #4a5168;
}

QToolButton:pressed {
    background-color: #50a6ff;
}

QDialog {
    background-color: #262930;
    color: #d6d6d6;
}

QDialogButtonBox QPushButton {
    min-width: 70px;
}

QToolTip {
    background-color: #1e2027;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    padding: 4px 8px;
    border-radius: 3px;
}
""".replace("__ICONS_DIR__", _ICONS_DIR)

MILLING_PROGRESS_BAR_STYLESHEET = """
            QProgressBar {
                border: 1px solid #3d4251;
                border-radius: 3px;
                text-align: center;
                background-color: #1e2027;
                color: #d6d6d6;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
            }
        """

USER_ATTENTION_BUTTON_STYLESHEET = """
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
            QPushButton:pressed {
                background-color: #e65100;
            }
        """

RUN_WORKFLOW_BUTTON_STYLESHEET = """
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #2e7d32;
            }
            QPushButton:disabled {
                background-color: #2d313b;
                color: #6b6b6b;
            }
        """


STOP_WORKFLOW_BUTTON_STYLESHEET = """
            QPushButton {
                background-color: #99121F;
                color: white;
                border: none;
                padding: 5px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #BF2A38;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """

SUPERVISION_STATUS_SUPERVISED_STYLESHEET = """
                QPushButton {
                    background-color: #007ACC;
                    color: white;
                    border: none;
                    padding: 5px 12px;
                    border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976d2;
                }
                QPushButton:pressed {
                    background-color: #1565c0;
                }
            """

SUPERVISION_STATUS_AUTOMATED_STYLESHEET = """
                QPushButton {
                    background-color: #4caf50;
                    color: white;
                    border: none;
                    padding: 5px 12px;
                    border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #388e3c;
                }
                QPushButton:pressed {
                    background-color: #2e7d32;
                }
            """

PRIMARY_BUTTON_STYLESHEET = """
    QPushButton {
        background-color: #007ACC;
        color: white;
        border: none;
        padding: 5px 12px;
        border-radius: 3px;
    }
    QPushButton:hover {
        background-color: #118FE4;
    }
    QPushButton:pressed {
        background-color: #2d72c4;
    }
    QPushButton:disabled {
        background-color: #2d313b;
        color: #6b6b6b;
    }
"""

SECONDARY_BUTTON_STYLESHEET = """
    QPushButton {
        background-color: #3d4251;
        color: #d6d6d6;
        border: none;
        padding: 5px 12px;
        border-radius: 3px;
    }
    QPushButton:hover {
        background-color: #4a5168;
    }
    QPushButton:pressed {
        background-color: #50a6ff;
    }
    QPushButton:disabled {
        background-color: #2d313b;
        color: #6b6b6b;
    }
"""

STATUS_BAR_STYLESHEET = """
                QStatusBar {
                    background-color: #1e2027;
                        color: #d6d6d6;
                        border-top: 1px solid #3d4251;
                    }
                """


INDETERMINATE_PROGRESS_BAR_STYLESHEET = """
            QProgressBar {
                border: 1px solid #3d4251;
                border-radius: 3px;
                text-align: center;
                background-color: #1e2027;
                color: #d6d6d6;
            }
            QProgressBar::chunk {
                background-color: #50a6ff;
            }
        """

# Color palette
PRIMARY_COLOR = "#007ACC"
PRIMARY_COLOR_HOVER = "#118FE4"
PRIMARY_COLOR_PRESSED = "#2d72c4"
GRAY_CANVAS_COLOR = "#000000"
GRAY_CONSOLE_COLOR = "#121212"
GRAY_BACKGROUND_COLOR = "#262930"
GRAY_FOREGROUND_COLOR = "#414851"
GRAY_PRIMARY_COLOR = "#5a626C"
GRAY_HIGHLIGHT_COLOR = "#6A7380"
GRAY_SECONDARY_COLOR = "#868E93"
GRAY_ICON_COLOR = "#D1D2D4"
GRAY_TEXT_COLOR = "#F0F1F2"
GRAY_WHITE_COLOR = "#FFFFFF"
SEMANTIC_ERROR_COLOR = "#99121F"
SEMANTIC_ERROR_HOVER_COLOR = "#BF2A38"
SEMANTIC_ERROR_PRESSED_COLOR = "#b71c1c"
SEMANTIC_WARNING_COLOR = "#E3B617"
AUTOMATED_COLOR = "#4caf50"
PURPLE_COLOR = "#7C3AED"
GREEN_COLOR = "#4caf50"
RED_COLOR = "#99121F"
ORANGE_COLOR = " #ff9800"
DEFECT_ORANGE_COLOR = "#e8a020"
DEFECT_RED_COLOR = "#d04040"

WORKFLOW_BORDER_STYLESHEET = """
    QFrame#workflow_border_frame[borderState="idle"]       { border: 4px solid #262930; }
    QFrame#workflow_border_frame[borderState="automated"]  { border: 4px solid #4caf50; }
    QFrame#workflow_border_frame[borderState="supervised"] { border: 4px solid #007ACC; }
    QFrame#workflow_border_frame[borderState="waiting"]    { border: 4px solid #ff9800; }
    QFrame#workflow_border_frame[borderState="finished"]  { border: 4px solid #4caf50; }
    QFrame#workflow_border_frame[borderState="stopped"]   { border: 4px solid #99121F; }
    QFrame#workflow_border_frame[borderState="agent"]     { border: 4px solid #BF00FF; }
"""

TOOLBUTTON_ICON_STYLESHEET = """
    QToolButton {
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 2px 6px;
        background-color: transparent;
    }
    QToolButton:hover {
        border: 1px solid #6a6a6a;
        background-color: rgba(255, 255, 255, 25);
    }
    QToolButton:checked {
        border: 1px solid #8a8a8a;
        background-color: rgba(255, 255, 255, 35);
    }
"""

LIST_WIDGET_STYLESHEET = """
            QListWidget {
                background: #2b2d31;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background: #2b2d31;
                border-bottom: 1px solid #3a3d42;
            }
            QListWidget::item:selected {
                background: #2d3f5c;
            }
        """