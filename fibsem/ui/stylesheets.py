# style sheets
# TODO: TEMPLATE THIS CSS   
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
    background-color: #1e2027;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    border-radius: 3px;
    padding: 4px 8px;
}

QComboBox:hover {
    border: 1px solid #50a6ff;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #1e2027;
    color: #d6d6d6;
    border: 1px solid #3d4251;
    selection-background-color: #3d4251;
}

QCheckBox {
    color: #d6d6d6;
    spacing: 6px;
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
"""

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
        background-color: #1e2027;
        color: #d6d6d6;
        text-align: center;
        height: 6px;
    }
    QProgressBar::chunk {
        background-color: #50a6ff;
        border-radius: 3px;
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

WORKFLOW_BORDER_IDLE_STYLESHEET = "QFrame#workflow_border_frame { border: 4px solid #262930; }"
WORKFLOW_BORDER_AUTOMATED_STYLESHEET = "QFrame#workflow_border_frame { border: 4px solid #4caf50; }"
WORKFLOW_BORDER_SUPERVISED_STYLESHEET = "QFrame#workflow_border_frame { border: 4px solid #007ACC; }"
WORKFLOW_BORDER_WAITING_STYLESHEET = "QFrame#workflow_border_frame { border: 4px solid #ff9800; }"
