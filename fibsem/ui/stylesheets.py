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

QCheckBox::indicator {
    width: 16px;
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