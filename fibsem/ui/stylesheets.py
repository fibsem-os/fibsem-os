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
"""