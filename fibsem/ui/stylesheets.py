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