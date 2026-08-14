# style sheets
import os as _os

# The colour palette lives in fibsem.ui.tokens and is imported here, at the top,
# so the QSS constants below can interpolate it. Every name is re-exported for
# the callers that import colours from this module rather than from tokens.
from fibsem.ui.tokens import (  # noqa: F401  (re-exported for existing callers)
    ACCENT_COLOR,
    AUTOMATED_COLOR,
    BORDER_COLOR,
    CANVAS_BG,
    DEFECT_ORANGE_COLOR,
    DISABLED_BG_COLOR,
    DISABLED_TEXT_COLOR,
    DEFECT_RED_COLOR,
    ERROR_COLOR,
    GRAY_BACKGROUND_COLOR,
    GRAY_CANVAS_COLOR,
    GRAY_CONSOLE_COLOR,
    GRAY_FOREGROUND_COLOR,
    GRAY_HIGHLIGHT_COLOR,
    GRAY_ICON_COLOR,
    GRAY_PRIMARY_COLOR,
    GRAY_SECONDARY_COLOR,
    GRAY_TEXT_COLOR,
    GRAY_WHITE_COLOR,
    GREEN_COLOR,
    OK_COLOR,
    ORANGE_COLOR,
    PANEL_COLOR,
    PRIMARY_ACCENT,
    PRIMARY_COLOR,
    PRIMARY_COLOR_HOVER,
    PRIMARY_COLOR_PRESSED,
    PURPLE_COLOR,
    RED_COLOR,
    ROW_ALT_COLOR,
    SEMANTIC_ERROR_COLOR,
    SEMANTIC_ERROR_HOVER_COLOR,
    SEMANTIC_ERROR_PRESSED_COLOR,
    SEMANTIC_WARNING_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
    WHITE_ICON_COLOR,
)

# The application-level sheet moved to its own module; re-exported so that both
# `from fibsem.ui.stylesheets import NAPARI_STYLE` and `stylesheets.NAPARI_STYLE`
# keep resolving.
from fibsem.ui.napari_style import NAPARI_STYLE  # noqa: F401
from fibsem.ui.tokens import (
    NEUTRAL_300,
    NEUTRAL_650,
    SURFACE_COLOR,
)

_ICONS_DIR = _os.path.join(_os.path.dirname(__file__), "icons").replace("\\", "/")

LABEL_INSTRUCTIONS_STYLE = """font-style: italic; color: gray; font-size: 12px;"""

# Filename header sitting above an image canvas. Lived in
# fm_image_display_widget until that module was retired (FIB-535); it describes a
# label, not an FM display, so this is where it belongs.
#
# The border keeps its literal rather than becoming BORDER_COLOR (#3d4251) --
# they are different colours, #3a3d42 is used raw in 39 other places, and a move
# is not the place to change one. Tokenising it is separate work.
IMAGE_HEADER_STYLE = (
    f"color: {NEUTRAL_300}; font-size: 11px; padding: 3px 8px; "
    f"background: {CANVAS_BG}; border-bottom: 1px solid #3a3d42;"
)

# A small plaque drawn *over* a canvas -- the cursor position readout on both overview
# tabs. Dark and translucent rather than the app's panel styling, because it sits on
# data and must not read as chrome; monospace so the digits do not shuffle the label
# sideways on every pixel of pointer travel.
#
# Here because it was written out twice, byte for byte, once per overview tab. Two
# copies of a style that has to match is the same failure as two copies of a colour.
#
# #e8e8e8 keeps its literal rather than becoming NEUTRAL_200 (#e0e0e0): they are
# different colours, and as with IMAGE_HEADER_STYLE above, a move is not the place to
# change one.
CANVAS_READOUT_STYLE = (
    "color: #e8e8e8; font-size: 10px; font-family: monospace;"
    "background: rgba(26, 26, 26, 160); border-radius: 3px; padding: 2px 5px;"
)


PROGRESS_BAR_STYLESHEET = f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                text-align: center;
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
            }}
            QProgressBar::chunk {{
                background-color: {OK_COLOR};
            }}
        """

# same bar, error-coloured chunk: rendered when an operation finishes in a failed
# state, so "Done" and "Failed" are not both a full green bar.
FAILED_PROGRESS_BAR_STYLESHEET = f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                text-align: center;
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
            }}
            QProgressBar::chunk {{
                background-color: {SEMANTIC_ERROR_COLOR};
            }}
        """

# TODO: no token -- #e65100, #f57c00
# (the hover and pressed shades of ORANGE_COLOR)
USER_ATTENTION_BUTTON_STYLESHEET = f"""
            QPushButton {{
                background-color: {ORANGE_COLOR};
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #f57c00;
            }}
            QPushButton:pressed {{
                background-color: #e65100;
            }}
        """

# TODO: no token -- #2e7d32, #388e3c
CONFIRM_BUTTON_STYLESHEET = f"""
            QPushButton {{
                background-color: {OK_COLOR};
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #388e3c;
            }}
            QPushButton:pressed {{
                background-color: #2e7d32;
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_BG_COLOR};
                color: {DISABLED_TEXT_COLOR};
            }}
        """


# The :disabled rule matches the primary/secondary/run sheets. Without it a
# disabled stop or delete button stays full crimson and only stops responding,
# which is what drove call sites to paint their own grey "off" state by hand.
DANGER_BUTTON_STYLESHEET = f"""
            QPushButton {{
                background-color: {SEMANTIC_ERROR_COLOR};
                color: white;
                border: none;
                padding: 5px 12px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {SEMANTIC_ERROR_HOVER_COLOR};
            }}
            QPushButton:pressed {{
                background-color: {SEMANTIC_ERROR_PRESSED_COLOR};
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_BG_COLOR};
                color: {DISABLED_TEXT_COLOR};
            }}
        """

# TODO: no token -- #1565c0, #1976d2
SUPERVISION_STATUS_SUPERVISED_STYLESHEET = f"""
                QPushButton {{
                    background-color: {PRIMARY_COLOR};
                    color: white;
                    border: none;
                    padding: 5px 12px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #1976d2;
                }}
                QPushButton:pressed {{
                    background-color: #1565c0;
                }}
            """

# TODO: no token -- #2e7d32, #388e3c
SUPERVISION_STATUS_AUTOMATED_STYLESHEET = f"""
                QPushButton {{
                    background-color: {OK_COLOR};
                    color: white;
                    border: none;
                    padding: 5px 12px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #388e3c;
                }}
                QPushButton:pressed {{
                    background-color: #2e7d32;
                }}
            """

PRIMARY_BUTTON_STYLESHEET = f"""
    QPushButton {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border: none;
        padding: 5px 12px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background-color: {PRIMARY_COLOR_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {PRIMARY_COLOR_PRESSED};
    }}
    QPushButton:disabled {{
        background-color: {DISABLED_BG_COLOR};
        color: {DISABLED_TEXT_COLOR};
    }}
"""

# TODO: no token -- #4a5168
SECONDARY_BUTTON_STYLESHEET = f"""
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
"""

MESSAGE_BOX_STYLESHEET = f"""
    QMessageBox {{
        background-color: {SURFACE_COLOR};
    }}
    QMessageBox QLabel {{
        color: {TEXT_COLOR};
        font-size: 12px;
    }}
    QMessageBox QLabel#qt_msgbox_label {{
        color: {TEXT_STRONG_COLOR};
        font-size: 13px;
        font-weight: 500;
    }}
"""
# The #qt_msgbox_label id is Qt's own name for the headline label, so it is only
# a lift on top of the rule above -- if a future Qt renames it the text stays
# readable rather than disappearing.

STATUS_BAR_STYLESHEET = f"""
                QStatusBar {{
                    background-color: {PANEL_COLOR};
                        color: {TEXT_COLOR};
                        border-top: 1px solid {BORDER_COLOR};
                    }}
                """


INDETERMINATE_PROGRESS_BAR_STYLESHEET = f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                text-align: center;
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR};
            }}
        """

# The tooltip rule, for restating alongside a selector-less widget stylesheet.
#
# A stylesheet set on a widget with no selector applies to every type Qt renders
# through it -- including the QToolTip shown over it -- and it wins over the
# application sheet because it sits nearer. So `background: transparent` on a
# table cell leaves that cell's tooltip as floating text with no panel behind it.
# Prefer custom_widgets.style_with_tooltip() to using this constant directly.
TOOLTIP_STYLESHEET = f"""
QToolTip {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    padding: 4px 8px;
    border-radius: 3px;
}}
"""

# TODO: no token -- #BF00FF
WORKFLOW_BORDER_STYLESHEET = f"""
    QFrame#workflow_border_frame[borderState="idle"]       {{ border: 4px solid {SURFACE_COLOR}; }}
    QFrame#workflow_border_frame[borderState="automated"]  {{ border: 4px solid {OK_COLOR}; }}
    QFrame#workflow_border_frame[borderState="supervised"] {{ border: 4px solid {PRIMARY_COLOR}; }}
    QFrame#workflow_border_frame[borderState="waiting"]    {{ border: 4px solid {ORANGE_COLOR}; }}
    QFrame#workflow_border_frame[borderState="finished"]  {{ border: 4px solid {OK_COLOR}; }}
    QFrame#workflow_border_frame[borderState="stopped"]   {{ border: 4px solid {SEMANTIC_ERROR_COLOR}; }}
    QFrame#workflow_border_frame[borderState="agent"]     {{ border: 4px solid #BF00FF; }}
"""

# TODO: no token -- #6a6a6a, #8a8a8a
TOOLBUTTON_ICON_STYLESHEET = f"""
    QToolButton {{
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 2px 6px;
        background-color: transparent;
    }}
    QToolButton:hover {{
        border: 1px solid {NEUTRAL_650};
        background-color: rgba(255, 255, 255, 25);
    }}
    QToolButton:checked {{
        border: 1px solid #8a8a8a;
        background-color: rgba(255, 255, 255, 35);
    }}
"""

# TODO: no token -- #2d3f5c, #3a3d42
LIST_WIDGET_STYLESHEET = f"""
            QListWidget {{
                background: {SURFACE_COLOR};
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                background: {SURFACE_COLOR};
                border-bottom: 1px solid #3a3d42;
            }}
            QListWidget::item:selected {{
                background: #2d3f5c;
            }}
        """

# QDateTimeEdit with visible up/down step arrows (calendar popup must be OFF for
# the buttons to appear). Stepping applies to whichever field has focus
# (year / month / day / hour / minute), also via mouse-wheel and arrow keys.
# TODO: no token -- #4a5168
DATETIME_EDIT_STYLESHEET = f"""
QDateTimeEdit {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 4px 8px;
}}

QDateTimeEdit:focus {{
    border: 1px solid {ACCENT_COLOR};
}}

QDateTimeEdit:disabled {{
    color: {DISABLED_TEXT_COLOR};
    background-color: {DISABLED_BG_COLOR};
}}

QDateTimeEdit::up-button {{
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

QDateTimeEdit::up-button:hover {{
    background-color: #4a5168;
}}

QDateTimeEdit::up-button:pressed {{
    background-color: {ACCENT_COLOR};
}}

QDateTimeEdit::up-arrow {{
    image: url("__ICONS_DIR__/plus.svg");
    width: 10px;
    height: 10px;
}}

QDateTimeEdit::down-button {{
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

QDateTimeEdit::down-button:hover {{
    background-color: #4a5168;
}}

QDateTimeEdit::down-button:pressed {{
    background-color: {ACCENT_COLOR};
}}

QDateTimeEdit::down-arrow {{
    image: url("__ICONS_DIR__/minus.svg");
    width: 10px;
    height: 10px;
}}
""".replace("__ICONS_DIR__", _ICONS_DIR)

# ---------------------------------------------------------------------------
# Deprecated aliases.
#
# These three were named for the first thing that used them, and then used for
# everything else. Only one of the progress bar's four consumers was milling;
# the "run workflow" green ended up on Load Images, Set Objective and Acquire;
# the "stop workflow" red ended up on Delete Position and Retract Manipulator.
#
# Kept so an out-of-tree plugin importing the old name still works. Prefer the
# names above in new code.
MILLING_PROGRESS_BAR_STYLESHEET = PROGRESS_BAR_STYLESHEET
RUN_WORKFLOW_BUTTON_STYLESHEET = CONFIRM_BUTTON_STYLESHEET
STOP_WORKFLOW_BUTTON_STYLESHEET = DANGER_BUTTON_STYLESHEET
